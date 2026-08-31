"""CARD: Counterfactual Animal Representation Distillation (D6 plan §15-§18).

A tiny zero-init residual adapter on the pooled molecular representation plus
a train-only distillation loss that aligns the adapter output with the
real-vs-shuffled animal teacher representation delta:

    a(x)      = up(gelu(down(h)))
    h'        = h + a(x)
    L_delta   = 1 - cos(a(x), stopgrad(delta_h(x)))

The teachers are deleted at inference; only the adapter remains, and because
its output layer is zero-initialised the model starts exactly at the HPS
baseline.

Safety contracts (review P1-1):
- Sample alignment is positional-exact: each retained batch position is
  paired with its own table position, so a missing id can never shift the
  teacher delta onto the wrong molecule.
- In training mode every batch sample_id must exist in the delta table
  (fail-fast); in eval mode missing ids simply skip the distillation term.
- The delta table is validated at load time (unique ids, 2-D, width equals
  hidden dim, all finite).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CardAdapter(nn.Module):
    """Zero-init bottleneck residual adapter with counterfactual distillation."""

    def __init__(self, hidden_dim: int, bottleneck: int, lambda_delta: float, delta_table_path: str):
        super().__init__()
        if lambda_delta < 0:
            raise ValueError("card lambda_delta must be >= 0")
        self.lambda_delta = float(lambda_delta)
        self.down = nn.Linear(hidden_dim, int(bottleneck))
        self.up = nn.Linear(int(bottleneck), hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        table = np.load(delta_table_path, allow_pickle=False)
        delta_ids = [str(value) for value in table["ids"]]
        if len(set(delta_ids)) != len(delta_ids):
            raise ValueError("CARD delta table contains duplicate sample ids")
        delta = np.asarray(table["delta"], dtype=np.float32)
        if delta.ndim != 2:
            raise ValueError(f"CARD delta table must be 2-D, got shape {delta.shape}")
        if delta.shape[0] != len(delta_ids):
            raise ValueError(
                f"CARD delta table rows ({delta.shape[0]}) != ids ({len(delta_ids)})"
            )
        if delta.shape[1] != int(hidden_dim):
            raise ValueError(
                f"CARD delta width {delta.shape[1]} != model hidden_dim {hidden_dim}"
            )
        if not np.isfinite(delta).all():
            raise ValueError("CARD delta table contains non-finite values")

        self.delta_ids = delta_ids
        self.id_to_index = {sample_id: index for index, sample_id in enumerate(self.delta_ids)}
        # Registered as a buffer so .to(device) moves it with the module; it is
        # a constant target (stopgrad by construction — no gradient path).
        self.register_buffer("delta_table", torch.tensor(delta))

        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        self.epoch_stats = {
            "loss_sum": 0.0,
            "loss_count": 0,
            "adapter_norm_sum": 0.0,
            "teacher_delta_norm_sum": 0.0,
            "cos_sum": 0.0,
            "valid_target_count": 0,
            "zero_target_count": 0,
        }

    def pop_epoch_stats(self) -> dict:
        count = max(self.epoch_stats["loss_count"], 1)
        stats = {
            "card_loss_mean": self.epoch_stats["loss_sum"] / count,
            "card_adapter_norm_mean": self.epoch_stats["adapter_norm_sum"] / count,
            "card_teacher_delta_norm_mean": self.epoch_stats["teacher_delta_norm_sum"] / count,
            "card_cos_mean": self.epoch_stats["cos_sum"] / count,
            "card_batches": self.epoch_stats["loss_count"],
            "card_valid_target_fraction": (
                self.epoch_stats["valid_target_count"]
                / max(self.epoch_stats["valid_target_count"] + self.epoch_stats["zero_target_count"], 1)
            ),
            "card_zero_target_count": self.epoch_stats["zero_target_count"],
        }
        self.reset_epoch_stats()
        return stats

    def forward(self, representation: torch.Tensor, sample_ids) -> torch.Tensor:
        adapter_output = self.up(torch.nn.functional.gelu(self.down(representation)))
        representation = representation + adapter_output

        sample_ids = [str(sample_id) for sample_id in (sample_ids or [])]
        pairs = [
            (batch_position, self.id_to_index.get(sample_id))
            for batch_position, sample_id in enumerate(sample_ids)
        ]
        # Review P1-6: a collator fault that drops sample ids would silently
        # disable the distillation term — fail fast instead.
        if self.training and len(sample_ids) != representation.size(0):
            raise RuntimeError(
                "CARD training requires one sample_id per representation row: "
                f"ids={len(sample_ids)} batch={representation.size(0)}"
            )
        if self.training:
            missing = [
                sample_id
                for batch_position, table_position in pairs
                if table_position is None
                for sample_id in [sample_ids[batch_position]]
            ]
            if missing:
                # Review P1-1/§22: never let a training batch silently skip or
                # misalign its distillation target.
                raise KeyError(
                    f"CARD delta table missing {len(missing)} training samples; "
                    f"examples={missing[:5]}"
                )
            valid_pairs = pairs
        else:
            # Eval/inference: the adapter prediction is enough; missing ids
            # simply contribute no distillation term.
            valid_pairs = [pair for pair in pairs if pair[1] is not None]

        if valid_pairs:
            batch_positions = torch.tensor(
                [pair[0] for pair in valid_pairs], device=adapter_output.device, dtype=torch.long
            )
            table_positions = torch.tensor(
                [pair[1] for pair in valid_pairs], device=adapter_output.device, dtype=torch.long
            )
            aligned = adapter_output[batch_positions]
            raw_delta = self.delta_table[table_positions]
            teacher_delta_norm = raw_delta.norm(dim=-1)

            # Review P0-1: cosine on a zero-init adapter produces a ~1e12
            # gradient singularity (F.normalize default eps=1e-12).  Train on a
            # unit-direction finite MSE instead: finite at aligned == 0, with an
            # initial per-sample loss ~1 so lambda_delta keeps its intended
            # relative weight.  Cosine stays a read-only diagnostic.
            valid_target = teacher_delta_norm > 1e-8
            if bool(valid_target.any()):
                aligned_valid = aligned[valid_target]
                target_valid = F.normalize(raw_delta[valid_target], dim=-1, eps=1e-8)
                per_sample_delta_loss = (aligned_valid - target_valid).pow(2).sum(dim=-1)
                loss = per_sample_delta_loss.mean()
                with torch.no_grad():
                    cosine = F.cosine_similarity(
                        aligned_valid, target_valid, dim=-1, eps=1e-8
                    )
            else:
                loss = representation.new_zeros(())
                cosine = torch.zeros(0, device=representation.device)
        else:
            raw_delta = self.delta_table[:0]
            teacher_delta_norm = raw_delta.new_zeros(())
            cosine = raw_delta.new_zeros(())
            loss = representation.new_zeros(())

        self.last_loss = loss
        # Review P1-1: only TRAINING forwards accumulate the epoch stats.
        # Validation would otherwise pollute card_loss_mean / card_cos_mean
        # after the trainer pops the per-epoch stats.
        if self.training:
            with torch.no_grad():
                self.epoch_stats["loss_sum"] += float(loss.detach())
                self.epoch_stats["loss_count"] += 1
                self.epoch_stats["adapter_norm_sum"] += float(
                    adapter_output.detach().norm(dim=-1).mean()
                )
                self.epoch_stats["teacher_delta_norm_sum"] += float(teacher_delta_norm.mean())
                self.epoch_stats["cos_sum"] += float(cosine.mean()) if cosine.numel() else 0.0
                self.epoch_stats["valid_target_count"] += int(valid_target.sum())
                self.epoch_stats["zero_target_count"] += int(
                    valid_target.numel() - valid_target.sum()
                )
        return representation
