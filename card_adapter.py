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
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CardAdapter(nn.Module):
    """Zero-init bottleneck residual adapter with counterfactual distillation."""

    def __init__(self, hidden_dim: int, bottleneck: int, lambda_delta: float, delta_table_path: str):
        super().__init__()
        self.lambda_delta = float(lambda_delta)
        self.down = nn.Linear(hidden_dim, int(bottleneck))
        self.up = nn.Linear(int(bottleneck), hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        table = np.load(delta_table_path, allow_pickle=False)
        self.delta_ids = [str(value) for value in table["ids"]]
        self.id_to_index = {sample_id: index for index, sample_id in enumerate(self.delta_ids)}
        # Registered as a buffer so .to(device) moves it with the module; it is
        # a constant target (stopgrad by construction — no gradient path).
        self.register_buffer(
            "delta_table", torch.tensor(np.asarray(table["delta"], dtype=np.float32))
        )
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        self.epoch_stats = {
            "loss_sum": 0.0,
            "loss_count": 0,
            "adapter_norm_sum": 0.0,
            "teacher_delta_norm_sum": 0.0,
            "cos_sum": 0.0,
        }

    def pop_epoch_stats(self) -> dict:
        count = max(self.epoch_stats["loss_count"], 1)
        stats = {
            "card_loss_mean": self.epoch_stats["loss_sum"] / count,
            "card_adapter_norm_mean": self.epoch_stats["adapter_norm_sum"] / count,
            "card_teacher_delta_norm_mean": self.epoch_stats["teacher_delta_norm_sum"] / count,
            "card_cos_mean": self.epoch_stats["cos_sum"] / count,
            "card_batches": self.epoch_stats["loss_count"],
        }
        self.reset_epoch_stats()
        return stats

    def forward(self, representation: torch.Tensor, sample_ids) -> torch.Tensor:
        adapter_output = self.up(torch.nn.functional.gelu(self.down(representation)))
        representation = representation + adapter_output

        indices = [self.id_to_index.get(str(sample_id)) for sample_id in (sample_ids or [])]
        valid = [index for index in indices if index is not None]
        if valid:
            delta = self.delta_table[torch.tensor(valid, device=adapter_output.device)]
            delta = torch.nn.functional.normalize(delta, dim=-1)
            aligned = adapter_output[: len(valid)]
            aligned = torch.nn.functional.normalize(aligned, dim=-1)
            cosine = (aligned * delta).sum(dim=-1)
            loss = (1.0 - cosine).mean()
        else:
            loss = adapter_output.new_zeros(())

        self.last_loss = loss
        with torch.no_grad():
            self.epoch_stats["loss_sum"] += float(loss.detach())
            self.epoch_stats["loss_count"] += 1
            self.epoch_stats["adapter_norm_sum"] += float(adapter_output.detach().norm(dim=-1).mean())
            self.epoch_stats["teacher_delta_norm_sum"] += (
                float(delta.norm(dim=-1).mean()) if valid else 0.0
            )
            self.epoch_stats["cos_sum"] += float(cosine.mean()) if valid else 0.0
        return representation
