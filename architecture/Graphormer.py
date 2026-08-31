"""Hard-parameter-sharing Graphormer architecture.

The encoder is deliberately stateless: one call consumes one collated batch and
returns one molecular representation.  Task-specific prediction is performed by
the decoder selected by ``task_name``.
"""

from __future__ import annotations

import torch
from torch import nn

from architecture.abstract_arch import AbsArchitecture
from architecture.graphormer_backbone import MolecularGraphormerBackbone


class Encoder(nn.Module):
    """Compatibility wrapper around the shared chemical Graphormer backbone."""

    def __init__(
        self,
        atoms_num_heads,
        task_name,
        atoms_embedders_num,
        atoms_hidden_dim,
        atoms_dropout_rate,
        atoms_input_dropout_rate,
        atoms_ffn_dim,
        atoms_reshape_dim,
        atoms_attention_dropout_rate,
        atoms_readout_dim,
        device,
        edge_bias_mode="path",
        spatial_pos_max_clip=20,
    ):
        super().__init__()
        del task_name, atoms_input_dropout_rate, atoms_reshape_dim, device
        self.backbone = MolecularGraphormerBackbone(
            hidden_dim=atoms_hidden_dim,
            num_heads=atoms_num_heads,
            num_layers=atoms_embedders_num,
            ffn_dim=atoms_ffn_dim,
            dropout=max(float(atoms_dropout_rate), float(atoms_attention_dropout_rate)),
            spatial_pos_max_clip=spatial_pos_max_clip,
            edge_bias_mode=edge_bias_mode,
        )
        if atoms_readout_dim != atoms_hidden_dim:
            self.readout = nn.Linear(atoms_hidden_dim, atoms_readout_dim)
        else:
            self.readout = nn.Identity()

    def forward(self, batch, task_name=None, mode=None):
        del task_name, mode
        return self.readout(self.backbone(batch))


class Graphormer(AbsArchitecture):
    """Shared backbone with independent task decoders (HPS baseline)."""

    def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
        super().__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.encoder = encoder_class(
            atoms_num_heads=args.a_heads,
            task_name=task_name,
            atoms_embedders_num=args.a_layers,
            atoms_hidden_dim=args.hidden_dim,
            atoms_dropout_rate=0.1,
            atoms_input_dropout_rate=0.0,
            atoms_ffn_dim=args.mid_dim,
            atoms_reshape_dim=args.hidden_dim,
            atoms_attention_dropout_rate=0.1,
            atoms_readout_dim=args.hidden_dim,
            device=device,
            edge_bias_mode=getattr(args, "edge_bias_mode", "path"),
            spatial_pos_max_clip=kwargs.get(
                "spatial_pos_max_clip", getattr(args, "spatial_pos_clip", 20)
            ),
        )
        # D6 O3 CARD: zero-init residual adapter + counterfactual distillation
        # (plan §15-§18).  Inert unless --card_lambda_delta > 0.
        self.card = None
        if float(getattr(args, "card_lambda_delta", 0.0)) > 0:
            from card_adapter import CardAdapter

            self.card = CardAdapter(
                hidden_dim=args.hidden_dim,
                bottleneck=int(getattr(args, "card_bottleneck", 32)),
                lambda_delta=float(args.card_lambda_delta),
                delta_table_path=str(getattr(args, "card_delta_table", "") or ""),
            )

    def forward(
        self,
        inputs,
        task_name=None,
        return_all_tasks=False,
        return_aux=False,
        source_mask=None,
        routing_enabled=True,
        mode=None,
    ):
        del source_mask, routing_enabled, mode
        representation = self.encoder(inputs)
        if self.card is not None:
            representation = self.card(
                representation, list(getattr(inputs, "sample_id", None) or [])
            )
        ones = torch.ones(representation.size(0), 1, device=representation.device, dtype=representation.dtype)
        if return_all_tasks or task_name is None:
            predictions = {task: self.decoders[task](representation) for task in self.task_name}
            if return_aux:
                diagnostics = {
                    task: {
                        "target_task": task,
                        "target_index": index,
                        "base_representation": representation,
                        "route_representation": representation,
                        "final_representation": representation,
                        "base_raw": predictions[task],
                        "route_raw": predictions[task],
                        "final_raw": predictions[task],
                        "null_weight": ones,
                    }
                    for index, task in enumerate(self.task_name)
                }
                return predictions, diagnostics
            return predictions
        if task_name not in self.decoders:
            raise KeyError(f"Unknown task: {task_name}")
        raw = self.decoders[task_name](representation)
        predictions = {task_name: raw}
        if return_aux:
            index = self.task_name.index(task_name)
            return predictions, {
                "target_task": task_name,
                "target_index": index,
                "base_representation": representation,
                "route_representation": representation,
                "final_representation": representation,
                "base_raw": raw,
                "route_raw": raw,
                "final_raw": raw,
                "null_weight": ones,
            }
        return predictions
