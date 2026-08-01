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
    ):
        super().__init__()
        del task_name, atoms_input_dropout_rate, atoms_reshape_dim, device
        self.backbone = MolecularGraphormerBackbone(
            hidden_dim=atoms_hidden_dim,
            num_heads=atoms_num_heads,
            num_layers=atoms_embedders_num,
            ffn_dim=atoms_ffn_dim,
            dropout=max(float(atoms_dropout_rate), float(atoms_attention_dropout_rate)),
            spatial_pos_max_clip=20,
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
        )

    def forward(self, inputs, task_name=None, return_all_tasks=False, mode=None):
        del mode
        representation = self.encoder(inputs)
        if return_all_tasks or task_name is None:
            return {task: self.decoders[task](representation) for task in self.task_name}
        if task_name not in self.decoders:
            raise KeyError(f"Unknown task: {task_name}")
        return {task_name: self.decoders[task_name](representation)}
