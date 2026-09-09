"""PyTorch Geometric AttentiveFP baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn.models import AttentiveFP


class AttentiveFPRegressor(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, *, dropout: float = 0.1, output_size: int = 3):
        super().__init__()
        self.model = AttentiveFP(
            in_channels=atom_dim,
            hidden_channels=128,
            out_channels=output_size,
            edge_dim=bond_dim,
            num_layers=3,
            num_timesteps=2,
            dropout=dropout,
        )

    def forward(self, batch):
        edge_attr = batch.edge_attr
        return self.model(batch.x, batch.edge_index, edge_attr, batch.batch)
