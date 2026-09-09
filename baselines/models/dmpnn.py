"""Chemprop 1.6.1 D-MPNN adapter using its native graph featurizer."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from ..constants import CHEMPROP_COMMIT, CHEMPROP_VERSION


def activate_chemprop(source_root: str | Path) -> None:
    root = str(Path(source_root).resolve())
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != CHEMPROP_COMMIT:
        raise RuntimeError(f"Expected Chemprop source commit {CHEMPROP_COMMIT}, found {commit}")
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("chemprop")
    if str(getattr(module, "__version__", "")) != CHEMPROP_VERSION:
        raise RuntimeError(f"Expected chemprop {CHEMPROP_VERSION}, found {getattr(module, '__version__', None)}")


def dmpnn_args(device: torch.device, dropout: float = 0.1) -> SimpleNamespace:
    return SimpleNamespace(
        atom_messages=False,
        hidden_size=300,
        bias=False,
        depth=3,
        undirected=False,
        device=device,
        aggregation="mean",
        aggregation_norm=100,
        is_atom_bond_targets=False,
        dropout=dropout,
        activation="ReLU",
        atom_descriptors=None,
        atom_descriptors_size=0,
        bond_descriptors=None,
        bond_descriptors_size=0,
    )


class ChempropDMPNN(nn.Module):
    def __init__(self, source_root: str | Path, device: torch.device, *, dropout: float = 0.1, output_size: int = 3):
        super().__init__()
        activate_chemprop(source_root)
        from chemprop.features import get_atom_fdim, get_bond_fdim
        from chemprop.models.ffn import build_ffn
        from chemprop.models.mpn import MPNEncoder
        from chemprop.nn_utils import initialize_weights

        args = dmpnn_args(device, dropout)
        self.encoder = MPNEncoder(args, get_atom_fdim(), get_bond_fdim(atom_messages=False))
        self.head = build_ffn(
            first_linear_dim=300,
            hidden_size=300,
            num_layers=2,
            output_size=output_size,
            dropout=dropout,
            activation="ReLU",
            dataset_type="regression",
        )
        initialize_weights(self)

    def forward(self, smiles: list[str]) -> torch.Tensor:
        from chemprop.features import mol2graph

        graph = mol2graph(smiles)
        return self.head(self.encoder(graph))


class NoamLikeScheduler:
    """Chemprop 1.6.1 NoamLR formula with explicit step-zero handling."""

    def __init__(self, optimizer, *, warmup_epochs: float, total_epochs: int, steps_per_epoch: int,
                 init_lr: float, max_lr: float, final_lr: float):
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
        self.total_steps = max(self.warmup_steps, int(total_epochs * steps_per_epoch))
        self.init_lr, self.max_lr, self.final_lr = map(float, (init_lr, max_lr, final_lr))
        self.current_step = 0
        self._set(self.init_lr)

    def _set(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> float:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.init_lr + (self.max_lr - self.init_lr) * self.current_step / self.warmup_steps
        elif self.current_step <= self.total_steps:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.max_lr * (self.final_lr / self.max_lr) ** progress
        else:
            lr = self.final_lr
        self._set(lr)
        return lr
