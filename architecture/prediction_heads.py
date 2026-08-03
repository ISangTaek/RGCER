"""Shared point and quantile prediction heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DecodedPrediction:
    median: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {"median": self.median, "lower": self.lower, "upper": self.upper}


class TaskPredictionHead(nn.Module):
    """A fair, task-independent point or median/width prediction head."""

    VALID_MODES = {"point", "quantile"}

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "quantile",
        head_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError("mode must be 'point' or 'quantile'")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if head_hidden_dim is not None and head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive")
        self.mode = mode
        output_dim = 1 if mode == "point" else 3
        if head_hidden_dim is None:
            self.network = nn.Linear(hidden_dim, output_dim)
            final_layer = self.network
        else:
            self.network = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, output_dim),
            )
            final_layer = self.network[-1]
        if mode == "quantile":
            nn.init.constant_(final_layer.bias[1:], -1.0)

    @property
    def output_dim(self) -> int:
        return 1 if self.mode == "point" else 3

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.ndim != 2:
            raise ValueError("representation must have shape [B, D]")
        return self.network(representation)


def point_from_raw(raw: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "point":
        if raw.size(-1) != 1:
            raise ValueError("Point output must have 1 channel")
        return raw[..., :1]
    if mode == "quantile":
        if raw.size(-1) != 3:
            raise ValueError("Quantile output must have 3 channels")
        return raw[..., :1]
    raise ValueError("mode must be 'point' or 'quantile'")


def decode_prediction(raw: torch.Tensor, mode: str) -> DecodedPrediction:
    if mode == "point":
        median = point_from_raw(raw, mode)
        return DecodedPrediction(median=median, lower=median, upper=median)
    if mode != "quantile" or raw.size(-1) != 3:
        raise ValueError("Quantile output must have 3 channels")
    median = raw[..., 0:1]
    lower_width = F.softplus(raw[..., 1:2])
    upper_width = F.softplus(raw[..., 2:3])
    return DecodedPrediction(
        median=median,
        lower=median - lower_width,
        upper=median + upper_width,
    )


__all__ = ["DecodedPrediction", "TaskPredictionHead", "decode_prediction", "point_from_raw"]
