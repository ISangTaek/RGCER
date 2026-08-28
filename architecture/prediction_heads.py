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
    """A fair, task-independent point or median/width prediction head.

    ``forward(..., deterministic=True)`` suppresses the hidden dropout for
    that single call — used by the RGCER response profile so the router sees
    stable preliminary responses in train mode. The default call keeps
    ordinary supervised dropout behaviour untouched.
    """

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
            self.dropout_probability = 0.0
        else:
            self.network = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, output_dim),
            )
            final_layer = self.network[-1]
            self.dropout_probability = float(dropout)
        if mode == "quantile":
            nn.init.constant_(final_layer.bias[1:], -1.0)

    @property
    def output_dim(self) -> int:
        return 1 if self.mode == "point" else 3

    def forward(
        self,
        representation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if representation.ndim != 2:
            raise ValueError("representation must have shape [B, D]")
        if not deterministic or self.dropout_probability <= 0:
            return self.network(representation)
        # Functional replay of the Sequential with the hidden dropout layer
        # pinned off for this single call (train mode included).
        stem = nn.Sequential(*list(self.network)[:3])
        final = self.network[-1]
        return final(stem(representation))


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


def blend_decoded_predictions(
    base: DecodedPrediction,
    route: DecodedPrediction,
    null_weight: torch.Tensor,
) -> DecodedPrediction:
    """Blend HPS and routed predictions in prediction space.

    ``null_weight`` is the probability assigned to the explicit no-transfer
    route.  Blending the decoded values, rather than representations before a
    nonlinear prediction head, makes the fallback probability directly
    interpretable and preserves quantile ordering.
    """

    if null_weight.ndim != 2 or null_weight.size(-1) != 1:
        raise ValueError("null_weight must have shape [B, 1]")
    if base.median.shape != route.median.shape:
        raise ValueError("base and route predictions must have matching shapes")
    if base.lower.shape != base.median.shape or base.upper.shape != base.median.shape:
        raise ValueError("base prediction tensors must have matching shapes")
    if route.lower.shape != route.median.shape or route.upper.shape != route.median.shape:
        raise ValueError("route prediction tensors must have matching shapes")
    if null_weight.shape[0] != base.median.shape[0]:
        raise ValueError("null_weight batch dimension must match predictions")

    w = null_weight
    return DecodedPrediction(
        median=w * base.median + (1.0 - w) * route.median,
        lower=w * base.lower + (1.0 - w) * route.lower,
        upper=w * base.upper + (1.0 - w) * route.upper,
    )


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Differentiable inverse of ``softplus`` for strictly positive values."""

    if torch.any(x <= 0):
        raise ValueError("softplus inverse requires positive input")
    return x + torch.log(-torch.expm1(-x))


def encode_decoded_prediction(decoded: DecodedPrediction, mode: str) -> torch.Tensor:
    """Encode decoded predictions into the raw tensor used by the heads.

    The helper lets the trainer and existing loss/calibration code continue to
    consume the same raw parameterization after a prediction-space blend.
    """

    if mode == "point":
        return decoded.median
    if mode != "quantile":
        raise ValueError("mode must be 'point' or 'quantile'")

    lower_width = decoded.median - decoded.lower
    upper_width = decoded.upper - decoded.median
    if torch.any(lower_width <= 0) or torch.any(upper_width <= 0):
        raise ValueError("Quantile widths must be positive")
    return torch.cat(
        [
            decoded.median,
            _inverse_softplus(lower_width),
            _inverse_softplus(upper_width),
        ],
        dim=-1,
    )


__all__ = [
    "DecodedPrediction",
    "TaskPredictionHead",
    "blend_decoded_predictions",
    "decode_prediction",
    "encode_decoded_prediction",
    "point_from_raw",
]
