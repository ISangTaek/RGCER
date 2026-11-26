"""Multi-View Biological Prompt Graph Transformer (MV-BioPGT) package.

Self-contained, teacher-free multi-endpoint acute toxicity modeling with:
- One-time RAM/disk feature & graph preloading from accepted GraphCache
- Train-only robust clipped 2D physicochemical descriptors + Avalon + Morgan
- Bond-aware GINE and Graphormer graph backbones
- Factorized biological endpoint prompting (organism / route / measurement / population)
- Tempered multi-task loss and strict Go/No-Go gate execution
"""

from .features import DESCRIPTOR_NAMES, MVDataStore, build_mv_cache, load_mv_cache
from .models import MVBioPGT, build_mv_model, count_parameters
from .training import (
    compute_group_metrics,
    tempered_huber_loss,
    tempered_mse_loss,
    train_and_evaluate,
)

__all__ = [
    "DESCRIPTOR_NAMES",
    "MVDataStore",
    "MVBioPGT",
    "build_mv_cache",
    "build_mv_model",
    "compute_group_metrics",
    "count_parameters",
    "load_mv_cache",
    "tempered_huber_loss",
    "tempered_mse_loss",
    "train_and_evaluate",
]
