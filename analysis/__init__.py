"""Analysis helpers for model diagnostics."""

from .routing_analysis import (
    collect_routing_records,
    make_source_mask,
    mean_routing_weights,
    routing_summary,
    top_k_routing,
)
from .negative_transfer import (
    intervention_masks,
    shuffled_endpoint_batches,
    shuffled_endpoint_name,
    shuffle_endpoint_batch,
    source_deletion_analysis,
)
from .interval_analysis import coverage_width, risk_coverage_curve, summarize_risk_coverage

__all__ = [
    "collect_routing_records",
    "make_source_mask",
    "mean_routing_weights",
    "routing_summary",
    "top_k_routing",
    "coverage_width",
    "intervention_masks",
    "risk_coverage_curve",
    "shuffled_endpoint_batches",
    "shuffled_endpoint_name",
    "shuffle_endpoint_batch",
    "source_deletion_analysis",
    "summarize_risk_coverage",
]
