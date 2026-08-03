"""Compatibility shim for the retired visualization entry point.

Old Prompt/Captum attribution code was intentionally removed.  New analyses
should import from :mod:`analysis`; this module only re-exports the routing and
interval helpers so older notebooks fail gracefully without reviving the old
Prompt self-attention assumptions.
"""

from analysis import (
    collect_routing_records,
    coverage_width,
    risk_coverage_curve,
    source_deletion_analysis,
    summarize_risk_coverage,
)

__all__ = [
    "collect_routing_records",
    "coverage_width",
    "risk_coverage_curve",
    "source_deletion_analysis",
    "summarize_risk_coverage",
]
