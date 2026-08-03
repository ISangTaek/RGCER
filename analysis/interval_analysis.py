"""Coverage-width and risk-coverage summaries for calibrated intervals."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch


def _flat(value):
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def coverage_width(lower, upper, target, alpha: Optional[float] = None) -> dict[str, float]:
    """Compute empirical coverage and interval width for one task."""

    lower, upper, target = _flat(lower), _flat(upper), _flat(target)
    if not (lower.numel() == upper.numel() == target.numel()):
        raise ValueError("lower, upper, and target must contain the same number of samples")
    if lower.numel() == 0:
        raise ValueError("coverage_width requires at least one sample")
    if torch.any(upper < lower):
        raise ValueError("upper must be greater than or equal to lower")
    covered = (target >= lower) & (target <= upper)
    widths = upper - lower
    result = {
        "coverage": float(covered.float().mean()),
        "mean_width": float(widths.mean()),
        "median_width": float(widths.median()),
        "count": float(target.numel()),
    }
    if alpha is not None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        result["target_coverage"] = 1.0 - float(alpha)
        result["coverage_gap"] = result["coverage"] - result["target_coverage"]
    return result


def risk_coverage_curve(
    prediction,
    target,
    uncertainty,
    points: int = 20,
) -> pd.DataFrame:
    """Return selective risk as coverage increases from most to least certain.

    ``uncertainty`` can be interval width, absolute residual proxy, or any
    scalar score where smaller values indicate more reliable predictions.
    """

    prediction, target, uncertainty = _flat(prediction), _flat(target), _flat(uncertainty)
    if not (prediction.numel() == target.numel() == uncertainty.numel()):
        raise ValueError("prediction, target, and uncertainty must have the same number of samples")
    if prediction.numel() == 0:
        return pd.DataFrame(columns=["coverage", "risk", "retained_count", "threshold"])
    if points <= 0:
        raise ValueError("points must be positive")
    order = torch.argsort(uncertainty, stable=True)
    errors = (prediction - target).abs()[order]
    sorted_uncertainty = uncertainty[order]
    counts = sorted(set(max(1, int(round(prediction.numel() * fraction))) for fraction in np.linspace(1 / points, 1.0, points)))
    rows = []
    for count in counts:
        rows.append(
            {
                "coverage": count / prediction.numel(),
                "risk": float(errors[:count].mean()),
                "retained_count": count,
                "threshold": float(sorted_uncertainty[count - 1]),
            }
        )
    return pd.DataFrame(rows)


def summarize_risk_coverage(curve: pd.DataFrame) -> dict[str, float]:
    """Summarize a risk-coverage curve with endpoint risk and AURC."""

    if curve.empty:
        return {"aurc": float("nan"), "risk_at_full_coverage": float("nan"), "risk_at_half_coverage": float("nan")}
    x = curve["coverage"].to_numpy(dtype=float)
    y = curve["risk"].to_numpy(dtype=float)
    full_index = int(np.argmax(x))
    half_index = int(np.argmin(np.abs(x - 0.5)))
    return {
        "aurc": float(np.trapz(y, x)),
        "risk_at_full_coverage": float(y[full_index]),
        "risk_at_half_coverage": float(y[half_index]),
    }


__all__ = ["coverage_width", "risk_coverage_curve", "summarize_risk_coverage"]
