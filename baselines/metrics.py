"""Human3 regression metrics in the original -log10(mol/kg) label units."""

from __future__ import annotations

import numpy as np


def _task_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if np.any(np.isfinite(y_true) & ~np.isfinite(y_pred)):
        raise ValueError("nonfinite prediction at an observed label")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    count = int(mask.sum())
    if count == 0:
        return {"n": 0, "rmse": None, "mae": None, "r2": None, "r2_reason": "no_finite_pairs"}
    truth = y_true[mask].astype(np.float64)
    pred = y_pred[mask].astype(np.float64)
    residual = truth - pred
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    if count < 2:
        r2, reason = None, "fewer_than_two_samples"
    else:
        denominator = float(np.sum((truth - np.mean(truth)) ** 2))
        if denominator <= 0:
            r2, reason = None, "constant_targets"
        else:
            r2, reason = float(1.0 - np.sum(residual * residual) / denominator), None
    return {"n": count, "rmse": rmse, "mae": mae, "r2": r2, "r2_reason": reason}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_names: list[str] | tuple[str, ...]) -> dict:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] != len(task_names):
        raise ValueError("metric arrays must have equal [samples, tasks] shape")
    per_task = {name: _task_metrics(truth[:, i], pred[:, i]) for i, name in enumerate(task_names)}
    rmses = [entry["rmse"] for entry in per_task.values()]
    if any(value is None for value in rmses):
        macro_rmse = None
        macro_reason = "at_least_one_task_has_no_finite_pairs"
    else:
        macro_rmse = float(np.mean(rmses))
        macro_reason = None
    mask = np.isfinite(truth) & np.isfinite(pred)
    pooled_rmse = float(np.sqrt(np.mean((truth[mask] - pred[mask]) ** 2))) if mask.any() else None
    return {
        "selection_metric": "validation_human3_macro_rmse",
        "macro_rmse": macro_rmse,
        "macro_rmse_reason": macro_reason,
        "pooled_rmse": pooled_rmse,
        "pooled_n": int(mask.sum()),
        "per_task": per_task,
    }
