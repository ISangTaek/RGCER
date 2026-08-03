"""Epoch-level metrics.

Metric objects buffer predictions and labels and calculate one score over the
complete epoch.  This is important for AUROC/AUPRC/RMSE/R2: calculating a
metric per mini-batch and averaging those values is generally not equivalent to
calculating it over the complete evaluation split.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def _flatten(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def compute_classification_metrics(pred, gt):
    pred = _flatten(pred)
    gt = _flatten(gt)
    if pred.size == 0 or gt.size == 0:
        return {"AUROC": np.nan, "AUPRC": np.nan}
    auroc = np.nan
    if np.unique(gt).size >= 2:
        auroc = float(roc_auc_score(gt, pred))
    try:
        auprc = float(average_precision_score(gt, pred))
    except ValueError:
        auprc = np.nan
    return {"AUROC": auroc, "AUPRC": auprc}


def compute_regression_metrics(pred, gt):
    pred = _flatten(pred)
    gt = _flatten(gt)
    if pred.size == 0 or gt.size == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    result = {
        "RMSE": float(np.sqrt(mean_squared_error(gt, pred))),
        "MAE": float(mean_absolute_error(gt, pred)),
        "R2": np.nan,
    }
    if gt.size >= 2 and np.unique(gt).size >= 2:
        result["R2"] = float(r2_score(gt, pred))
    return result


def compute_interval_metrics(lower, upper, target, alpha):
    lower = _flatten(lower)
    upper = _flatten(upper)
    target = _flatten(target)
    if not (lower.size == upper.size == target.size):
        raise ValueError("lower, upper, and target must have the same number of values")
    covered = (target >= lower) & (target <= upper)
    width = upper - lower
    interval_score = (
        width
        + 2.0 / alpha * (lower - target) * (target < lower)
        + 2.0 / alpha * (target - upper) * (target > upper)
    )
    coverage = float(covered.mean()) if covered.size else np.nan
    return {
        "Coverage": coverage,
        "MeanWidth": float(width.mean()) if width.size else np.nan,
        "IntervalScore": float(interval_score.mean()) if width.size else np.nan,
        "CoverageError": abs(coverage - (1.0 - alpha)) if width.size else np.nan,
    }


def compute_transfer_metrics(base_prediction, route_prediction, target, null_weight):
    base_error = np.abs(_flatten(base_prediction) - _flatten(target))
    route_error = np.abs(_flatten(route_prediction) - _flatten(target))
    null_weight = _flatten(null_weight)
    regret = route_error - base_error
    harmful = regret > 0
    result = {
        "RouteRegretMean": float(regret.mean()) if regret.size else np.nan,
        "NegativeTransferRate": float(harmful.mean()) if regret.size else np.nan,
        "route_regret": regret,
    }
    if regret.size and np.unique(harmful).size == 2:
        from sklearn.metrics import roc_auc_score

        result["NullHarmfulAUROC"] = float(roc_auc_score(harmful.astype(int), 1.0 - null_weight))
    else:
        result["NullHarmfulAUROC"] = np.nan
    if regret.size >= 2 and np.std(null_weight) > 0 and np.std(regret) > 0:
        from scipy.stats import spearmanr

        result["NullRegretSpearman"] = float(spearmanr(null_weight, regret).statistic)
    else:
        result["NullRegretSpearman"] = np.nan
    return result


class ClsMetric:
    """Buffered AUROC and AUPRC for one task."""

    def __init__(self):
        self.predictions = []
        self.labels = []
        # Legacy attributes are kept for callers that inspect metric objects.
        self.auroc_record = []
        self.auprc_record = []
        self.bs = []

    def update_fun(self, pred, gt):
        pred = _flatten(pred)
        gt = _flatten(gt)
        self.predictions.append(pred)
        self.labels.append(gt)
        self.bs.append(pred.size)

    def score_fun(self):
        if not self.predictions:
            return [np.nan, np.nan]
        result = compute_classification_metrics(
            np.concatenate(self.predictions), np.concatenate(self.labels)
        )
        return [result["AUROC"], result["AUPRC"]]

    def reinit(self):
        self.predictions.clear()
        self.labels.clear()
        self.auroc_record.clear()
        self.auprc_record.clear()
        self.bs.clear()


class RegMetric:
    """Buffered RMSE and R2 for one task."""

    def __init__(self):
        self.predictions = []
        self.labels = []
        self.rmse_record = []
        self.r2_record = []
        self.bs = []

    def update_fun(self, pred, gt):
        pred = _flatten(pred)
        gt = _flatten(gt)
        self.predictions.append(pred)
        self.labels.append(gt)
        self.bs.append(pred.size)

    def score_fun(self):
        if not self.predictions:
            return [np.nan, np.nan]
        result = compute_regression_metrics(
            np.concatenate(self.predictions), np.concatenate(self.labels)
        )
        return [result["RMSE"], result["R2"]]

    def reinit(self):
        self.predictions.clear()
        self.labels.clear()
        self.rmse_record.clear()
        self.r2_record.clear()
        self.bs.clear()
