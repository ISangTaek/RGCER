"""Prediction losses used by baselines and RGCER."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss:
    def __init__(self):
        self.record = []
        self.bs = []
        self.loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)

    def update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size(0))
        return loss

    def average_loss(self):
        record = np.asarray(self.record)
        bs = np.asarray(self.bs)
        return (record * bs).sum() / bs.sum()

    def reinit(self):
        self.record = []
        self.bs = []


class MSELoss:
    def __init__(self):
        self.record = []
        self.bs = []
        self.loss_fn = nn.MSELoss()

    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt)

    def update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size(0))
        return loss

    def average_loss(self):
        record = np.asarray(self.record)
        bs = np.asarray(self.bs)
        return (record * bs).sum() / bs.sum()

    def reinit(self):
        self.record = []
        self.bs = []


def pinball_loss(prediction, target, quantile):
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


class QuantileRegressionLoss:
    """Median Huber loss plus lower/upper pinball losses."""

    def __init__(
        self,
        lower_quantile=0.05,
        upper_quantile=0.95,
        median_weight=1.0,
        quantile_weight=1.0,
        huber_delta=1.0,
    ):
        if not 0.0 < lower_quantile < upper_quantile < 1.0:
            raise ValueError("lower_quantile and upper_quantile must satisfy 0 < lower < upper < 1")
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.median_weight = float(median_weight)
        self.quantile_weight = float(quantile_weight)
        self.huber_delta = float(huber_delta)

    def compute_loss(self, raw, target):
        if raw.ndim != 2 or raw.size(-1) != 3:
            raise ValueError("Quantile raw output must have shape [B, 3]")
        from architecture.prediction_heads import decode_prediction

        decoded = decode_prediction(raw, mode="quantile")
        median_loss = F.huber_loss(decoded.median, target, delta=self.huber_delta)
        lower_loss = pinball_loss(decoded.lower, target, self.lower_quantile)
        upper_loss = pinball_loss(decoded.upper, target, self.upper_quantile)
        return self.median_weight * median_loss + self.quantile_weight * (lower_loss + upper_loss)


__all__ = ["BCELoss", "MSELoss", "QuantileRegressionLoss", "pinball_loss"]
