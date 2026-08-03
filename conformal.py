"""Task-wise split conformal calibration for quantile predictions."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, Mapping

import torch


@dataclass
class TaskConformalState:
    alpha: float
    qhat: float
    count: int


class ConformalCalibrator:
    def __init__(self, alpha: float = 0.10, min_calibration_size: int = 30):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if min_calibration_size < 0:
            raise ValueError("min_calibration_size must be non-negative")
        self.alpha = float(alpha)
        self.min_calibration_size = int(min_calibration_size)
        self.states: Dict[str, TaskConformalState] = {}

    @staticmethod
    def conformity_scores(lower, upper, target):
        lower = torch.as_tensor(lower)
        upper = torch.as_tensor(upper)
        target = torch.as_tensor(target)
        # CQR uses the signed conformity score.  Negative scores are valid:
        # they allow an over-wide base interval to contract after calibration.
        return torch.maximum(lower - target, target - upper)

    def fit_task(self, task, lower, upper, target):
        scores = self.conformity_scores(lower, upper, target).reshape(-1)
        scores = scores[torch.isfinite(scores)]
        count = int(scores.numel())
        if count == 0:
            raise ValueError(f"No calibration samples for {task}.")
        if count < self.min_calibration_size:
            warnings.warn(
                f"Calibration task {task!r} has only {count} samples; requested minimum is "
                f"{self.min_calibration_size}.",
                RuntimeWarning,
                stacklevel=2,
            )
        rank = min(max(math.ceil((count + 1) * (1.0 - self.alpha)), 1), count)
        qhat = float(torch.kthvalue(scores, rank).values.detach().cpu())
        self.states[str(task)] = TaskConformalState(self.alpha, qhat, count)
        return self.states[str(task)]

    def fit(self, calibration_records: Mapping[str, Mapping[str, torch.Tensor]]):
        for task, record in calibration_records.items():
            self.fit_task(task, record["lower"], record["upper"], record["target"])
        return self

    def apply(self, task, lower, upper):
        if task not in self.states:
            raise KeyError(f"No conformal state fitted for task {task!r}")
        qhat = self.states[task].qhat
        return torch.as_tensor(lower) - qhat, torch.as_tensor(upper) + qhat

    def state_dict(self):
        return {
            "alpha": self.alpha,
            "min_calibration_size": self.min_calibration_size,
            "states": {task: asdict(state) for task, state in self.states.items()},
        }

    def load_state_dict(self, state):
        if state is None:
            self.states = {}
            return
        if float(state.get("alpha", self.alpha)) != self.alpha:
            raise ValueError("Conformal alpha does not match the current configuration")
        self.states = {
            task: TaskConformalState(**values) for task, values in state.get("states", {}).items()
        }


__all__ = ["ConformalCalibrator", "TaskConformalState"]
