"""Per-task label scaling with explicit empty-task semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskScaler:
    means: np.ndarray
    stds: np.ndarray
    counts: np.ndarray
    task_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        labels: np.ndarray,
        task_names: tuple[str, ...] | list[str],
        *,
        allow_empty: bool,
    ) -> "TaskScaler":
        values = np.asarray(labels, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(task_names):
            raise ValueError("labels must be [samples, tasks] and match task_names")
        means = np.zeros(values.shape[1], dtype=np.float64)
        stds = np.ones(values.shape[1], dtype=np.float64)
        counts = np.zeros(values.shape[1], dtype=np.int64)
        for column in range(values.shape[1]):
            finite = values[np.isfinite(values[:, column]), column]
            counts[column] = finite.size
            if finite.size == 0:
                if not allow_empty:
                    raise ValueError(f"No finite training labels for task {task_names[column]}")
                continue
            means[column] = float(np.mean(finite))
            stds[column] = max(float(np.std(finite, ddof=0)), 1e-6)
        return cls(means, stds, counts, tuple(task_names))

    def transform(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(labels, dtype=np.float32)
        if values.shape[-1] != len(self.task_names):
            raise ValueError("label width does not match scaler")
        mask = np.isfinite(values)
        filled = np.where(mask, values, self.means).astype(np.float32)
        scaled = (filled - self.means.astype(np.float32)) / self.stds.astype(np.float32)
        return scaled.astype(np.float32), mask.astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return array * self.stds + self.means

    def to_dict(self) -> dict:
        return {
            "task_names": list(self.task_names),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "counts": self.counts.tolist(),
            "ddof": 0,
            "min_std": 1e-6,
            "empty_task_policy": "mean=0,std=1,count=0",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskScaler":
        return cls(
            np.asarray(payload["means"], dtype=np.float64),
            np.asarray(payload["stds"], dtype=np.float64),
            np.asarray(payload["counts"], dtype=np.int64),
            tuple(payload["task_names"]),
        )
