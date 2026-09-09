"""Deterministic frozen trial-list generation."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def _unique_random(rng: np.random.RandomState, count: int, sampler) -> list[dict[str, Any]]:
    rows, seen = [], set()
    while len(rows) < count:
        row = sampler()
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def rf_full_grid() -> list[dict[str, Any]]:
    """Return the ordered, frozen 54-combination RF grid."""

    return [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf, "max_features": feature}
        for n, d, leaf, feature in itertools.product(
            (500, 1000), (None, 10, 30), (1, 2, 4), ("sqrt", 0.3, 1.0)
        )
    ]


def generate_trials(method: str, seed: int = 42) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed)
    if method == "rf":
        # Parameter and candidate order are protocol-significant.
        grid = rf_full_grid()
        return [grid[index] for index in rng.choice(len(grid), size=50, replace=False)]
    if method == "afp":
        return _unique_random(rng, 20, lambda: {
            "lr": float(10 ** rng.uniform(-5, -3)),
            "dropout": float(rng.choice((0.0, 0.1, 0.2))),
            "weight_decay": float(rng.choice((0.0, 1e-6, 1e-5, 1e-4))),
        })
    if method == "dmpnn":
        return _unique_random(rng, 20, lambda: {
            "max_lr": float(10 ** rng.uniform(-5, -3)),
            "init_lr": None,
            "final_lr": None,
            "dropout": float(rng.choice((0.0, 0.1, 0.2))),
        })
    if method == "grover":
        return _unique_random(rng, 10, lambda: {
            "max_lr": float(10 ** rng.uniform(-5, -3)),
            "init_lr": None,
            "final_lr": None,
            "dropout": float(rng.choice((0.0, 0.1))),
            "weight_decay": float(rng.choice((0.0, 1e-5, 1e-4))),
        })
    if method == "toxacol":
        default = {"dropout": 0.1, "lr_multiplier": 1.0}
        return [default] + _unique_random(rng, 9, lambda: {
            "dropout": float(rng.choice((0.05, 0.1, 0.2))),
            "lr_multiplier": float(10 ** rng.uniform(np.log10(0.5), np.log10(2.0))),
        })
    raise ValueError(f"Unknown baseline method: {method}")


def finalize_learning_rates(method: str, trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if method not in {"dmpnn", "grover"}:
        return trials
    divisor = 100.0 if method == "dmpnn" else 10.0
    return [
        {**trial, "init_lr": trial["max_lr"] / divisor, "final_lr": trial["max_lr"] / divisor}
        for trial in trials
    ]
