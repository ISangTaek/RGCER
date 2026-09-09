"""Strict best-checkpoint selection and protocol metadata validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .constants import DATASTORE_BUILD_ID, DATASTORE_FINGERPRINT, PROTOCOL_SHA256, SPLIT_MANIFEST_HASH
from .contracts import MODEL_TYPES, expected_feature_schema, expected_task_names
from .scaling import TaskScaler


def protocol_identity() -> dict[str, str]:
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "datastore_build_id": DATASTORE_BUILD_ID,
        "datastore_fingerprint": DATASTORE_FINGERPRINT,
        "split_manifest_hash": SPLIT_MANIFEST_HASH,
    }


def validate_protocol_identity(payload: dict) -> None:
    for key, expected in protocol_identity().items():
        actual = payload.get(key)
        if actual != expected:
            raise ValueError(f"Checkpoint identity mismatch for {key}: {actual!r} != {expected!r}")


def validate_checkpoint_contract(payload: dict, *, expected_method: str | None = None) -> TaskScaler:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint is not a dictionary")
    validate_protocol_identity(payload)
    method = payload.get("method")
    if method not in MODEL_TYPES:
        raise ValueError(f"Unknown checkpoint method: {method!r}")
    if expected_method is not None and method != expected_method:
        raise ValueError(f"Checkpoint method mismatch: {method!r} != {expected_method!r}")
    if payload.get("model_type") != MODEL_TYPES[method]:
        raise ValueError("Checkpoint model_type does not match the approved implementation")
    expected_tasks = list(expected_task_names(method))
    if payload.get("task_names") != expected_tasks:
        raise ValueError("Checkpoint task order does not match the approved task order")
    expected_schema = expected_feature_schema(method)
    if payload.get("feature_schema") != expected_schema:
        raise ValueError("Checkpoint feature schema does not match the approved schema")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its resolved config")
    for key, expected in (
        ("method", method),
        ("model_type", MODEL_TYPES[method]),
        ("task_names", expected_tasks),
        ("feature_schema", expected_schema),
    ):
        if config.get(key) != expected:
            raise ValueError(f"Checkpoint config mismatch for {key}")
    scaler_payload = payload.get("scaler")
    if not isinstance(scaler_payload, dict):
        raise ValueError("Checkpoint is missing scaler metadata")
    scaler = TaskScaler.from_dict(scaler_payload)
    if list(scaler.task_names) != expected_tasks:
        raise ValueError("Checkpoint scaler task order does not match checkpoint tasks")
    if not (
        scaler.means.shape == scaler.stds.shape == scaler.counts.shape == (len(expected_tasks),)
        and np.isfinite(scaler.means).all()
        and np.isfinite(scaler.stds).all()
        and np.all(scaler.stds > 0)
    ):
        raise ValueError("Checkpoint scaler arrays are invalid")
    return scaler


@dataclass
class StrictBest:
    path: Path
    best_metric: float = np.inf
    best_epoch: int | None = None

    def consider(self, metric: float, epoch: int, save: Callable[[Path], None]) -> bool:
        value = float(metric)
        if not np.isfinite(value):
            raise ValueError("selection metric must be finite")
        if value < self.best_metric:  # strict: ties preserve the earlier checkpoint
            save(self.path)
            self.best_metric = value
            self.best_epoch = int(epoch)
            return True
        return False
