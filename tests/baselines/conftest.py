from __future__ import annotations

import os
from pathlib import Path
import json

import joblib
import numpy as np
import pytest

from baselines.checkpoint import protocol_identity
from baselines.config import resolve_training_config
from baselines.constants import (
    BASELINE_TRAINING_GIT_COMMIT,
    BASELINE_TRAINING_IMPLEMENTATION_SHA256,
    HUMAN3_TASKS,
)
from baselines.contracts import expected_feature_schema
from baselines.inference_authorization import authorize_test_inference, current_inference_identity
from baselines.scaling import TaskScaler
from baselines.utils import sha256_file


ROOT = Path(__file__).resolve().parents[2]


class ConstantRFModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, features):
        return np.full(len(features), self.value, dtype=np.float64)


def pytest_addoption(parser):
    group = parser.getgroup("baseline integration assets")
    group.addoption("--baseline-datastore", action="store", default=None)
    group.addoption("--chemprop-source", action="store", default=None)
    group.addoption("--grover-source", action="store", default=None)
    group.addoption("--grover-pretrained", action="store", default=None)
    group.addoption("--toxacol-source", action="store", default=None)


def _asset(request, option: str, environment: str, label: str) -> Path:
    value = request.config.getoption(option) or os.environ.get(environment)
    if not value:
        pytest.skip(f"{label} integration asset not supplied via {option} or {environment}")
    path = Path(value).resolve()
    if not path.exists():
        pytest.fail(f"Configured {label} integration asset does not exist: {path}")
    return path


@pytest.fixture
def baseline_datastore(request):
    return _asset(request, "--baseline-datastore", "BASELINE_DATASTORE", "DataStore")


@pytest.fixture
def chemprop_source(request):
    return _asset(request, "--chemprop-source", "CHEMPROP_SOURCE", "Chemprop source")


@pytest.fixture
def grover_source(request):
    return _asset(request, "--grover-source", "GROVER_SOURCE", "GROVER source")


@pytest.fixture
def grover_pretrained(request):
    return _asset(request, "--grover-pretrained", "GROVER_PRETRAINED", "GROVER_base weights")


@pytest.fixture
def toxacol_source(request):
    return _asset(request, "--toxacol-source", "TOXACOL_SOURCE", "TOXACol source")


@pytest.fixture
def formal_rf_checkpoint(tmp_path):
    config = resolve_training_config("rf", role="formal", seed=42, device="cpu", trial_index=2)
    scaler = TaskScaler.fit(
        np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float64),
        HUMAN3_TASKS,
        allow_empty=False,
    )
    payload = {
        **protocol_identity(),
        "format_version": 2,
        "method": "rf",
        "model_type": config["model_type"],
        "models": [ConstantRFModel(1.25), ConstantRFModel(2.5), ConstantRFModel(3.75)],
        "config": config,
        "scaler": scaler.to_dict(),
        "task_names": list(HUMAN3_TASKS),
        "feature_schema": expected_feature_schema("rf"),
        "seed": 42,
        "epoch": 0,
        "best_validation_human3_macro_rmse": 1.0,
    }
    path = tmp_path / "checkpoint.joblib"
    joblib.dump(payload, path)
    return path, payload


def authorization_payload(checkpoint: Path, *, asset_id: str = "rf-t02-s42") -> dict:
    identity = current_inference_identity(ROOT)
    return {
        "schema_version": 1,
        "task_id": "S3C2A_SYNTHETIC_TEST",
        "purpose": "fixed_checkpoint_test_export",
        "allowed_splits": ["test"],
        "inference_git_commit": identity["git_commit"],
        "inference_implementation_sha256": identity["implementation_sha256"],
        **protocol_identity(),
        "assets": [{
            "asset_id": asset_id,
            "method": "rf",
            "seed": 42,
            "trial": 2,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_git_commit": BASELINE_TRAINING_GIT_COMMIT,
            "training_implementation_sha256": BASELINE_TRAINING_IMPLEMENTATION_SHA256,
        }],
    }


@pytest.fixture
def authorized_rf_context(tmp_path, formal_rf_checkpoint):
    checkpoint, _ = formal_rf_checkpoint
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(authorization_payload(checkpoint), sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return authorize_test_inference(
        authorization_path=authorization,
        authorization_sha256=sha256_file(authorization),
        task_id="S3C2A_SYNTHETIC_TEST",
        asset_id="rf-t02-s42",
        method="rf",
        checkpoint=checkpoint,
        repository_root=ROOT,
    )
