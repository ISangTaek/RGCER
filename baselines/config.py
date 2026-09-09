"""Configuration resolution shared by formal training and local smoke runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import protocol_identity
from .constants import METHODS
from .contracts import MODEL_TYPES, expected_feature_schema, expected_task_names
from .utils import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _read_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {source}")
    return payload


def resolve_training_config(
    method: str,
    *,
    role: str,
    seed: int,
    device: str,
    config_path: str | Path | None = None,
    trials_path: str | Path | None = None,
    trial_index: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unknown baseline {method!r}; expected {METHODS}")
    if role not in {"formal", "smoke"}:
        raise ValueError("role must be 'formal' or 'smoke'")
    config_source = Path(config_path) if config_path else ROOT / "configs" / "baselines" / f"{method}.json"
    base = _read_object(config_source)
    if base.get("method") != method:
        raise ValueError(f"Config method mismatch: {base.get('method')!r} != {method!r}")

    trial = None
    trial_source = None
    if role == "formal":
        if trial_index is None:
            raise ValueError("Formal training requires --trial")
        trial_source = Path(trials_path) if trials_path else ROOT / "configs" / "baselines" / "trials" / f"{method}.json"
        trial_payload = _read_object(trial_source)
        if trial_payload.get("method") != method:
            raise ValueError("Trial file method does not match requested method")
        rows = trial_payload.get("trials")
        if not isinstance(rows, list) or not 0 <= int(trial_index) < len(rows):
            raise ValueError(f"Trial index {trial_index} is outside the available trial list")
        selected = dict(rows[int(trial_index)])
        stored_trial_id = selected.pop("trial_id", int(trial_index))
        if int(stored_trial_id) != int(trial_index):
            raise ValueError("Trial row id does not match its list position")
        trial = {"trial_id": int(trial_index), "parameters": selected}
        training = {**base["formal"], **selected}
    else:
        training = dict(base["smoke"])

    if overrides:
        unknown = sorted(set(overrides) - set(training))
        if unknown:
            raise ValueError(f"Unknown training override fields: {unknown}")
        training.update(overrides)

    config = {
        "format_version": 2,
        "mode": "human3_smoke" if role == "smoke" else "formal_train_validation",
        "role": role.upper(),
        "method": method,
        "model_type": MODEL_TYPES[method],
        "seed": int(seed),
        "device": str(device),
        "num_workers": 0,
        "allowed_splits": ["train", "validation"],
        "training_task_scope": "joint59" if method == "toxacol" else "human3",
        "selection_task_scope": "human3",
        "task_names": list(expected_task_names(method)),
        "feature_schema": expected_feature_schema(method),
        "architecture": base.get("architecture", {}),
        "training": training,
        "selection": base["selection"],
        "config_source": str(config_source.resolve()),
        "trial_index": int(trial_index) if trial_index is not None else None,
        "trial": trial,
        "trial_source": str(trial_source.resolve()) if trial_source else None,
        "trial_file_sha256": sha256_file(trial_source) if trial_source else None,
        "smoke_override": role == "smoke",
        **protocol_identity(),
    }
    return config
