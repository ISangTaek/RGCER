"""Exact Stage 3A2 smoke artifact schema and independent verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import protocol_identity
from .code_identity import implementation_identity
from .constants import HUMAN3_TASKS
from .metrics import regression_metrics
from .utils import read_json, sha256_file, write_json


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def model_sha256(model) -> str:
    return state_dict_sha256(model.state_dict())


def smoke_samples_payload(data) -> dict[str, Any]:
    rows = []
    for partition in (data.train, data.validation):
        labels = partition.labels_human3
        for row, sample_id in enumerate(partition.sample_ids):
            rows.append({
                "sample_id": sample_id,
                "row_index": int(partition.raw_row_indices[row]),
                "split": partition.split,
                "task_mask": {
                    task: bool(np.isfinite(labels[row, column]))
                    for column, task in enumerate(HUMAN3_TASKS)
                },
            })
    return {
        "role": "SMOKE",
        "tasks": list(HUMAN3_TASKS),
        "sample_identity_hash": data.sample_identity_hash,
        "train_count": len(data.train.sample_ids),
        "validation_count": len(data.validation.sample_ids),
        "samples": rows,
    }


def endpoint_prediction_rows(data, predictions: np.ndarray, scaler) -> list[dict[str, Any]]:
    rows = []
    truth = data.validation.labels_human3
    scaler_index = {name: index for index, name in enumerate(scaler.task_names)}
    for sample_row, sample_id in enumerate(data.validation.sample_ids):
        for task_column, task in enumerate(HUMAN3_TASKS):
            value = truth[sample_row, task_column]
            scale_column = scaler_index[task]
            rows.append({
                "sample_id": sample_id,
                "row_index": int(data.validation.raw_row_indices[sample_row]),
                "split": "validation",
                "endpoint": task,
                "y_true": float(value) if np.isfinite(value) else None,
                "y_pred": float(predictions[sample_row, task_column]),
                "mask": bool(np.isfinite(value)),
                "scale": {
                    "mean": float(scaler.means[scale_column]),
                    "std": float(scaler.stds[scale_column]),
                    "fit_split": "train",
                },
            })
    return rows


def run_manifest_payload(*, config: dict, data, environment: dict, checkpoint: Path,
                         metrics: dict, audit: dict, status: str, reason: str | None) -> dict:
    return {
        "method": config["method"],
        "role": config["role"],
        "seed": config["seed"],
        "status": status,
        "reason": reason,
        "device": config["device"],
        "environment": environment,
        "code_identity": {"git_commit": environment.get("git_commit")},
        "data_identity": {
            **protocol_identity(),
            "sample_identity_hash": data.sample_identity_hash,
            "datastore_root": data.datastore_root,
        },
        "input_manifest_sha256": data.sample_identity_hash,
        "selection_metric": metrics["selection_metric"],
        "best_validation_human3_macro_rmse": metrics["macro_rmse"],
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_audit": audit,
    }


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    required = (
        "run_manifest.json", "resolved_config.json", "smoke_samples.json", "scaler.json",
        "history.json", "validation_predictions.jsonl", "metrics.json", "stdout.log", "stderr.log",
        "command.json", "exit_code.txt", "reload_verification.json", "model_audit.json",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run is missing required artifacts: {missing}")
    manifest = read_json(root / "run_manifest.json")
    config = read_json(root / "resolved_config.json")
    samples = read_json(root / "smoke_samples.json")
    if manifest.get("role") != "SMOKE" or manifest.get("status") != "PASS":
        raise ValueError("run_manifest is not a passing SMOKE run")
    if config.get("allowed_splits") != ["train", "validation"]:
        raise ValueError("resolved config exposes a forbidden split")
    current_code = implementation_identity(Path(__file__).resolve().parents[1])
    if manifest.get("code_identity", {}).get("implementation_sha256") != current_code["implementation_sha256"]:
        raise ValueError("Run implementation hash differs from current baseline code")
    for key, expected in protocol_identity().items():
        if manifest["data_identity"].get(key) != expected or config.get(key) != expected:
            raise ValueError(f"Frozen identity mismatch for {key}")
    if samples["train_count"] > 96 or samples["validation_count"] > 24:
        raise ValueError("Smoke sample union exceeds protocol limits")
    if any(row["split"] not in {"train", "validation"} for row in samples["samples"]):
        raise ValueError("smoke_samples contains calibration/test")
    sample_validation = [row["sample_id"] for row in samples["samples"] if row["split"] == "validation"]

    rows = [json.loads(line) for line in (root / "validation_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected_pairs = [(sample_id, task) for sample_id in sample_validation for task in HUMAN3_TASKS]
    actual_pairs = [(row["sample_id"], row["endpoint"]) for row in rows]
    if actual_pairs != expected_pairs:
        raise ValueError("Prediction sample/endpoint rows differ from smoke_samples order")
    truth = np.full((len(sample_validation), 3), np.nan, dtype=np.float64)
    pred = np.empty((len(sample_validation), 3), dtype=np.float64)
    for row_index, row in enumerate(rows):
        sample_column, task_column = divmod(row_index, 3)
        if row["split"] != "validation" or bool(row["mask"]) != (row["y_true"] is not None):
            raise ValueError("Prediction split/mask contract is invalid")
        if row["scale"].get("fit_split") != "train":
            raise ValueError("Prediction scale was not fitted on train")
        if row["y_true"] is not None:
            truth[sample_column, task_column] = float(row["y_true"])
        pred[sample_column, task_column] = float(row["y_pred"])
    recomputed = regression_metrics(truth, pred, HUMAN3_TASKS)
    saved = read_json(root / "metrics.json")
    for key in ("selection_metric", "macro_rmse", "macro_rmse_reason", "pooled_rmse", "pooled_n", "per_task"):
        if json.dumps(recomputed[key], sort_keys=True) != json.dumps(saved[key], sort_keys=True):
            raise ValueError(f"Saved metric {key} differs from independent recomputation")
    if saved.get("reload", {}).get("allclose") is not True:
        raise ValueError("Reload allclose did not pass")
    if saved.get("training_update", {}).get("passed") is not True:
        raise ValueError("Training update check did not pass")
    checkpoint = root / manifest["checkpoint"]
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    if (root / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("Recorded command exit code is not zero")
    result = {
        "status": "passed",
        "method": manifest["method"],
        "run_dir": str(root),
        "validation_samples": len(sample_validation),
        "prediction_rows": len(rows),
        "metrics_recomputed": True,
        "checkpoint_sha256_verified": True,
        "training_update_verified": True,
        "forbidden_splits_absent": True,
    }
    write_json(root / "verification.json", result)
    return result
