"""Deterministic Stage 3C2A inference artifacts and independent verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .constants import HUMAN3_TASKS
from .data import BaselinePartition, InferenceSampleExpectation
from .metrics import regression_metrics
from .utils import canonical_sha256, sha256_file, write_json


_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "status",
    "method",
    "model_type",
    "seed",
    "trial",
    "asset_id",
    "split",
    "sample_count",
    "prediction_row_count",
    "sample_identity_hash",
    "checkpoint",
    "task_order",
    "checkpoint_task_names",
    "feature_schema",
    "restored_scaler_from_checkpoint",
    "restored_config_from_checkpoint",
    "training_identity",
    "inference_identity",
    "authorization",
    "data_identity",
    "source_root",
    "source_commit",
    "device",
})
_CHECKPOINT_FIELDS = frozenset({"path", "bytes", "sha256"})
_CODE_IDENTITY_FIELDS = frozenset({"git_commit", "implementation_sha256"})
_AUTHORIZATION_FIELDS = frozenset({"required", "path", "sha256", "task_id", "asset_id"})
_DATA_IDENTITY_FIELDS = frozenset({
    "protocol_sha256",
    "datastore_build_id",
    "datastore_fingerprint",
    "split_manifest_hash",
    "datastore_argument",
})
_METRICS_FIELDS = frozenset({
    "schema_version", "split", "human3_macro_rmse", "human3_macro_rmse_reason", "per_endpoint",
})
_ENDPOINT_METRIC_FIELDS = frozenset({"n", "rmse", "mae", "r2", "r2_reason"})
_METRIC_ABS_TOL = 1e-7
_METRIC_REL_TOL = 1e-9


@dataclass(frozen=True)
class InferenceManifestExpectation:
    """Trusted manifest identity assembled by the CLI before data inference."""

    checkpoint_path: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    method: str
    model_type: str
    seed: int
    trial: int | None
    asset_id: str | None
    task_order: tuple[str, ...]
    checkpoint_task_names: tuple[str, ...]
    feature_schema: dict[str, Any]
    restored_scaler_from_checkpoint: bool
    restored_config_from_checkpoint: bool
    authorization_required: bool
    authorization_path: str | None
    authorization_sha256: str | None
    authorization_task_id: str | None
    authorization_asset_id: str | None
    training_git_commit: str | None
    training_implementation_sha256: str | None
    inference_git_commit: str
    inference_implementation_sha256: str
    protocol_sha256: str
    datastore_build_id: str
    datastore_fingerprint: str
    split_manifest_hash: str
    datastore_argument: str
    source_root: str | None
    source_commit: str | None
    device: str


def _strict_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON number is forbidden in {path.name}: {value}")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"Duplicate JSON key is forbidden in {path.name}: {key!r}")
            payload[key] = value
        return payload

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=reject,
    )


def _require_exact_fields(payload: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")
    return payload


def _require_string_or_none(value: Any, label: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{label} must be a string or null")


def _require_exact_manifest_value(actual: Any, expected: Any, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"inference_manifest.json mismatch for {label}")
        return
    if type(expected) is bool:
        if type(actual) is not bool:
            raise ValueError(f"inference_manifest.json type mismatch for {label}")
    elif type(expected) is int:
        if type(actual) is not int:
            raise ValueError(f"inference_manifest.json type mismatch for {label}")
    elif isinstance(expected, str):
        if not isinstance(actual, str):
            raise ValueError(f"inference_manifest.json type mismatch for {label}")
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            raise ValueError(f"inference_manifest.json type mismatch for {label}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"inference_manifest.json type mismatch for {label}")
    if actual != expected:
        raise ValueError(f"inference_manifest.json mismatch for {label}")


def validate_prediction_matrix(predictions: np.ndarray, sample_count: int) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    expected_shape = (sample_count, len(HUMAN3_TASKS))
    if values.shape != expected_shape:
        raise ValueError(f"Predictions must have shape {expected_shape}, found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Every exported prediction must be finite")
    return values


def prediction_rows(
    partition: BaselinePartition,
    predictions: np.ndarray,
    *,
    asset_id: str | None,
    method: str,
    seed: int,
) -> list[dict[str, Any]]:
    values = validate_prediction_matrix(predictions, len(partition.sample_ids))
    rows = []
    for sample_row, sample_id in enumerate(partition.sample_ids):
        for task_column, task in enumerate(HUMAN3_TASKS):
            truth = partition.labels_human3[sample_row, task_column]
            rows.append({
                "asset_id": asset_id,
                "method": method,
                "seed": int(seed),
                "sample_id": sample_id,
                "row_index": int(partition.raw_row_indices[sample_row]),
                "split": partition.split,
                "endpoint": task,
                "y_true": float(truth) if np.isfinite(truth) else None,
                "y_pred": float(values[sample_row, task_column]),
                "mask": bool(np.isfinite(truth)),
            })
    return rows


def inference_metrics(y_true: np.ndarray, predictions: np.ndarray, split: str) -> dict[str, Any]:
    values = validate_prediction_matrix(predictions, len(y_true))
    metrics = regression_metrics(y_true, values, HUMAN3_TASKS)
    return {
        "schema_version": 1,
        "split": split,
        "human3_macro_rmse": metrics["macro_rmse"],
        "human3_macro_rmse_reason": metrics["macro_rmse_reason"],
        "per_endpoint": metrics["per_task"],
    }


def _independent_inference_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    split: str,
) -> dict[str, Any]:
    """Recompute metrics directly, without calling the artifact-generation metric path."""

    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    expected_shape = (truth.shape[0], len(HUMAN3_TASKS)) if truth.ndim == 2 else None
    if truth.shape != predicted.shape or truth.shape != expected_shape:
        raise ValueError("Independent metric arrays must have equal [samples, Human3 tasks] shape")
    if not np.isfinite(predicted).all():
        raise ValueError("Every prediction must be finite before independent metric calculation")
    per_endpoint: dict[str, dict[str, Any]] = {}
    for column, task in enumerate(HUMAN3_TASKS):
        finite = np.isfinite(truth[:, column])
        count = int(np.count_nonzero(finite))
        if count == 0:
            per_endpoint[task] = {
                "n": 0,
                "rmse": None,
                "mae": None,
                "r2": None,
                "r2_reason": "no_finite_pairs",
            }
            continue
        endpoint_truth = [float(value) for value in truth[finite, column]]
        endpoint_prediction = [float(value) for value in predicted[finite, column]]
        errors = [actual - expected for actual, expected in zip(endpoint_prediction, endpoint_truth)]
        squared_errors = [error * error for error in errors]
        sse = math.fsum(squared_errors)
        rmse = math.sqrt(sse / count)
        mae = math.fsum(abs(error) for error in errors) / count
        if count < 2:
            r2 = None
            reason = "fewer_than_two_samples"
        else:
            mean_truth = math.fsum(endpoint_truth) / count
            sst = math.fsum((value - mean_truth) ** 2 for value in endpoint_truth)
            if sst <= 0.0:
                r2 = None
                reason = "constant_targets"
            else:
                r2 = 1.0 - sse / sst
                reason = None
        per_endpoint[task] = {
            "n": count,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2) if r2 is not None else None,
            "r2_reason": reason,
        }
    endpoint_rmses = [per_endpoint[task]["rmse"] for task in HUMAN3_TASKS]
    if any(value is None for value in endpoint_rmses):
        macro_rmse = None
        macro_reason = "at_least_one_task_has_no_finite_pairs"
    else:
        macro_rmse = math.fsum(float(value) for value in endpoint_rmses) / len(HUMAN3_TASKS)
        macro_reason = None
    return {
        "schema_version": 1,
        "split": split,
        "human3_macro_rmse": float(macro_rmse) if macro_rmse is not None else None,
        "human3_macro_rmse_reason": macro_reason,
        "per_endpoint": per_endpoint,
    }


def _verify_metric_float(actual: Any, expected: float | None, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"metrics.json mismatch for {label}: expected null")
        return
    if type(actual) is not float or not math.isfinite(actual):
        raise ValueError(f"metrics.json {label} must be a finite JSON float")
    if not math.isclose(actual, expected, abs_tol=_METRIC_ABS_TOL, rel_tol=_METRIC_REL_TOL):
        raise ValueError(f"metrics.json differs from independent original-scale recomputation at {label}")


def _verify_metrics(actual: Any, expected: dict[str, Any]) -> None:
    metrics = _require_exact_fields(actual, _METRICS_FIELDS, "metrics.json")
    if type(metrics["schema_version"]) is not int or metrics["schema_version"] != 1:
        raise ValueError("metrics.json schema_version must be integer 1")
    if not isinstance(metrics["split"], str) or metrics["split"] != expected["split"]:
        raise ValueError("metrics.json split mismatch")
    _verify_metric_float(
        metrics["human3_macro_rmse"],
        expected["human3_macro_rmse"],
        "human3_macro_rmse",
    )
    if metrics["human3_macro_rmse_reason"] != expected["human3_macro_rmse_reason"]:
        raise ValueError("metrics.json human3_macro_rmse_reason mismatch")
    endpoint_payload = _require_exact_fields(
        metrics["per_endpoint"], frozenset(HUMAN3_TASKS), "metrics.json per_endpoint",
    )
    for task in HUMAN3_TASKS:
        actual_endpoint = _require_exact_fields(
            endpoint_payload[task], _ENDPOINT_METRIC_FIELDS, f"metrics.json per_endpoint.{task}",
        )
        expected_endpoint = expected["per_endpoint"][task]
        if type(actual_endpoint["n"]) is not int or actual_endpoint["n"] != expected_endpoint["n"]:
            raise ValueError(f"metrics.json integer N mismatch for {task}")
        for field_name in ("rmse", "mae", "r2"):
            _verify_metric_float(
                actual_endpoint[field_name],
                expected_endpoint[field_name],
                f"per_endpoint.{task}.{field_name}",
            )
        if actual_endpoint["r2_reason"] != expected_endpoint["r2_reason"]:
            raise ValueError(f"metrics.json r2_reason mismatch for {task}")


def _verify_manifest(
    manifest: Any,
    sample_expectation: InferenceSampleExpectation,
    identity: InferenceManifestExpectation,
    *,
    prediction_row_count: int,
    sample_identity_hash: str,
) -> None:
    payload = _require_exact_fields(manifest, _MANIFEST_FIELDS, "inference_manifest.json")
    expected_root = {
        "schema_version": 1,
        "status": "EXECUTED_NOT_SCIENTIFIC_PASS",
        "method": identity.method,
        "model_type": identity.model_type,
        "seed": identity.seed,
        "trial": identity.trial,
        "asset_id": identity.asset_id,
        "split": sample_expectation.split,
        "sample_count": len(sample_expectation.sample_ids),
        "prediction_row_count": prediction_row_count,
        "sample_identity_hash": sample_identity_hash,
        "task_order": list(identity.task_order),
        "checkpoint_task_names": list(identity.checkpoint_task_names),
        "feature_schema": identity.feature_schema,
        "restored_scaler_from_checkpoint": identity.restored_scaler_from_checkpoint,
        "restored_config_from_checkpoint": identity.restored_config_from_checkpoint,
        "source_root": identity.source_root,
        "source_commit": identity.source_commit,
        "device": identity.device,
    }
    for field_name, expected_value in expected_root.items():
        _require_exact_manifest_value(payload[field_name], expected_value, field_name)
    checkpoint = _require_exact_fields(payload["checkpoint"], _CHECKPOINT_FIELDS, "inference_manifest.json checkpoint")
    for field_name, expected_value in {
        "path": identity.checkpoint_path,
        "bytes": identity.checkpoint_bytes,
        "sha256": identity.checkpoint_sha256,
    }.items():
        _require_exact_manifest_value(checkpoint[field_name], expected_value, f"checkpoint.{field_name}")
    authorization = _require_exact_fields(
        payload["authorization"], _AUTHORIZATION_FIELDS, "inference_manifest.json authorization",
    )
    for field_name, expected_value in {
        "required": identity.authorization_required,
        "path": identity.authorization_path,
        "sha256": identity.authorization_sha256,
        "task_id": identity.authorization_task_id,
        "asset_id": identity.authorization_asset_id,
    }.items():
        _require_exact_manifest_value(authorization[field_name], expected_value, f"authorization.{field_name}")
    training_identity = _require_exact_fields(
        payload["training_identity"], _CODE_IDENTITY_FIELDS, "inference_manifest.json training_identity",
    )
    for field_name, expected_value in {
        "git_commit": identity.training_git_commit,
        "implementation_sha256": identity.training_implementation_sha256,
    }.items():
        _require_exact_manifest_value(
            training_identity[field_name], expected_value, f"training_identity.{field_name}",
        )
    inference_identity = _require_exact_fields(
        payload["inference_identity"], _CODE_IDENTITY_FIELDS, "inference_manifest.json inference_identity",
    )
    for field_name, expected_value in {
        "git_commit": identity.inference_git_commit,
        "implementation_sha256": identity.inference_implementation_sha256,
    }.items():
        _require_exact_manifest_value(
            inference_identity[field_name], expected_value, f"inference_identity.{field_name}",
        )
    data_identity = _require_exact_fields(
        payload["data_identity"], _DATA_IDENTITY_FIELDS, "inference_manifest.json data_identity",
    )
    for field_name, expected_value in {
        "protocol_sha256": identity.protocol_sha256,
        "datastore_build_id": identity.datastore_build_id,
        "datastore_fingerprint": identity.datastore_fingerprint,
        "split_manifest_hash": identity.split_manifest_hash,
        "datastore_argument": identity.datastore_argument,
    }.items():
        _require_exact_manifest_value(data_identity[field_name], expected_value, f"data_identity.{field_name}")


def _sample_rows(
    split: str,
    sample_ids: tuple[str, ...],
    raw_row_indices: np.ndarray,
    labels_human3: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_order": index,
            "sample_id": sample_id,
            "row_index": int(raw_row_indices[index]),
            "split": split,
            "task_mask": {
                task: bool(np.isfinite(labels_human3[index, column]))
                for column, task in enumerate(HUMAN3_TASKS)
            },
        }
        for index, sample_id in enumerate(sample_ids)
    ]


def sample_manifest_payload(
    split: str,
    sample_ids: tuple[str, ...],
    raw_row_indices: np.ndarray,
    labels_human3: np.ndarray,
) -> dict[str, Any]:
    rows = _sample_rows(split, sample_ids, raw_row_indices, labels_human3)
    identity = {
        "split": split,
        "task_names": list(HUMAN3_TASKS),
        "samples": rows,
    }
    return {
        "schema_version": 1,
        **identity,
        "sample_count": len(rows),
        "sample_identity_hash": canonical_sha256(identity),
    }


def verify_inference_artifacts(
    output_dir: str | Path,
    expectation: InferenceSampleExpectation,
    manifest_expectation: InferenceManifestExpectation,
) -> dict[str, Any]:
    """Re-read outputs and verify them against independently supplied execution identity."""

    output = Path(output_dir).resolve()
    required = ("predictions.jsonl", "metrics.json", "inference_manifest.json", "sample_manifest.json")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Inference output is missing required files: {missing}")
    prediction_records = [
        json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON number is forbidden in predictions.jsonl: {value}")
        ))
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_pairs = [
        (sample_id, task)
        for sample_id in expectation.sample_ids
        for task in HUMAN3_TASKS
    ]
    actual_pairs = [(row.get("sample_id"), row.get("endpoint")) for row in prediction_records]
    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError("Prediction output contains duplicate sample/endpoint keys")
    if actual_pairs != expected_pairs:
        raise ValueError("Prediction output has missing, extra, or misordered sample/endpoint keys")
    predictions = np.empty((len(expectation.sample_ids), len(HUMAN3_TASKS)), dtype=np.float64)
    expected_fields = {
        "asset_id", "method", "seed", "sample_id", "row_index", "split",
        "endpoint", "y_true", "y_pred", "mask",
    }
    for row_number, row in enumerate(prediction_records):
        if set(row) != expected_fields:
            raise ValueError(f"Prediction row fields mismatch at row {row_number}")
        sample_index, task_index = divmod(row_number, len(HUMAN3_TASKS))
        truth = expectation.labels_human3[sample_index, task_index]
        expected_truth = float(truth) if np.isfinite(truth) else None
        if (
            row["asset_id"] != manifest_expectation.asset_id
            or row["method"] != manifest_expectation.method
            or type(row["seed"]) is not int
            or row["seed"] != manifest_expectation.seed
            or row["split"] != expectation.split
            or type(row["row_index"]) is not int
            or row["row_index"] != int(expectation.raw_row_indices[sample_index])
            or type(row["mask"]) is not bool
            or row["mask"] != bool(np.isfinite(truth))
            or row["y_true"] != expected_truth
        ):
            raise ValueError(f"Prediction provenance, row index, truth, or mask mismatch at row {row_number}")
        prediction = row["y_pred"]
        if isinstance(prediction, bool) or not isinstance(prediction, (int, float)) or not np.isfinite(prediction):
            raise ValueError(f"Prediction is not finite numeric at row {row_number}")
        predictions[sample_index, task_index] = float(prediction)
    expected_sample_manifest = sample_manifest_payload(
        expectation.split,
        expectation.sample_ids,
        expectation.raw_row_indices,
        expectation.labels_human3,
    )
    if _strict_json(output / "sample_manifest.json") != expected_sample_manifest:
        raise ValueError("sample_manifest.json differs from the independently derived expected sample set")
    expected_metrics = _independent_inference_metrics(
        expectation.labels_human3,
        predictions,
        expectation.split,
    )
    _verify_metrics(_strict_json(output / "metrics.json"), expected_metrics)
    _verify_manifest(
        _strict_json(output / "inference_manifest.json"),
        expectation,
        manifest_expectation,
        prediction_row_count=len(prediction_records),
        sample_identity_hash=expected_sample_manifest["sample_identity_hash"],
    )
    checkpoint_sha256_after = sha256_file(manifest_expectation.checkpoint_path)
    if checkpoint_sha256_after != manifest_expectation.checkpoint_sha256:
        raise ValueError("Input checkpoint SHA-256 changed during inference")
    result = {
        "schema_version": 1,
        "status": "VERIFIED_EXECUTION",
        "scientific_acceptance": "NOT_GRANTED",
        "split": expectation.split,
        "expected_sample_count": len(expectation.sample_ids),
        "prediction_row_count": len(prediction_records),
        "unique_sample_endpoint_keys": True,
        "expected_coverage_exact": True,
        "label_masks_verified": True,
        "finite_predictions_verified": True,
        "original_scale_metrics_recomputed": True,
        "independent_metric_calculation": True,
        "sample_identity_hash_verified": True,
        "manifest_identity_verified": True,
        "checkpoint_sha256_before": manifest_expectation.checkpoint_sha256,
        "checkpoint_sha256_after": checkpoint_sha256_after,
        "input_checkpoint_unchanged": True,
    }
    write_json(output / "verification.json", result)
    return result
