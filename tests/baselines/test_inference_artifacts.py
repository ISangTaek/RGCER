from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pytest

import baselines.inference_artifacts as inference_artifacts_module
from baselines.checkpoint import protocol_identity
from baselines.constants import HUMAN3_TASKS, TOXACUTE_TASKS
from baselines.data import BaselinePartition, InferenceSampleExpectation
from baselines.inference import predict_partition
from baselines.inference_artifacts import (
    InferenceManifestExpectation,
    inference_metrics,
    prediction_rows,
    sample_manifest_payload,
    validate_prediction_matrix,
    verify_inference_artifacts,
)
from baselines.scaling import TaskScaler
from baselines.utils import read_json, sha256_file, write_json
from conftest import ROOT


def _partition(split="test"):
    labels = np.full((3, len(TOXACUTE_TASKS)), np.nan, dtype=np.float32)
    for column, task in enumerate(HUMAN3_TASKS):
        labels[:, TOXACUTE_TASKS.index(task)] = [1.0 + column, 1.0 + column, np.nan]
    labels[2, TOXACUTE_TASKS.index(HUMAN3_TASKS[2])] = 4.0
    return BaselinePartition(
        split=split,
        global_indices=np.asarray([4, 7, 9], dtype=np.int64),
        raw_row_indices=np.asarray([10, 20, 30], dtype=np.int64),
        sample_ids=("sample-a", "sample-b", "sample-c"),
        smiles=("CC", "CCC", "CCCC"),
        labels_all=labels,
        graph_records=({}, {}, {}),
    )


def _expectation(partition):
    return InferenceSampleExpectation(
        split=partition.split,
        sample_ids=partition.sample_ids,
        raw_row_indices=partition.raw_row_indices.copy(),
        labels_human3=partition.labels_human3.copy(),
    )


def _manifest_expectation(partition, checkpoint):
    data_identity = protocol_identity()
    return InferenceManifestExpectation(
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=sha256_file(checkpoint),
        method="rf",
        model_type="synthetic.rf",
        seed=42,
        trial=2,
        asset_id="rf-t02-s42",
        task_order=tuple(HUMAN3_TASKS),
        checkpoint_task_names=tuple(HUMAN3_TASKS),
        feature_schema={"name": "synthetic", "schema_hash": "a" * 64},
        restored_scaler_from_checkpoint=True,
        restored_config_from_checkpoint=True,
        authorization_required=True,
        authorization_path=str((checkpoint.parent / "authorization.json").resolve()),
        authorization_sha256="b" * 64,
        authorization_task_id="S3C2A_SYNTHETIC_TEST",
        authorization_asset_id="rf-t02-s42",
        training_git_commit="1" * 40,
        training_implementation_sha256="2" * 64,
        inference_git_commit="3" * 40,
        inference_implementation_sha256="4" * 64,
        protocol_sha256=data_identity["protocol_sha256"],
        datastore_build_id=data_identity["datastore_build_id"],
        datastore_fingerprint=data_identity["datastore_fingerprint"],
        split_manifest_hash=data_identity["split_manifest_hash"],
        datastore_argument=str((checkpoint.parent / "synthetic-datastore").resolve()),
        source_root=None,
        source_commit=None,
        device="cpu",
    )


def _manifest_payload(partition, rows, sample_manifest, identity):
    return {
        "schema_version": 1,
        "status": "EXECUTED_NOT_SCIENTIFIC_PASS",
        "method": identity.method,
        "model_type": identity.model_type,
        "seed": identity.seed,
        "trial": identity.trial,
        "asset_id": identity.asset_id,
        "split": partition.split,
        "sample_count": len(partition.sample_ids),
        "prediction_row_count": len(rows),
        "sample_identity_hash": sample_manifest["sample_identity_hash"],
        "checkpoint": {
            "path": identity.checkpoint_path,
            "bytes": identity.checkpoint_bytes,
            "sha256": identity.checkpoint_sha256,
        },
        "task_order": list(identity.task_order),
        "checkpoint_task_names": list(identity.checkpoint_task_names),
        "feature_schema": identity.feature_schema,
        "restored_scaler_from_checkpoint": identity.restored_scaler_from_checkpoint,
        "restored_config_from_checkpoint": identity.restored_config_from_checkpoint,
        "training_identity": {
            "git_commit": identity.training_git_commit,
            "implementation_sha256": identity.training_implementation_sha256,
        },
        "inference_identity": {
            "git_commit": identity.inference_git_commit,
            "implementation_sha256": identity.inference_implementation_sha256,
        },
        "authorization": {
            "required": identity.authorization_required,
            "path": identity.authorization_path,
            "sha256": identity.authorization_sha256,
            "task_id": identity.authorization_task_id,
            "asset_id": identity.authorization_asset_id,
        },
        "data_identity": {
            "protocol_sha256": identity.protocol_sha256,
            "datastore_build_id": identity.datastore_build_id,
            "datastore_fingerprint": identity.datastore_fingerprint,
            "split_manifest_hash": identity.split_manifest_hash,
            "datastore_argument": identity.datastore_argument,
        },
        "source_root": identity.source_root,
        "source_commit": identity.source_commit,
        "device": identity.device,
    }


def _write_artifacts(output, partition, predictions, identity, *, metrics_payload=None):
    rows = prediction_rows(
        partition,
        predictions,
        asset_id=identity.asset_id,
        method=identity.method,
        seed=identity.seed,
    )
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    if metrics_payload is None:
        metrics_payload = inference_artifacts_module.inference_metrics(
            partition.labels_human3, predictions, partition.split,
        )
    write_json(output / "metrics.json", metrics_payload)
    sample_manifest = sample_manifest_payload(
        partition.split,
        partition.sample_ids,
        partition.raw_row_indices,
        partition.labels_human3,
    )
    write_json(output / "sample_manifest.json", sample_manifest)
    write_json(output / "inference_manifest.json", _manifest_payload(partition, rows, sample_manifest, identity))


def _partition_with_human_labels(labels, *, split="test"):
    human_labels = np.asarray(labels, dtype=np.float32)
    all_labels = np.full((len(human_labels), len(TOXACUTE_TASKS)), np.nan, dtype=np.float32)
    for column, task in enumerate(HUMAN3_TASKS):
        all_labels[:, TOXACUTE_TASKS.index(task)] = human_labels[:, column]
    return BaselinePartition(
        split=split,
        global_indices=np.arange(len(human_labels), dtype=np.int64),
        raw_row_indices=np.arange(100, 100 + len(human_labels), dtype=np.int64),
        sample_ids=tuple(f"metric-{index}" for index in range(len(human_labels))),
        smiles=tuple("C" * (index + 1) for index in range(len(human_labels))),
        labels_all=all_labels,
        graph_records=tuple({} for _ in range(len(human_labels))),
    )


def test_rows_expand_every_selected_molecule_to_three_endpoints_with_masks():
    partition = _partition()
    predictions = np.arange(9, dtype=np.float64).reshape(3, 3) / 10.0
    rows = prediction_rows(partition, predictions, asset_id="rf-t02-s42", method="rf", seed=42)
    assert len(rows) == 9
    assert [(row["sample_id"], row["endpoint"]) for row in rows] == [
        (sample_id, task) for sample_id in partition.sample_ids for task in HUMAN3_TASKS
    ]
    assert [row["mask"] for row in rows[-3:]] == [False, False, True]
    assert [row["y_true"] for row in rows[-3:]] == [None, None, 4.0]


def test_constant_target_r2_is_explicitly_undefined():
    partition = _partition()
    metrics = inference_metrics(partition.labels_human3, np.ones((3, 3)), "test")
    first = metrics["per_endpoint"][HUMAN3_TASKS[0]]
    assert first["r2"] is None
    assert first["r2_reason"] == "constant_targets"
    assert metrics["human3_macro_rmse"] is not None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_are_rejected(value):
    predictions = np.ones((3, 3), dtype=np.float64)
    predictions[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        validate_prediction_matrix(predictions, 3)


def test_independent_verifier_recomputes_metrics_and_confirms_checkpoint_unchanged(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"immutable synthetic checkpoint")
    partition = _partition()
    predictions = np.arange(9, dtype=np.float64).reshape(3, 3) / 5.0
    identity = _manifest_expectation(partition, checkpoint)
    _write_artifacts(output, partition, predictions, identity)
    result = verify_inference_artifacts(
        output,
        _expectation(partition),
        identity,
    )
    assert result["original_scale_metrics_recomputed"] is True
    assert result["independent_metric_calculation"] is True
    assert result["manifest_identity_verified"] is True
    assert result["input_checkpoint_unchanged"] is True
    assert read_json(output / "verification.json") == result


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "missing, extra, or misordered"),
    (lambda rows: rows.__setitem__(1, dict(rows[0])), "duplicate"),
])
def test_independent_verifier_rejects_missing_or_duplicate_prediction_keys(tmp_path, mutation, match):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"immutable")
    partition = _partition()
    identity = _manifest_expectation(partition, checkpoint)
    _write_artifacts(output, partition, np.ones((3, 3), dtype=np.float64), identity)
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    mutation(rows)
    (output / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        verify_inference_artifacts(
            output,
            _expectation(partition),
            identity,
        )


_MANIFEST_IDENTITY_MUTATIONS = [
    (("schema_version",), 2),
    (("status",), "WRONG_STATUS"),
    (("method",), "afp"),
    (("model_type",), "wrong.model"),
    (("seed",), 43),
    (("trial",), 9),
    (("asset_id",), "wrong-asset"),
    (("split",), "validation"),
    (("sample_count",), 999),
    (("prediction_row_count",), 999),
    (("sample_identity_hash",), "0" * 64),
    (("checkpoint", "path"), "C:/wrong/checkpoint.bin"),
    (("checkpoint", "bytes"), 1),
    (("checkpoint", "sha256"), "0" * 64),
    (("task_order",), list(reversed(HUMAN3_TASKS))),
    (("checkpoint_task_names",), list(reversed(HUMAN3_TASKS))),
    (("feature_schema",), {"name": "wrong"}),
    (("restored_scaler_from_checkpoint",), False),
    (("restored_config_from_checkpoint",), False),
    (("authorization", "required"), False),
    (("authorization", "path"), "C:/wrong/authorization.json"),
    (("authorization", "sha256"), "0" * 64),
    (("authorization", "task_id"), "WRONG_TASK"),
    (("authorization", "asset_id"), "wrong-asset"),
    (("training_identity", "git_commit"), "0" * 40),
    (("training_identity", "implementation_sha256"), "0" * 64),
    (("inference_identity", "git_commit"), "0" * 40),
    (("inference_identity", "implementation_sha256"), "0" * 64),
    (("data_identity", "protocol_sha256"), "0" * 64),
    (("data_identity", "datastore_build_id"), "wrong-build"),
    (("data_identity", "datastore_fingerprint"), "0" * 64),
    (("data_identity", "split_manifest_hash"), "0" * 64),
    (("data_identity", "datastore_argument"), "C:/wrong/datastore"),
    (("source_root",), "C:/wrong/source"),
    (("source_commit",), "0" * 40),
    (("device",), "cuda:0"),
]


def _manifest_field(payload, path):
    current = payload
    for key in path[:-1]:
        current = current[key]
    return current, path[-1]


@pytest.mark.parametrize("path,wrong_value", _MANIFEST_IDENTITY_MUTATIONS, ids=lambda value: str(value))
@pytest.mark.parametrize("mutation", ("wrong", "missing"))
def test_manifest_every_identity_field_rejects_mismatch_or_missing(
    tmp_path, path, wrong_value, mutation,
):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"manifest identity checkpoint")
    partition = _partition()
    identity = _manifest_expectation(partition, checkpoint)
    _write_artifacts(output, partition, np.ones((3, 3), dtype=np.float64), identity)
    manifest = read_json(output / "inference_manifest.json")
    owner, key = _manifest_field(manifest, path)
    if mutation == "wrong":
        owner[key] = wrong_value
    else:
        del owner[key]
    write_json(output / "inference_manifest.json", manifest)
    with pytest.raises(ValueError, match="inference_manifest.json"):
        verify_inference_artifacts(output, _expectation(partition), identity)
    assert not (output / "verification.json").exists()


@pytest.mark.parametrize("path,wrong_type", [
    (("checkpoint",), []),
    (("checkpoint", "bytes"), True),
    (("seed",), True),
    (("trial",), "2"),
    (("task_order",), "not-a-list"),
    (("feature_schema",), []),
    (("restored_scaler_from_checkpoint",), 1),
    (("authorization", "required"), 1),
    (("training_identity",), []),
    (("inference_identity",), []),
    (("data_identity",), []),
    (("device",), 0),
])
def test_manifest_identity_schema_rejects_wrong_types(tmp_path, path, wrong_type):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"manifest type checkpoint")
    partition = _partition()
    identity = _manifest_expectation(partition, checkpoint)
    _write_artifacts(output, partition, np.ones((3, 3), dtype=np.float64), identity)
    manifest = read_json(output / "inference_manifest.json")
    owner, key = _manifest_field(manifest, path)
    owner[key] = wrong_type
    write_json(output / "inference_manifest.json", manifest)
    with pytest.raises(ValueError, match="inference_manifest.json"):
        verify_inference_artifacts(output, _expectation(partition), identity)


def test_independent_metric_calculation_accepts_hand_computed_values(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"hand metric checkpoint")
    partition = _partition_with_human_labels([
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 3.0],
        [np.nan, np.nan, 5.0],
    ])
    predictions = np.asarray([
        [2.0, 1.0, 1.0],
        [2.0, 5.0, 2.0],
        [0.0, 0.0, 7.0],
    ])
    task_a, task_b, task_c = HUMAN3_TASKS
    hand_metrics = {
        "schema_version": 1,
        "split": "test",
        "human3_macro_rmse": (math.sqrt(0.5) + 1.0 + math.sqrt(5.0 / 3.0)) / 3.0,
        "human3_macro_rmse_reason": None,
        "per_endpoint": {
            task_a: {"n": 2, "rmse": math.sqrt(0.5), "mae": 0.5, "r2": -1.0, "r2_reason": None},
            task_b: {"n": 2, "rmse": 1.0, "mae": 1.0, "r2": 0.0, "r2_reason": None},
            task_c: {
                "n": 3,
                "rmse": math.sqrt(5.0 / 3.0),
                "mae": 1.0,
                "r2": 0.375,
                "r2_reason": None,
            },
        },
    }
    identity = _manifest_expectation(partition, checkpoint)
    _write_artifacts(output, partition, predictions, identity, metrics_payload=hand_metrics)
    result = verify_inference_artifacts(output, _expectation(partition), identity)
    assert result["independent_metric_calculation"] is True


def test_independent_metric_calculation_covers_undefined_reasons_and_missing_macro(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"undefined metric checkpoint")
    partition = _partition_with_human_labels([
        [np.nan, 2.0, 4.0],
        [np.nan, np.nan, 4.0],
        [np.nan, np.nan, 4.0],
    ])
    predictions = np.ones((3, 3), dtype=np.float64)
    identity = _manifest_expectation(partition, checkpoint)
    _write_artifacts(output, partition, predictions, identity)
    verify_inference_artifacts(output, _expectation(partition), identity)
    metrics = read_json(output / "metrics.json")
    assert metrics["per_endpoint"][HUMAN3_TASKS[0]]["r2_reason"] == "no_finite_pairs"
    assert metrics["per_endpoint"][HUMAN3_TASKS[1]]["r2_reason"] == "fewer_than_two_samples"
    assert metrics["per_endpoint"][HUMAN3_TASKS[2]]["r2_reason"] == "constant_targets"
    assert metrics["human3_macro_rmse"] is None
    assert metrics["human3_macro_rmse_reason"] == "at_least_one_task_has_no_finite_pairs"


def test_verifier_rejects_correlated_generation_metric_fault(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"correlated metric fault checkpoint")
    partition = _partition()
    predictions = np.arange(9, dtype=np.float64).reshape(3, 3) / 5.0
    identity = _manifest_expectation(partition, checkpoint)
    original = inference_artifacts_module.inference_metrics
    calls = 0

    def faulty_generation_metrics(*args, **kwargs):
        nonlocal calls
        calls += 1
        payload = original(*args, **kwargs)
        payload["human3_macro_rmse"] = 999.0
        return payload

    monkeypatch.setattr(inference_artifacts_module, "inference_metrics", faulty_generation_metrics)
    _write_artifacts(output, partition, predictions, identity)
    assert calls == 1
    with pytest.raises(ValueError, match="independent original-scale recomputation"):
        verify_inference_artifacts(output, _expectation(partition), identity)
    assert calls == 1
    assert not (output / "verification.json").exists()


def test_independent_metric_float_tolerance_accepts_small_rounding_delta(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"metric tolerance checkpoint")
    partition = _partition()
    predictions = np.arange(9, dtype=np.float64).reshape(3, 3) / 5.0
    identity = _manifest_expectation(partition, checkpoint)
    _write_artifacts(output, partition, predictions, identity)
    metrics = read_json(output / "metrics.json")
    metrics["per_endpoint"][HUMAN3_TASKS[0]]["rmse"] += 5e-8
    write_json(output / "metrics.json", metrics)
    verify_inference_artifacts(output, _expectation(partition), identity)


def test_independent_metric_calculation_rejects_wrong_n_and_out_of_tolerance_float(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"metric rejection checkpoint")
    partition = _partition()
    predictions = np.arange(9, dtype=np.float64).reshape(3, 3) / 5.0
    identity = _manifest_expectation(partition, checkpoint)
    _write_artifacts(output, partition, predictions, identity)
    metrics = read_json(output / "metrics.json")
    metrics["per_endpoint"][HUMAN3_TASKS[0]]["n"] += 1
    metrics["per_endpoint"][HUMAN3_TASKS[1]]["rmse"] += 1e-5
    write_json(output / "metrics.json", metrics)
    with pytest.raises(ValueError, match="integer N mismatch"):
        verify_inference_artifacts(output, _expectation(partition), identity)


def test_inference_does_not_call_training_optimizer_or_scaler_fit(formal_rf_checkpoint, monkeypatch):
    checkpoint, _ = formal_rf_checkpoint
    partition = _partition(split="validation")

    def forbidden(*args, **kwargs):
        raise AssertionError("training-only path was called")

    monkeypatch.setattr(TaskScaler, "fit", forbidden)
    import sklearn.ensemble
    import torch

    monkeypatch.setattr(sklearn.ensemble.RandomForestRegressor, "fit", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    predictions, payload, _ = predict_partition(checkpoint, partition, expected_method="rf")
    assert payload["method"] == "rf"
    assert predictions.shape == (3, 3)


def _load_predict_script(name):
    script = ROOT / "scripts" / "baseline_predict.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_authorized_cli_positive_uses_only_isolated_synthetic_test_fixture(
    tmp_path, formal_rf_checkpoint, authorized_rf_context, monkeypatch
):
    checkpoint, _ = formal_rf_checkpoint
    partition = _partition()
    module = _load_predict_script("baseline_predict_positive")
    monkeypatch.setattr(module, "load_authorized_inference_split", lambda *args, **kwargs: partition)
    monkeypatch.setattr(module, "derive_authorized_sample_expectation", lambda *args, **kwargs: _expectation(partition))
    output = tmp_path / "authorized-output"
    before = sha256_file(checkpoint)
    assert module.main([
        "--checkpoint", str(checkpoint), "--method", "rf",
        "--datastore", str(tmp_path / "synthetic-datastore-not-opened"),
        "--output", str(output), "--split", "test",
        "--authorization", authorized_rf_context.authorization_path,
        "--authorization-sha256", authorized_rf_context.authorization_sha256,
        "--task-id", authorized_rf_context.authorization.task_id,
        "--asset-id", authorized_rf_context.asset.asset_id,
    ]) == 0
    assert sha256_file(checkpoint) == before
    verification = read_json(output / "verification.json")
    manifest = read_json(output / "inference_manifest.json")
    assert verification["input_checkpoint_unchanged"] is True
    assert verification["manifest_identity_verified"] is True
    assert manifest["checkpoint"]["sha256"] == before
    assert manifest["authorization"]["sha256"] == authorized_rf_context.authorization_sha256
    assert manifest["authorization"]["task_id"] == authorized_rf_context.authorization.task_id
    assert manifest["inference_identity"]["git_commit"] == authorized_rf_context.inference_git_commit
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 9


def test_validation_cli_matches_direct_original_scale_prediction(tmp_path, formal_rf_checkpoint, monkeypatch):
    checkpoint, _ = formal_rf_checkpoint
    partition = _partition(split="validation")
    direct, _, _ = predict_partition(checkpoint, partition, expected_method="rf")
    module = _load_predict_script("baseline_predict_validation")
    monkeypatch.setattr(module, "load_authorized_inference_split", lambda *args, **kwargs: partition)
    monkeypatch.setattr(module, "derive_authorized_sample_expectation", lambda *args, **kwargs: _expectation(partition))
    output = tmp_path / "validation-output"
    assert module.main([
        "--checkpoint", str(checkpoint), "--method", "rf",
        "--datastore", str(tmp_path / "synthetic-datastore-not-opened"),
        "--output", str(output), "--split", "validation",
    ]) == 0
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    actual = np.asarray([row["y_pred"] for row in rows]).reshape(3, 3)
    manifest = read_json(output / "inference_manifest.json")
    assert np.allclose(actual, direct, atol=0.0, rtol=0.0)
    assert manifest["authorization"] == {
        "required": False,
        "path": None,
        "sha256": None,
        "task_id": None,
        "asset_id": None,
    }
    assert manifest["training_identity"] == {"git_commit": None, "implementation_sha256": None}
    assert manifest["device"] == "cpu"


def test_cli_refuses_existing_nonempty_output_before_data_read(tmp_path, formal_rf_checkpoint, monkeypatch):
    checkpoint, _ = formal_rf_checkpoint
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    module = _load_predict_script("baseline_predict_existing")
    monkeypatch.setattr(
        module,
        "load_authorized_inference_split",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("data reader was called")),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.main([
            "--checkpoint", str(checkpoint), "--method", "rf", "--datastore", "unused",
            "--output", str(output), "--split", "validation",
        ])
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"
