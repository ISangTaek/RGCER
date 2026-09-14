"""S3C3-R2/R3 artifact identity, sample binding, and scheduling tests."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from checkpoint_inference_compatibility import HUMAN3_TASKS, PRODUCTION_POLICY
from s3c3_r2_control import (
    BASELINE_IDENTITIES,
    CORE_RUN_IDS,
    FrozenTestSample,
    FrozenTestSampleBinding,
    INHERITED_VALIDATION_RUN_IDS,
    NEW_VALIDATION_RUN_IDS,
    S3C3ControlError,
    plan_after_validation,
    validate_core_rows,
    verify_baseline_tests,
    verify_historical_validations,
)
from scripts.run_s3c3_core_test_r2 import (
    _core_test_binding_summary,
    _verified_worker_result,
)


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")


def _jsonl(rows):
    return b"".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )


def _digest(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_checksummed_zip(tmp_path, members, *, name="evidence.zip", bad_checksum=False):
    checksums = {
        member_name: hashlib.sha256(data).hexdigest()
        for member_name, data in members.items()
    }
    lines = []
    for member_name, digest in sorted(checksums.items()):
        if bad_checksum and not lines:
            digest = "0" * 64
        lines.append(f"{digest}  {member_name}\n")
    all_members = dict(members)
    all_members["checksums.sha256"] = "".join(lines).encode("utf-8")
    path = tmp_path / name
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_name, data in all_members.items():
            archive.writestr(member_name, data)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _core_samples():
    return [
        {"sample_id": "s0", "row_index": 10, "labels": [1.0, 2.0, 3.0]},
        {"sample_id": "s1", "row_index": 11, "labels": [2.0, 3.0, 4.0]},
    ]


def _frozen_shape_samples():
    return [
        {
            "sample_id": f"test-{sample_index:02d}",
            "row_index": 1000 + sample_index,
            "labels": [
                float(sample_index + task_index)
                if sample_index < finite_count
                else None
                for task_index, finite_count in enumerate((14, 13, 13))
            ],
        }
        for sample_index in range(28)
    ]


def _frozen_binding(samples=None, *, source_sha="a" * 64):
    samples = samples if samples is not None else _frozen_shape_samples()
    frozen_samples = tuple(
        FrozenTestSample(
            sample_id=sample["sample_id"],
            row_index=sample["row_index"],
            labels=tuple(sample["labels"]),
        )
        for sample in samples
    )
    return FrozenTestSampleBinding(
        source_archive_sha256=source_sha,
        sample_identity_sha256=_digest(
            [sample.as_json() for sample in frozen_samples]
        ),
        samples=frozen_samples,
    )


def _core_rows(method, seed, split, samples=None):
    samples = samples if samples is not None else _core_samples()
    return [
        {
            "method": method,
            "seed": seed,
            "sample_id": sample["sample_id"],
            "row_index": sample["row_index"],
            "split": split,
            "endpoint": task,
            "y_true": sample["labels"][task_index],
            "y_pred": float(sample_index + task_index) + 0.5,
            "mask": sample["labels"][task_index] is not None,
        }
        for sample_index, sample in enumerate(samples)
        for task_index, task in enumerate(HUMAN3_TASKS)
    ]


def _write_worker_output(
    tmp_path,
    asset,
    samples,
    *,
    rows=None,
    metrics=None,
    name="worker",
):
    output = tmp_path / name
    output.mkdir()
    rows = rows if rows is not None else _core_rows(
        asset["method"],
        asset["seed"],
        "test",
        samples,
    )
    if metrics is None:
        metrics = validate_core_rows(
            rows,
            samples,
            method=asset["method"],
            seed=asset["seed"],
            split="test",
        )
    (output / "expected_samples.json").write_bytes(_json(samples))
    (output / "predictions.jsonl").write_bytes(_jsonl(rows))
    (output / "metrics.json").write_bytes(_json(metrics))
    return output


def _asset_record(run_id=CORE_RUN_IDS[0]):
    asset = PRODUCTION_POLICY.asset(run_id)
    return {
        "run_id": asset.run_id,
        "method": asset.method,
        "seed": asset.seed,
    }


def _historical_members(*, wrong_run_id=None, include_failed_directory=True):
    members = {}
    samples = _core_samples()
    for run_id in INHERITED_VALIDATION_RUN_IDS:
        asset = PRODUCTION_POLICY.asset(run_id)
        prefix = f"validation_{run_id}/"
        rows = _core_rows(asset.method, asset.seed, "validation", samples)
        metrics = validate_core_rows(
            rows,
            samples,
            method=asset.method,
            seed=asset.seed,
            split="validation",
        )
        restored_run_id = wrong_run_id if wrong_run_id and run_id == INHERITED_VALIDATION_RUN_IDS[0] else run_id
        members[prefix + "expected_samples.json"] = _json(samples)
        members[prefix + "predictions.jsonl"] = _jsonl(rows)
        members[prefix + "metrics.json"] = _json(metrics)
        members[prefix + "restoration.json"] = _json(
            {
                "asset": {
                    "run_id": restored_run_id,
                    "method": asset.method,
                    "seed": asset.seed,
                    "best_checkpoint": {
                        "sha256": asset.checkpoint_sha256,
                        "size_bytes": asset.checkpoint_size_bytes,
                    },
                },
                "checkpoint_unchanged": True,
                "model_unchanged": True,
                "training_called": False,
                "calibration_called": False,
            }
        )
        members[prefix + "validation_comparison.json"] = _json(
            {
                "status": "PASS",
                "reference_sha256": "d" * 64,
                "max_abs_difference": 0.0,
            }
        )
    if include_failed_directory:
        members["validation_D8_formal_B1_s42/error.txt"] = b"historical failure"
    return members


def test_historical_six_are_reused_only_after_zip_member_run_and_metric_checks(tmp_path):
    path, archive_sha = _write_checksummed_zip(tmp_path, _historical_members())
    verified = verify_historical_validations(path, expected_sha256=archive_sha)
    assert set(verified.verified_identities) == {
        (PRODUCTION_POLICY.asset(run_id).method, PRODUCTION_POLICY.asset(run_id).seed)
        for run_id in INHERITED_VALIDATION_RUN_IDS
    }
    assert len(verified.verified_identities) == 6
    assert all(
        "validation_D8_formal_B1_s42/" not in name
        for name in verified.selected_members
    )


def test_historical_pass_string_without_machine_runs_is_rejected(tmp_path):
    path, archive_sha = _write_checksummed_zip(
        tmp_path, {"batch_status.json": _json({"content_status": "PASS"})}
    )
    with pytest.raises(S3C3ControlError, match="member identity"):
        verify_historical_validations(path, expected_sha256=archive_sha)


def test_historical_wrong_run_identity_is_rejected(tmp_path):
    path, archive_sha = _write_checksummed_zip(
        tmp_path, _historical_members(wrong_run_id="wrong")
    )
    with pytest.raises(S3C3ControlError, match="restoration identity"):
        verify_historical_validations(path, expected_sha256=archive_sha)


def test_reuse_zip_requires_exact_sha_and_exact_member_checksums(tmp_path):
    path, archive_sha = _write_checksummed_zip(
        tmp_path, _historical_members(), bad_checksum=True
    )
    with pytest.raises(S3C3ControlError, match="checksum differs"):
        verify_historical_validations(path, expected_sha256=archive_sha)
    with pytest.raises(S3C3ControlError, match="ZIP SHA"):
        verify_historical_validations(path, expected_sha256="0" * 64)


def _baseline_members(*, duplicate_identity=False):
    samples = [
        {
            "sample_order": 0,
            "sample_id": "t0",
            "row_index": 20,
            "split": "test",
            "labels": [1.0, 2.0, 3.0],
        },
        {
            "sample_order": 1,
            "sample_id": "t1",
            "row_index": 21,
            "split": "test",
            "labels": [2.0, 3.0, 4.0],
        },
    ]
    expected = {
        "split": "test",
        "task_names": list(HUMAN3_TASKS),
        "samples": samples,
    }
    members = {"expected_test_samples.json": _json(expected)}
    identities = sorted(BASELINE_IDENTITIES)
    if duplicate_identity:
        identities[-1] = identities[0]
    for index, (method, seed) in enumerate(identities):
        asset_id = f"{method}_trial{index:03d}_seed{seed}"
        prefix = f"{asset_id}/"
        rows = []
        sample_manifest_rows = []
        for sample in samples:
            mask = {
                task: sample["labels"][task_index] is not None
                for task_index, task in enumerate(HUMAN3_TASKS)
            }
            sample_manifest_rows.append(
                {
                    "sample_order": sample["sample_order"],
                    "sample_id": sample["sample_id"],
                    "row_index": sample["row_index"],
                    "split": "test",
                    "task_mask": mask,
                }
            )
            for task_index, task in enumerate(HUMAN3_TASKS):
                rows.append(
                    {
                        "asset_id": asset_id,
                        "method": method,
                        "seed": seed,
                        "sample_id": sample["sample_id"],
                        "row_index": sample["row_index"],
                        "split": "test",
                        "endpoint": task,
                        "y_true": sample["labels"][task_index],
                        "y_pred": sample["labels"][task_index] + 1.0,
                        "mask": True,
                    }
                )
        simple_rows = [
            {key: value for key, value in row.items() if key != "asset_id"}
            for row in rows
        ]
        metrics = validate_core_rows(
            simple_rows, samples, method=method, seed=seed, split="test"
        )
        sample_identity = {
            "split": "test",
            "task_names": list(HUMAN3_TASKS),
            "samples": sample_manifest_rows,
        }
        sample_hash = _digest(sample_identity)
        checkpoint_sha = f"{index + 1:064x}"
        members[prefix + "predictions.jsonl"] = _jsonl(rows)
        members[prefix + "metrics.json"] = _json(metrics)
        members[prefix + "sample_manifest.json"] = _json(
            {
                "schema_version": 1,
                **sample_identity,
                "sample_count": len(samples),
                "sample_identity_hash": sample_hash,
            }
        )
        members[prefix + "inference_manifest.json"] = _json(
            {
                "asset_id": asset_id,
                "method": method,
                "seed": seed,
                "split": "test",
                "task_order": list(HUMAN3_TASKS),
                "sample_count": len(samples),
                "prediction_row_count": len(rows),
                "sample_identity_hash": sample_hash,
                "checkpoint": {
                    "path": f"/assets/{asset_id}.pt",
                    "bytes": 10,
                    "sha256": checkpoint_sha,
                },
            }
        )
        members[prefix + "verification.json"] = _json(
            {
                "status": "VERIFIED_EXECUTION",
                "expected_coverage_exact": True,
                "label_masks_verified": True,
                "finite_predictions_verified": True,
                "independent_metric_calculation": True,
                "manifest_identity_verified": True,
                "input_checkpoint_unchanged": True,
                "checkpoint_sha256_before": checkpoint_sha,
                "checkpoint_sha256_after": checkpoint_sha,
            }
        )
    return members


def test_baseline_25_reuse_recomputes_rows_metrics_and_all_identities(tmp_path):
    path, archive_sha = _write_checksummed_zip(
        tmp_path, _baseline_members(), name="baseline.zip"
    )
    verified = verify_baseline_tests(path, expected_sha256=archive_sha)
    assert len(verified.verified_identities) == 25
    assert set(verified.verified_identities) == BASELINE_IDENTITIES
    binding = verified.frozen_test_sample_binding
    assert binding is not None
    assert binding.source_archive_sha256 == archive_sha
    assert binding.sample_identity_sha256 == _digest(binding.as_json_samples())
    assert all(
        set(sample) == {"sample_id", "row_index", "labels"}
        for sample in binding.as_json_samples()
    )


def test_baseline_frozen_table_rejects_boolean_row_index(tmp_path):
    members = _baseline_members()
    expected = json.loads(members["expected_test_samples.json"])
    expected["samples"][0]["row_index"] = True
    members["expected_test_samples.json"] = _json(expected)
    path, archive_sha = _write_checksummed_zip(
        tmp_path,
        members,
        name="baseline-bool-row-index.zip",
    )
    with pytest.raises(S3C3ControlError, match="true int"):
        verify_baseline_tests(path, expected_sha256=archive_sha)


def test_baseline_duplicate_identity_is_rejected_despite_complete_files(tmp_path):
    path, archive_sha = _write_checksummed_zip(
        tmp_path,
        _baseline_members(duplicate_identity=True),
        name="baseline-duplicate.zip",
    )
    with pytest.raises(S3C3ControlError, match="identity coverage"):
        verify_baseline_tests(path, expected_sha256=archive_sha)


def test_fifteen_core_results_require_and_record_one_frozen_test_binding(tmp_path):
    samples = _frozen_shape_samples()
    binding = _frozen_binding(samples)
    completed = []
    for run_id in CORE_RUN_IDS:
        asset = _asset_record(run_id)
        output = _write_worker_output(
            tmp_path,
            asset,
            samples,
            name=f"test_{run_id}",
        )
        completed.append(
            _verified_worker_result(
                output,
                asset,
                "test",
                frozen_test_binding=binding,
            )
        )
        verification = json.loads(
            (output / "frozen_test_sample_verification.json").read_text(
                encoding="utf-8"
            )
        )
        assert verification["status"] == "VERIFIED"
        assert verification["sample_count"] == 28
        assert verification["exact_ordered_match"] is True
    summary = _core_test_binding_summary(completed, binding)
    assert summary["status"] == "VERIFIED"
    assert summary["verified_run_count"] == 15
    assert [row["run_id"] for row in summary["runs"]] == list(CORE_RUN_IDS)
    assert summary["frozen_test_sample_identity_sha256"] == (
        binding.sample_identity_sha256
    )


@pytest.mark.parametrize(
    "case",
    (
        "different_sample_ids",
        "different_row_indices",
        "different_labels",
        "different_order",
    ),
)
def test_runner_rejects_self_consistent_wrong_test_identity_before_completion(
    tmp_path,
    case,
):
    frozen_samples = _frozen_shape_samples()
    binding = _frozen_binding(frozen_samples)
    actual = copy.deepcopy(frozen_samples)
    if case == "different_sample_ids":
        actual[0]["sample_id"] = "replacement-id"
    elif case == "different_row_indices":
        actual[0]["row_index"] += 10000
    elif case == "different_labels":
        actual[0]["labels"][0] += 0.25
    else:
        actual[0], actual[1] = actual[1], actual[0]
    asset = _asset_record()
    output = _write_worker_output(tmp_path, asset, actual, name=case)
    stored_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert [
        stored_metrics["per_endpoint"][task]["n"] for task in HUMAN3_TASKS
    ] == [14, 13, 13]
    completed = []
    with pytest.raises(S3C3ControlError, match="frozen"):
        completed.append(
            _verified_worker_result(
                output,
                asset,
                "test",
                frozen_test_binding=binding,
            )
        )
    assert completed == []
    assert not (output / "frozen_test_sample_verification.json").exists()
    assert not (tmp_path / "core_test_sample_binding.json").exists()
    assert not (tmp_path / "unified_40_test_metrics.json").exists()


@pytest.mark.parametrize("case", ("duplicate", "missing", "extra"))
def test_runner_rejects_duplicate_missing_or_extra_test_samples(tmp_path, case):
    frozen_samples = _frozen_shape_samples()
    binding = _frozen_binding(frozen_samples)
    actual = copy.deepcopy(frozen_samples)
    if case == "duplicate":
        actual[1] = copy.deepcopy(actual[0])
    elif case == "missing":
        actual.pop()
    else:
        actual.append(
            {
                "sample_id": "extra-sample",
                "row_index": 9999,
                "labels": [None, None, None],
            }
        )
    asset = _asset_record()
    valid_rows = _core_rows(asset["method"], asset["seed"], "test", frozen_samples)
    valid_metrics = validate_core_rows(
        valid_rows,
        frozen_samples,
        method=asset["method"],
        seed=asset["seed"],
        split="test",
    )
    output = _write_worker_output(
        tmp_path,
        asset,
        actual,
        rows=_core_rows(asset["method"], asset["seed"], "test", actual),
        metrics=valid_metrics,
        name=case,
    )
    with pytest.raises(S3C3ControlError, match="frozen"):
        _verified_worker_result(
            output,
            asset,
            "test",
            frozen_test_binding=binding,
        )
    assert not (output / "frozen_test_sample_verification.json").exists()


def test_runner_rejects_boolean_pseudo_integer_row_index(tmp_path):
    frozen_samples = _frozen_shape_samples()
    binding = _frozen_binding(frozen_samples)
    actual = copy.deepcopy(frozen_samples)
    actual[0]["row_index"] = True
    asset = _asset_record()
    valid_rows = _core_rows(asset["method"], asset["seed"], "test", frozen_samples)
    valid_metrics = validate_core_rows(
        valid_rows,
        frozen_samples,
        method=asset["method"],
        seed=asset["seed"],
        split="test",
    )
    output = _write_worker_output(
        tmp_path,
        asset,
        actual,
        metrics=valid_metrics,
        name="bool-row-index",
    )
    with pytest.raises(S3C3ControlError, match="true int"):
        _verified_worker_result(
            output,
            asset,
            "test",
            frozen_test_binding=binding,
        )
    assert not (output / "frozen_test_sample_verification.json").exists()


def test_runner_rejects_equal_numeric_label_with_wrong_json_type(tmp_path):
    frozen_samples = _frozen_shape_samples()
    binding = _frozen_binding(frozen_samples)
    actual = copy.deepcopy(frozen_samples)
    actual[0]["labels"][0] = int(actual[0]["labels"][0])
    asset = _asset_record()
    output = _write_worker_output(tmp_path, asset, actual, name="label-type")
    with pytest.raises(S3C3ControlError, match="value/type"):
        _verified_worker_result(
            output,
            asset,
            "test",
            frozen_test_binding=binding,
        )


def test_runner_rejects_null_label_rewritten_in_worker_expectation(tmp_path):
    frozen_samples = _frozen_shape_samples()
    binding = _frozen_binding(frozen_samples)
    actual = copy.deepcopy(frozen_samples)
    actual[-1]["labels"][0] = 0.0
    asset = _asset_record()
    output = _write_worker_output(tmp_path, asset, actual, name="null-label")
    with pytest.raises(S3C3ControlError, match="value/type"):
        _verified_worker_result(
            output,
            asset,
            "test",
            frozen_test_binding=binding,
        )


def test_runner_rejects_prediction_mask_that_disagrees_with_frozen_null(tmp_path):
    samples = _frozen_shape_samples()
    binding = _frozen_binding(samples)
    asset = _asset_record()
    rows = _core_rows(asset["method"], asset["seed"], "test", samples)
    metrics = validate_core_rows(
        rows,
        samples,
        method=asset["method"],
        seed=asset["seed"],
        split="test",
    )
    null_row = next(row for row in rows if row["y_true"] is None)
    null_row["mask"] = True
    output = _write_worker_output(
        tmp_path,
        asset,
        samples,
        rows=rows,
        metrics=metrics,
        name="wrong-mask",
    )
    with pytest.raises(S3C3ControlError, match="mask differs"):
        _verified_worker_result(
            output,
            asset,
            "test",
            frozen_test_binding=binding,
        )
    assert not (output / "frozen_test_sample_verification.json").exists()


def test_validation_worker_result_does_not_require_or_claim_test_binding(tmp_path):
    samples = _core_samples()
    asset = _asset_record()
    output = tmp_path / "validation"
    output.mkdir()
    rows = _core_rows(asset["method"], asset["seed"], "validation", samples)
    metrics = validate_core_rows(
        rows,
        samples,
        method=asset["method"],
        seed=asset["seed"],
        split="validation",
    )
    (output / "expected_samples.json").write_bytes(_json(samples))
    (output / "predictions.jsonl").write_bytes(_jsonl(rows))
    (output / "metrics.json").write_bytes(_json(metrics))
    result = _verified_worker_result(output, asset, "validation")
    assert result["split"] == "validation"
    assert "frozen_test_sample_verification" not in result
    assert not (output / "frozen_test_sample_verification.json").exists()


def test_one_new_validation_failure_schedules_zero_tests_and_retains_completed():
    results = {run_id: True for run_id in NEW_VALIDATION_RUN_IDS}
    results[NEW_VALIDATION_RUN_IDS[3]] = False
    completed = CORE_RUN_IDS[:2]
    decision = plan_after_validation(
        historical_validation_verified=True,
        new_validation_results=results,
        completed_core_tests=completed,
    )
    assert decision["validation_gate_passed"] is False
    assert decision["core_tests_to_schedule"] == []
    assert decision["completed_core_tests_retained"] == list(completed)
    assert decision["new_validation_failed"] == [NEW_VALIDATION_RUN_IDS[3]]


def test_normal_batch_simulation_is_six_plus_nine_then_fifteen_plus_twenty_five():
    before = plan_after_validation(
        historical_validation_verified=True,
        new_validation_results={},
    )
    assert len(before["inherited_validation_runs"]) == 6
    assert len(before["new_validation_pending"]) == 9
    assert before["core_tests_to_schedule"] == []

    after = plan_after_validation(
        historical_validation_verified=True,
        new_validation_results={run_id: True for run_id in NEW_VALIDATION_RUN_IDS},
    )
    assert after["validation_gate_passed"] is True
    assert after["new_validation_pending"] == []
    assert after["core_tests_to_schedule"] == list(CORE_RUN_IDS)
    assert len(set(after["core_tests_to_schedule"])) == 15
    assert after["baseline_tests_to_reuse"] == 25


def test_versioned_runner_imports_repository_from_an_unrelated_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_s3c3_core_test_r2.py"
    result = subprocess.run(
        [sys.executable, str(script), "--mode", "selftest"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert Path(payload["repository_root"]).resolve() == script.parents[1]
