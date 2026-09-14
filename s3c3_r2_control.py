"""Independent S3C3-R2 archive checks, metrics, and validation gate logic."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from checkpoint_inference_compatibility import HUMAN3_TASKS, PRODUCTION_POLICY


HISTORICAL_VALIDATION_ZIP_SHA256 = (
    "2c35df5f2b5963c45f14197acadf29172065b45e20aab2cb083c55860f169641"
)
BASELINE_TEST_ZIP_SHA256 = (
    "8fc445fa8f21815c81f32cbb4e5b337cbf3c63219029c9938f3130f167af50eb"
)

INHERITED_VALIDATION_RUN_IDS = (
    "D8_formal_B0_s42",
    "D8_formal_B0_s43",
    "D8_formal_B0_s44",
    "D8_formal_B0_s45",
    "D8_formal_B0_s46",
    "D8_formal_B1_s43",
)
CORE_RUN_IDS = tuple(asset.run_id for asset in PRODUCTION_POLICY.assets)
NEW_VALIDATION_RUN_IDS = tuple(
    run_id for run_id in CORE_RUN_IDS if run_id not in INHERITED_VALIDATION_RUN_IDS
)
BASELINE_METHODS = ("rf", "afp", "dmpnn", "grover", "toxacol")
BASELINE_IDENTITIES = frozenset(
    (method, seed) for method in BASELINE_METHODS for seed in range(42, 47)
)


class S3C3ControlError(ValueError):
    """An inherited artifact or scheduling state fails closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_loads(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise S3C3ControlError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            S3C3ControlError(f"non-finite JSON value: {value}")
        ),
    )


def _identity_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_member(members: Mapping[str, bytes], name: str) -> Any:
    try:
        data = members[name]
    except KeyError as exc:
        raise S3C3ControlError(f"required ZIP member is missing: {name}") from exc
    try:
        return strict_json_loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise S3C3ControlError(f"ZIP member is not UTF-8 JSON: {name}") from exc


def metric_values(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    count = len(pairs)
    if not count:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "r2_reason": "no_finite_pairs",
        }
    target_mean = math.fsum(target for target, _ in pairs) / count
    squared_error = math.fsum((target - prediction) ** 2 for target, prediction in pairs)
    target_variation = math.fsum((target - target_mean) ** 2 for target, _ in pairs)
    return {
        "n": count,
        "rmse": math.sqrt(squared_error / count),
        "mae": math.fsum(abs(target - prediction) for target, prediction in pairs)
        / count,
        "r2": None if count < 2 or target_variation == 0 else 1 - squared_error / target_variation,
        "r2_reason": (
            "fewer_than_two_samples"
            if count < 2
            else "constant_targets"
            if target_variation == 0
            else None
        ),
    }


def _same_metric(actual: Any, expected: Any, context: str) -> None:
    if expected is None or isinstance(expected, str):
        if actual != expected:
            raise S3C3ControlError(f"{context} differs ({actual!r} != {expected!r})")
        return
    if type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise S3C3ControlError(f"{context} differs ({actual!r} != {expected!r})")
        return
    if type(actual) not in (int, float) or not math.isfinite(actual):
        raise S3C3ControlError(f"{context} is not a finite number")
    if not math.isclose(actual, expected, abs_tol=1e-7, rel_tol=1e-9):
        raise S3C3ControlError(f"{context} differs ({actual!r} != {expected!r})")


def validate_core_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_samples: Sequence[Mapping[str, Any]],
    *,
    method: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise S3C3ControlError("core rows use an unauthorized split")
    keys = [
        (sample["sample_id"], task)
        for sample in expected_samples
        for task in HUMAN3_TASKS
    ]
    row_keys = [(row.get("sample_id"), row.get("endpoint")) for row in rows]
    if row_keys != keys or len(keys) != len(set(keys)):
        raise S3C3ControlError("core prediction coverage/order is not exact")
    per_endpoint: dict[str, list[tuple[float, float]]] = {
        task: [] for task in HUMAN3_TASKS
    }
    for sample_index, sample in enumerate(expected_samples):
        labels = sample.get("labels")
        if not isinstance(labels, list) or len(labels) != len(HUMAN3_TASKS):
            raise S3C3ControlError("expected sample labels are not Human3 ordered")
        for task_index, task in enumerate(HUMAN3_TASKS):
            row = rows[sample_index * len(HUMAN3_TASKS) + task_index]
            target = labels[task_index]
            if row.get("method") != method:
                raise S3C3ControlError("core prediction method identity differs")
            if type(row.get("seed")) is not int or row["seed"] != seed:
                raise S3C3ControlError("core prediction seed identity differs")
            if row.get("split") != split:
                raise S3C3ControlError("core prediction split identity differs")
            if type(row.get("row_index")) is not int or row["row_index"] != sample["row_index"]:
                raise S3C3ControlError("core prediction row identity differs")
            if row.get("y_true") != target:
                raise S3C3ControlError("core prediction target differs")
            expected_mask = target is not None
            if type(row.get("mask")) is not bool or row["mask"] is not expected_mask:
                raise S3C3ControlError("core prediction mask differs")
            prediction = row.get("y_pred")
            if type(prediction) not in (int, float) or not math.isfinite(prediction):
                raise S3C3ControlError("core prediction is not finite")
            if expected_mask:
                if type(target) not in (int, float) or not math.isfinite(target):
                    raise S3C3ControlError("core target is not finite")
                per_endpoint[task].append((target, prediction))
    calculated = {task: metric_values(values) for task, values in per_endpoint.items()}
    if any(value["n"] < 2 for value in calculated.values()):
        raise S3C3ControlError("core endpoint has fewer than two finite pairs")
    return {
        "per_endpoint": calculated,
        "human3_macro_rmse": math.fsum(
            value["rmse"] for value in calculated.values()
        )
        / len(HUMAN3_TASKS),
    }


def _validate_metrics(stored: Mapping[str, Any], calculated: Mapping[str, Any]) -> None:
    if set(stored.get("per_endpoint", {})) != set(HUMAN3_TASKS):
        raise S3C3ControlError("stored metrics do not cover exactly Human3")
    for task in HUMAN3_TASKS:
        for name, expected in calculated["per_endpoint"][task].items():
            _same_metric(
                stored["per_endpoint"][task].get(name),
                expected,
                f"metrics.{task}.{name}",
            )
    _same_metric(
        stored.get("human3_macro_rmse"),
        calculated["human3_macro_rmse"],
        "metrics.human3_macro_rmse",
    )


def _safe_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise S3C3ControlError(f"unsafe ZIP member name: {name!r}")


def _parse_checksums(data: bytes) -> dict[str, str]:
    result = {}
    for line in data.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise S3C3ControlError("invalid checksums.sha256 line")
        digest, name = match.groups()
        _safe_member_name(name)
        if name in result:
            raise S3C3ControlError(f"duplicate checksum member: {name}")
        result[name] = digest
    return result


@dataclass(frozen=True)
class FrozenTestSample:
    """One ordered Human3 test identity from the verified 034 archive."""

    sample_id: str
    row_index: int
    labels: tuple[int | float | None, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "row_index": self.row_index,
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class FrozenTestSampleBinding:
    """Independent ordered test identity carried out of verified 034 evidence."""

    source_archive_sha256: str
    sample_identity_sha256: str
    samples: tuple[FrozenTestSample, ...]

    def as_json_samples(self) -> list[dict[str, Any]]:
        return [sample.as_json() for sample in self.samples]


@dataclass(frozen=True)
class VerifiedReuseArchive:
    source_path: str
    archive_sha256: str
    selected_members: Mapping[str, bytes]
    verified_identities: tuple[tuple[str, int], ...]
    frozen_test_sample_binding: FrozenTestSampleBinding | None = None


def _verified_zip_members(path: str | Path, expected_sha256: str) -> dict[str, bytes]:
    archive_path = Path(path).resolve()
    if sha256_file(archive_path) != expected_sha256:
        raise S3C3ControlError("reuse ZIP SHA does not match the frozen source")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise S3C3ControlError("reuse ZIP contains duplicate member names")
        if archive.testzip() is not None:
            raise S3C3ControlError("reuse ZIP CRC validation failed")
        file_names = [name for name in names if not name.endswith("/")]
        for name in file_names:
            _safe_member_name(name)
        checksum_names = [
            name for name in file_names if PurePosixPath(name).name == "checksums.sha256"
        ]
        if len(checksum_names) != 1:
            raise S3C3ControlError("reuse ZIP must contain one checksums.sha256")
        members = {name: archive.read(name) for name in file_names}
    checksum_name = checksum_names[0]
    checksums = _parse_checksums(members[checksum_name])
    expected_coverage = set(file_names) - {checksum_name}
    if set(checksums) != expected_coverage:
        raise S3C3ControlError("reuse ZIP checksum coverage is not exact")
    for name, expected in checksums.items():
        if hashlib.sha256(members[name]).hexdigest() != expected:
            raise S3C3ControlError(f"reuse ZIP checksum differs: {name}")
    return members


def _rows_member(members: Mapping[str, bytes], name: str) -> list[Mapping[str, Any]]:
    rows = []
    for line in members[name].decode("utf-8").splitlines():
        if line.strip():
            value = strict_json_loads(line)
            if not isinstance(value, Mapping):
                raise S3C3ControlError(f"prediction row is not an object: {name}")
            rows.append(value)
    return rows


def verify_historical_validations(
    path: str | Path,
    *,
    expected_sha256: str = HISTORICAL_VALIDATION_ZIP_SHA256,
) -> VerifiedReuseArchive:
    members = _verified_zip_members(path, expected_sha256)
    selected: dict[str, bytes] = {}
    identities = []
    for run_id in INHERITED_VALIDATION_RUN_IDS:
        suffix = f"validation_{run_id}/predictions.jsonl"
        matches = [name for name in members if name.endswith(suffix)]
        if len(matches) != 1:
            raise S3C3ControlError(
                f"historical validation member identity is not unique: {run_id}"
            )
        predictions_name = matches[0]
        prefix = predictions_name[: -len("predictions.jsonl")]
        expected_name = prefix + "expected_samples.json"
        metrics_name = prefix + "metrics.json"
        restoration_name = prefix + "restoration.json"
        comparison_name = prefix + "validation_comparison.json"
        for required in (
            expected_name,
            metrics_name,
            restoration_name,
            comparison_name,
        ):
            if required not in members:
                raise S3C3ControlError(
                    f"historical validation evidence is incomplete: {required}"
                )
        asset = PRODUCTION_POLICY.asset(run_id)
        expected_samples = _json_member(members, expected_name)
        rows = _rows_member(members, predictions_name)
        calculated = validate_core_rows(
            rows,
            expected_samples,
            method=asset.method,
            seed=asset.seed,
            split="validation",
        )
        _validate_metrics(_json_member(members, metrics_name), calculated)
        restoration = _json_member(members, restoration_name)
        restored_asset = restoration.get("asset", {})
        checkpoint = restored_asset.get("best_checkpoint", {})
        if (
            restored_asset.get("run_id") != run_id
            or restored_asset.get("method") != asset.method
            or type(restored_asset.get("seed")) is not int
            or restored_asset["seed"] != asset.seed
            or checkpoint.get("sha256") != asset.checkpoint_sha256
            or checkpoint.get("size_bytes") != asset.checkpoint_size_bytes
            or restoration.get("checkpoint_unchanged") is not True
            or restoration.get("model_unchanged") is not True
            or restoration.get("training_called") is not False
            or restoration.get("calibration_called") is not False
        ):
            raise S3C3ControlError(
                f"historical validation restoration identity differs: {run_id}"
            )
        comparison = _json_member(members, comparison_name)
        if (
            comparison.get("status") != "PASS"
            or type(comparison.get("reference_sha256")) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", comparison["reference_sha256"])
            or type(comparison.get("max_abs_difference")) not in (int, float)
            or comparison["max_abs_difference"] > 1e-6
        ):
            raise S3C3ControlError(
                f"historical validation comparison differs: {run_id}"
            )
        for name, data in members.items():
            if name.startswith(prefix):
                selected[name] = data
        identities.append((asset.method, asset.seed))
    if {f"D8_formal_{method}_s{seed}" for method, seed in identities} != set(
        INHERITED_VALIDATION_RUN_IDS
    ):
        raise S3C3ControlError("historical validation identity coverage differs")
    return VerifiedReuseArchive(
        source_path=str(Path(path).resolve()),
        archive_sha256=expected_sha256,
        selected_members=selected,
        verified_identities=tuple(identities),
    )


def _baseline_expected_samples(expected: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (
        not isinstance(expected, Mapping)
        or expected.get("split") != "test"
        or expected.get("task_names") != list(HUMAN3_TASKS)
        or not isinstance(expected.get("samples"), list)
        or not expected["samples"]
    ):
        raise S3C3ControlError("baseline expected test samples are invalid")
    samples = expected["samples"]
    sample_ids = []
    row_indices = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise S3C3ControlError("baseline expected test sample is not an object")
        sample_id = sample.get("sample_id")
        row_index = sample.get("row_index")
        labels = sample.get("labels")
        if type(sample_id) is not str or not sample_id:
            raise S3C3ControlError("baseline expected sample_id is invalid")
        if type(row_index) is not int:
            raise S3C3ControlError("baseline expected row_index is not a true int")
        if not isinstance(labels, list) or len(labels) != len(HUMAN3_TASKS):
            raise S3C3ControlError("baseline expected labels are not Human3 ordered")
        for task_index, label in enumerate(labels):
            if label is not None and (
                type(label) not in (int, float) or not math.isfinite(label)
            ):
                raise S3C3ControlError(
                    "baseline expected label is neither null nor a finite number "
                    f"at sample {sample_index}, task {HUMAN3_TASKS[task_index]}"
                )
        sample_ids.append(sample_id)
        row_indices.append(row_index)
    if len(set(sample_ids)) != len(samples):
        raise S3C3ControlError("baseline expected test samples are duplicated")
    if len(set(row_indices)) != len(samples):
        raise S3C3ControlError("baseline expected test row indices are duplicated")
    return samples


def _frozen_test_sample_binding(
    samples: Sequence[Mapping[str, Any]],
    *,
    source_archive_sha256: str,
) -> FrozenTestSampleBinding:
    frozen_samples = tuple(
        FrozenTestSample(
            sample_id=sample["sample_id"],
            row_index=sample["row_index"],
            labels=tuple(sample["labels"]),
        )
        for sample in samples
    )
    identity_sha256 = _identity_digest(
        [sample.as_json() for sample in frozen_samples]
    )
    return FrozenTestSampleBinding(
        source_archive_sha256=source_archive_sha256,
        sample_identity_sha256=identity_sha256,
        samples=frozen_samples,
    )


def verify_frozen_test_sample_binding(
    actual_samples: Any,
    frozen: FrozenTestSampleBinding,
) -> dict[str, Any]:
    """Compare worker test expectations to the independent 034 ordered table."""

    if not isinstance(actual_samples, list):
        raise S3C3ControlError("core test expected samples are not a list")
    if len(actual_samples) != len(frozen.samples):
        raise S3C3ControlError(
            "core test frozen sample count differs "
            f"({len(actual_samples)} != {len(frozen.samples)})"
        )
    actual_payload = []
    required_keys = {"sample_id", "row_index", "labels"}
    for sample_index, (actual, expected) in enumerate(
        zip(actual_samples, frozen.samples)
    ):
        if not isinstance(actual, Mapping) or set(actual) != required_keys:
            raise S3C3ControlError(
                f"core test expected sample schema differs at index {sample_index}"
            )
        sample_id = actual["sample_id"]
        if type(sample_id) is not str:
            raise S3C3ControlError(
                f"core test sample_id is not a string at index {sample_index}"
            )
        if sample_id != expected.sample_id:
            raise S3C3ControlError(
                f"core test frozen sample_id differs at index {sample_index}"
            )
        row_index = actual["row_index"]
        if type(row_index) is not int:
            raise S3C3ControlError(
                f"core test row_index is not a true int at index {sample_index}"
            )
        if row_index != expected.row_index:
            raise S3C3ControlError(
                f"core test frozen row_index differs at index {sample_index}"
            )
        labels = actual["labels"]
        if not isinstance(labels, list) or len(labels) != len(HUMAN3_TASKS):
            raise S3C3ControlError(
                f"core test labels are not Human3 ordered at index {sample_index}"
            )
        for task_index, (actual_label, expected_label) in enumerate(
            zip(labels, expected.labels)
        ):
            if (
                type(actual_label) is not type(expected_label)
                or actual_label != expected_label
            ):
                raise S3C3ControlError(
                    "core test frozen label value/type differs at sample "
                    f"{sample_index}, task {HUMAN3_TASKS[task_index]}"
                )
        actual_payload.append(
            {
                "sample_id": sample_id,
                "row_index": row_index,
                "labels": list(labels),
            }
        )
    actual_identity_sha256 = _identity_digest(actual_payload)
    if actual_identity_sha256 != frozen.sample_identity_sha256:
        raise S3C3ControlError("core test frozen sample identity hash differs")
    return {
        "status": "VERIFIED",
        "source_archive_sha256": frozen.source_archive_sha256,
        "frozen_test_sample_identity_sha256": frozen.sample_identity_sha256,
        "actual_test_sample_identity_sha256": actual_identity_sha256,
        "sample_count": len(actual_payload),
        "exact_ordered_match": True,
    }


def verify_baseline_tests(
    path: str | Path,
    *,
    expected_sha256: str = BASELINE_TEST_ZIP_SHA256,
) -> VerifiedReuseArchive:
    members = _verified_zip_members(path, expected_sha256)
    expected_names = [name for name in members if name == "expected_test_samples.json"]
    if len(expected_names) != 1:
        raise S3C3ControlError("baseline ZIP expected-sample member differs")
    expected = _json_member(members, expected_names[0])
    samples = _baseline_expected_samples(expected)
    frozen_test_binding = _frozen_test_sample_binding(
        samples,
        source_archive_sha256=expected_sha256,
    )
    manifest_names = sorted(
        name for name in members if name.endswith("/inference_manifest.json")
    )
    if len(manifest_names) != 25:
        raise S3C3ControlError("baseline ZIP does not contain exactly 25 manifests")
    selected = {expected_names[0]: members[expected_names[0]]}
    identities = []
    for manifest_name in manifest_names:
        prefix = manifest_name[: -len("inference_manifest.json")]
        predictions_name = prefix + "predictions.jsonl"
        metrics_name = prefix + "metrics.json"
        sample_manifest_name = prefix + "sample_manifest.json"
        verification_name = prefix + "verification.json"
        for required in (
            predictions_name,
            metrics_name,
            sample_manifest_name,
            verification_name,
        ):
            if required not in members:
                raise S3C3ControlError(f"baseline run evidence is incomplete: {required}")
        manifest = _json_member(members, manifest_name)
        method, seed = manifest.get("method"), manifest.get("seed")
        if type(method) is not str or type(seed) is not int:
            raise S3C3ControlError("baseline manifest method/seed types are invalid")
        if (method, seed) not in BASELINE_IDENTITIES:
            raise S3C3ControlError("baseline manifest identity is not authorized")
        if manifest.get("split") != "test" or manifest.get("task_order") != list(HUMAN3_TASKS):
            raise S3C3ControlError("baseline manifest split/task order differs")
        if manifest.get("sample_count") != len(samples):
            raise S3C3ControlError("baseline manifest sample count differs")
        rows = _rows_member(members, predictions_name)
        if manifest.get("prediction_row_count") != len(rows):
            raise S3C3ControlError("baseline manifest prediction count differs")
        expected_rows = []
        asset_id = manifest.get("asset_id")
        sample_manifest_rows = []
        for sample_index, sample in enumerate(samples):
            if sample.get("sample_order") != sample_index or sample.get("split") != "test":
                raise S3C3ControlError("baseline expected sample order/split differs")
            labels = sample.get("labels")
            if not isinstance(labels, list) or len(labels) != len(HUMAN3_TASKS):
                raise S3C3ControlError("baseline expected labels are invalid")
            task_mask = {
                task: labels[task_index] is not None
                for task_index, task in enumerate(HUMAN3_TASKS)
            }
            sample_manifest_rows.append(
                {
                    "sample_order": sample_index,
                    "sample_id": sample["sample_id"],
                    "row_index": sample["row_index"],
                    "split": "test",
                    "task_mask": task_mask,
                }
            )
            for task_index, task in enumerate(HUMAN3_TASKS):
                row = rows[sample_index * len(HUMAN3_TASKS) + task_index]
                expected_keys = {
                    "asset_id",
                    "method",
                    "seed",
                    "sample_id",
                    "row_index",
                    "split",
                    "endpoint",
                    "y_true",
                    "y_pred",
                    "mask",
                }
                if set(row) != expected_keys:
                    raise S3C3ControlError("baseline prediction schema differs")
                if row["asset_id"] != asset_id:
                    raise S3C3ControlError("baseline prediction asset identity differs")
                expected_rows.append(
                    {
                        "method": row["method"],
                        "seed": row["seed"],
                        "sample_id": row["sample_id"],
                        "row_index": row["row_index"],
                        "split": row["split"],
                        "endpoint": row["endpoint"],
                        "y_true": row["y_true"],
                        "y_pred": row["y_pred"],
                        "mask": row["mask"],
                    }
                )
        calculated = validate_core_rows(
            expected_rows, samples, method=method, seed=seed, split="test"
        )
        _validate_metrics(_json_member(members, metrics_name), calculated)
        sample_identity = {
            "split": "test",
            "task_names": list(HUMAN3_TASKS),
            "samples": sample_manifest_rows,
        }
        sample_identity_hash = _identity_digest(sample_identity)
        expected_sample_manifest = {
            "schema_version": 1,
            **sample_identity,
            "sample_count": len(sample_manifest_rows),
            "sample_identity_hash": sample_identity_hash,
        }
        if _json_member(members, sample_manifest_name) != expected_sample_manifest:
            raise S3C3ControlError("baseline sample manifest identity differs")
        if manifest.get("sample_identity_hash") != sample_identity_hash:
            raise S3C3ControlError("baseline inference manifest sample identity differs")
        checkpoint_identity = manifest.get("checkpoint")
        if (
            not isinstance(checkpoint_identity, Mapping)
            or type(checkpoint_identity.get("path")) is not str
            or type(checkpoint_identity.get("bytes")) is not int
            or checkpoint_identity["bytes"] <= 0
            or type(checkpoint_identity.get("sha256")) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_identity["sha256"])
        ):
            raise S3C3ControlError("baseline checkpoint identity is invalid")
        verification = _json_member(members, verification_name)
        required_true = (
            "expected_coverage_exact",
            "label_masks_verified",
            "finite_predictions_verified",
            "independent_metric_calculation",
            "manifest_identity_verified",
            "input_checkpoint_unchanged",
        )
        if verification.get("status") != "VERIFIED_EXECUTION" or any(
            verification.get(name) is not True for name in required_true
        ):
            raise S3C3ControlError("baseline execution verification differs")
        if (
            verification.get("checkpoint_sha256_before")
            != checkpoint_identity["sha256"]
            or verification.get("checkpoint_sha256_after")
            != checkpoint_identity["sha256"]
        ):
            raise S3C3ControlError("baseline checkpoint before/after identity differs")
        for name, data in members.items():
            if name.startswith(prefix):
                selected[name] = data
        identities.append((method, seed))
    if len(identities) != len(set(identities)) or set(identities) != BASELINE_IDENTITIES:
        raise S3C3ControlError("baseline 25-run identity coverage differs")
    return VerifiedReuseArchive(
        source_path=str(Path(path).resolve()),
        archive_sha256=expected_sha256,
        selected_members=selected,
        verified_identities=tuple(identities),
        frozen_test_sample_binding=frozen_test_binding,
    )


def write_verified_reuse(archive: VerifiedReuseArchive, destination: str | Path) -> None:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name, data in archive.selected_members.items():
        target = (root / PurePosixPath(name)).resolve()
        if not target.is_relative_to(root):
            raise S3C3ControlError(f"reuse member escapes destination: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def plan_after_validation(
    *,
    historical_validation_verified: bool,
    new_validation_results: Mapping[str, bool],
    completed_core_tests: Sequence[str] = (),
) -> dict[str, Any]:
    unknown_validation = set(new_validation_results) - set(NEW_VALIDATION_RUN_IDS)
    unknown_tests = set(completed_core_tests) - set(CORE_RUN_IDS)
    if unknown_validation or unknown_tests or len(completed_core_tests) != len(set(completed_core_tests)):
        raise S3C3ControlError("batch state contains unknown or duplicate run identities")
    failed = sorted(
        run_id for run_id, passed in new_validation_results.items() if passed is not True
    )
    pending_validation = [
        run_id for run_id in NEW_VALIDATION_RUN_IDS if run_id not in new_validation_results
    ]
    validation_gate_passed = (
        historical_validation_verified
        and not failed
        and not pending_validation
        and all(new_validation_results.values())
    )
    test_to_schedule = (
        [run_id for run_id in CORE_RUN_IDS if run_id not in completed_core_tests]
        if validation_gate_passed
        else []
    )
    return {
        "historical_validation_verified": historical_validation_verified,
        "inherited_validation_runs": (
            list(INHERITED_VALIDATION_RUN_IDS)
            if historical_validation_verified
            else []
        ),
        "new_validation_completed": sorted(new_validation_results),
        "new_validation_failed": failed,
        "new_validation_pending": pending_validation,
        "validation_gate_passed": validation_gate_passed,
        "completed_core_tests_retained": list(completed_core_tests),
        "core_tests_to_schedule": test_to_schedule,
        "baseline_tests_to_reuse": 25,
    }
