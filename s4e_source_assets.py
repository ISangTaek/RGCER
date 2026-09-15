"""Read-only S4E source-asset collection primitives.

This module never constructs a model, performs a forward pass, selects a
mechanism probe, or reads prediction files.  It binds the fifty frozen target
runs to their exact best checkpoint, actual initialization, provenance sidecar,
and source teacher before projecting only the metadata and tensor identities
needed for the later Codex scientific freeze.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS


SCHEMA_VERSION = 1
METHODS = ("B1", "RPT")
CORE_METHODS = ("B0", "B1", "RPT")
SEEDS = tuple(range(42, 47))
LOW_FRACTIONS = (10, 25, 50, 75)
ADAPTATION_FIELDS = (
    "d6_candidate",
    "d7_candidate",
    "freeze_backbone_epochs",
    "backbone_lr_multiplier",
    "trainable_last_blocks",
    "retention_probe_per_task",
    "retention_damage_threshold",
    "feature_drift_probe_size",
    "feature_drift_epochs",
)
METHOD_CANDIDATE_IDENTITY = {
    "B1": {"d6_candidate": "b1", "d7_candidate": "none"},
    "RPT": {"d6_candidate": "none", "d7_candidate": "s1"},
}
BACKBONE_PREFIX = "encoder.backbone."
ENCODER_PREFIX = "encoder."
DECODER_PREFIX = "decoders."
BUFFER_SUFFIXES = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)
SECRET_KEY = re.compile(
    r"(^|_)(password|passwd|secret|api_key|access_token|private_key|credential|authorization)($|_)",
    re.IGNORECASE,
)
LABEL_KEY = re.compile(
    r"(^|_)(label|labels|target|targets|prediction|predictions|y_true|y_pred)($|_)",
    re.IGNORECASE,
)
SAFE_EVIDENCE_SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".tsv", ".sha256"}
TRAINING_REFERENCE_MARKERS = ("sample", "subset", "split", "manifest")
SCOPE_FIELD_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "task_names": (
        ("task_names",),
        ("architecture_config", "task_names"),
        ("data_config", "task_names"),
    ),
    "dataset": (
        ("dataset",),
        ("configuration", "dataset"),
        ("args", "dataset"),
    ),
    "datastore_fingerprint": (
        ("datastore_fingerprint",),
        ("data_config", "datastore_fingerprint"),
    ),
    "split_manifest_hash": (
        ("split_manifest_hash",),
        ("split_manifest_sha256",),
        ("data_config", "split_manifest_hash"),
    ),
    "feature_schema_version": (
        ("feature_schema_version",),
        ("data_config", "feature_schema_version"),
    ),
    "max_nodes_filter": (
        ("max_nodes_filter",),
        ("configuration", "max_nodes_filter"),
        ("data_config", "max_nodes_filter"),
        ("args", "max_nodes_filter"),
    ),
    "splitting": (
        ("splitting",),
        ("configuration", "splitting"),
        ("data_config", "splitting"),
        ("args", "splitting"),
    ),
    "split_seed": (
        ("split_seed",),
        ("configuration", "split_seed"),
        ("data_config", "split_seed"),
        ("args", "split_seed"),
    ),
    "task_sampling": (
        ("task_sampling",),
        ("configuration", "task_sampling"),
        ("args", "task_sampling"),
    ),
    "train_fraction": (
        ("train_fraction",),
        ("configuration", "train_fraction"),
        ("args", "train_fraction"),
    ),
    "train_eval_scope": (
        ("train_eval_scope",),
        ("configuration", "train_eval_scope"),
        ("args", "train_eval_scope"),
    ),
    "fit_conformal": (
        ("fit_conformal",),
        ("configuration", "fit_conformal"),
        ("args", "fit_conformal"),
    ),
}
REQUIRED_SCOPE_FIELDS = frozenset(SCOPE_FIELD_PATHS)
_MISSING = object()


class S4ECollectionError(RuntimeError):
    """Base class for fail-closed collection errors."""


class RunCollectionError(S4ECollectionError):
    """A bounded per-run error that must be preserved in the output."""

    def __init__(self, code: str, reason: str, *, status: str = "ERROR") -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.status = status


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    seed: int
    fraction: int
    run_dir: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    best_epoch: int
    lock_init_state_path: str | None
    expected_adaptation: Mapping[str, Any]
    lock_source: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S4ECollectionError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_loads(text: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise S4ECollectionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise S4ECollectionError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise S4ECollectionError(f"invalid JSON: {exc}") from exc


def read_json_document(path: str | Path) -> tuple[Any, bytes, str]:
    source = Path(path)
    data = source.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise S4ECollectionError(f"JSON is not UTF-8: {source}") from exc
    return strict_json_loads(text), data, sha256_bytes(data)


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _strict_string(value: Any, field: str) -> str:
    require(type(value) is str and bool(value), f"{field} must be a non-empty string")
    return value


def _strict_int(value: Any, field: str) -> int:
    require(type(value) is int, f"{field} must be an integer, not {type(value).__name__}")
    return value


def _strict_bool(value: Any, field: str) -> bool:
    require(type(value) is bool, f"{field} must be a JSON boolean")
    return value


def _strict_sha256(value: Any, field: str) -> str:
    value = _strict_string(value, field)
    require(bool(re.fullmatch(r"[0-9a-f]{64}", value)), f"{field} must be lowercase SHA-256")
    return value


def _safe_run_id(value: Any, field: str) -> str:
    value = _strict_string(value, field)
    require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value)), f"{field} is not path-safe")
    return value


def _checkpoint_identity(document: Mapping[str, Any], field: str) -> tuple[str, str, int]:
    require(type(document) is dict, f"{field} must be an object")
    path = _strict_string(document.get("path"), f"{field}.path")
    digest = _strict_sha256(document.get("sha256"), f"{field}.sha256")
    size = _strict_int(document.get("size_bytes"), f"{field}.size_bytes")
    require(size > 0, f"{field}.size_bytes must be positive")
    return path, digest, size


def _validate_expected_adaptation(
    document: Any,
    field: str,
    method: str,
    *,
    require_all_fields: bool,
) -> dict[str, Any]:
    require(type(document) is dict, f"{field} must be an object")
    if require_all_fields:
        missing = [name for name in ADAPTATION_FIELDS if name not in document]
        require(not missing, f"{field} is missing frozen fields: {missing}")
    expected = {
        name: document[name]
        for name in ADAPTATION_FIELDS
        if name in document
    }
    for required in ("d6_candidate", "d7_candidate", "freeze_backbone_epochs"):
        require(required in expected, f"{field}.{required} is required")
    for name, value in expected.items():
        value_field = f"{field}.{name}"
        if name in {"d6_candidate", "d7_candidate", "feature_drift_epochs"}:
            _strict_string(value, value_field)
        elif name in {
            "freeze_backbone_epochs",
            "trainable_last_blocks",
            "retention_probe_per_task",
            "feature_drift_probe_size",
        }:
            number = _strict_int(value, value_field)
            require(number >= 0, f"{value_field} must be non-negative")
        else:
            require(
                type(value) in (int, float) and math.isfinite(float(value)),
                f"{value_field} must be a finite JSON number",
            )
    identity = METHOD_CANDIDATE_IDENTITY[method]
    for name, value in identity.items():
        require(
            expected.get(name) == value,
            f"{field}.{name}={expected.get(name)!r} conflicts with frozen {method} identity {value!r}",
        )
    return expected


def build_run_specs(low_policy: Any, core_lock: Any) -> list[RunSpec]:
    require(type(low_policy) is dict, "low policy must be an object")
    require(type(core_lock) is dict, "core lock must be an object")
    low_assets = low_policy.get("assets")
    core_assets = core_lock.get("assets")
    require(type(low_assets) is list, "low policy assets must be a list")
    require(type(core_assets) is list, "core lock assets must be a list")

    low_expected = {
        (method, seed, fraction)
        for method in METHODS
        for seed in SEEDS
        for fraction in LOW_FRACTIONS
    }
    core_expected = {
        (method, seed)
        for method in CORE_METHODS
        for seed in SEEDS
    }
    low_seen: set[tuple[str, int, int]] = set()
    core_seen: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    specs: list[RunSpec] = []

    for index, asset in enumerate(low_assets):
        field = f"low_policy.assets[{index}]"
        require(type(asset) is dict, f"{field} must be an object")
        method = _strict_string(asset.get("method"), f"{field}.method")
        seed = _strict_int(asset.get("seed"), f"{field}.seed")
        fraction = _strict_int(asset.get("fraction_percent"), f"{field}.fraction_percent")
        require((method, seed, fraction) in low_expected, f"{field} has illegal matrix identity")
        require((method, seed, fraction) not in low_seen, f"duplicate low matrix identity: {(method, seed, fraction)}")
        low_seen.add((method, seed, fraction))
        run_id = _safe_run_id(asset.get("run_id"), f"{field}.run_id")
        require(run_id not in run_ids, f"duplicate run_id: {run_id}")
        run_ids.add(run_id)
        checkpoint_path, checkpoint_sha, checkpoint_size = _checkpoint_identity(
            asset.get("checkpoint"), f"{field}.checkpoint"
        )
        expected_configuration = asset.get("expected_configuration")
        require(type(expected_configuration) is dict, f"{field}.expected_configuration must be an object")
        expected_adaptation = _validate_expected_adaptation(
            asset.get("expected_adaptation"),
            f"{field}.expected_adaptation",
            method,
            require_all_fields=True,
        )
        migration_required = asset.get("migration_required")
        if migration_required is not None:
            _strict_bool(migration_required, f"{field}.migration_required")
        init_path = expected_configuration.get("init_state_path")
        if init_path is not None:
            init_path = _strict_string(init_path, f"{field}.expected_configuration.init_state_path")
        specs.append(
            RunSpec(
                run_id=run_id,
                method=method,
                seed=seed,
                fraction=fraction,
                run_dir=_strict_string(asset.get("run_dir"), f"{field}.run_dir"),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_size_bytes=checkpoint_size,
                best_epoch=_strict_int(asset.get("best_epoch"), f"{field}.best_epoch"),
                lock_init_state_path=init_path,
                expected_adaptation=expected_adaptation,
                lock_source="low_policy",
            )
        )

    for index, asset in enumerate(core_assets):
        field = f"core_lock.assets[{index}]"
        require(type(asset) is dict, f"{field} must be an object")
        method = _strict_string(asset.get("method"), f"{field}.method")
        seed = _strict_int(asset.get("seed"), f"{field}.seed")
        require((method, seed) in core_expected, f"{field} has illegal matrix identity")
        require((method, seed) not in core_seen, f"duplicate core matrix identity: {(method, seed)}")
        core_seen.add((method, seed))
        if method == "B0":
            continue
        run_id = _safe_run_id(asset.get("run_id"), f"{field}.run_id")
        require(run_id not in run_ids, f"duplicate run_id: {run_id}")
        run_ids.add(run_id)
        checkpoint_path, checkpoint_sha, checkpoint_size = _checkpoint_identity(
            asset.get("best_checkpoint"), f"{field}.best_checkpoint"
        )
        config = asset.get("config")
        require(type(config) is dict, f"{field}.config must be an object")
        expected_adaptation = _validate_expected_adaptation(
            config,
            f"{field}.config",
            method,
            require_all_fields=False,
        )
        init_path = config.get("init_state_path")
        if init_path is not None:
            init_path = _strict_string(init_path, f"{field}.config.init_state_path")
        specs.append(
            RunSpec(
                run_id=run_id,
                method=method,
                seed=seed,
                fraction=100,
                run_dir=_strict_string(asset.get("run_dir"), f"{field}.run_dir"),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_size_bytes=checkpoint_size,
                best_epoch=_strict_int(asset.get("best_epoch"), f"{field}.best_epoch"),
                lock_init_state_path=init_path,
                expected_adaptation=expected_adaptation,
                lock_source="core_lock",
            )
        )

    require(low_seen == low_expected, "low policy does not contain the exact 40-run matrix")
    require(core_seen == core_expected, "core lock does not contain the exact B0/B1/RPT five-seed matrix")
    require(len(specs) == 50, "combined source mapping must contain exactly 50 B1/RPT runs")
    combined = {(row.method, row.seed, row.fraction) for row in specs}
    expected_combined = low_expected | {
        (method, seed, 100) for method in METHODS for seed in SEEDS
    }
    require(combined == expected_combined, "combined source mapping is incomplete")
    return sorted(specs, key=lambda row: (row.method, row.fraction, row.seed, row.run_id))


def json_safe(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        return {"value": None, "reason": f"nonfinite_float:{value!r}"}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                return {"value": None, "reason": "mapping_has_non_string_key"}
            result[key] = json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return {
        "value": None,
        "reason": f"unsupported_type:{type(value).__module__}.{type(value).__name__}",
    }


def contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is str and SECRET_KEY.search(key):
                return True
            if contains_secret_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(contains_secret_key(item) for item in value)
    return False


def _redact_label_fields(
    value: Any,
    prefix: tuple[str, ...] = (),
) -> tuple[Any, list[str]]:
    """Remove label-like mapping fields without retaining their values."""
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        removed: list[str] = []
        for key, item in value.items():
            require(type(key) is str, "training evidence mapping key must be a string")
            field_path = prefix + (key,)
            if LABEL_KEY.search(key):
                removed.append(".".join(field_path))
                continue
            projected_item, item_removed = _redact_label_fields(item, field_path)
            projected[key] = projected_item
            removed.extend(item_removed)
        return projected, removed
    if isinstance(value, list):
        projected_items: list[Any] = []
        removed = []
        for index, item in enumerate(value):
            projected_item, item_removed = _redact_label_fields(
                item,
                prefix + (f"[{index}]",),
            )
            projected_items.append(projected_item)
            removed.extend(item_removed)
        return projected_items, removed
    return value, []


def _json_projection_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _collect_structured_training_evidence(
    path: Path,
    evidence: "EvidenceWriter",
    destination: str,
) -> dict[str, Any]:
    """Copy safe training identity evidence, projecting out label-like values."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        parsed, _, _ = read_json_document(path)
        if contains_secret_key(parsed):
            return evidence.copy(
                path,
                destination,
                parsed=parsed,
                category="training_scope",
            )
        projected, removed = _redact_label_fields(parsed)
        if removed:
            return evidence.write_projection(
                path,
                destination,
                _json_projection_bytes(projected),
                category="training_scope",
                removed_fields=removed,
            )
        return evidence.copy(
            path,
            destination,
            parsed=parsed,
            category="training_scope",
        )

    if suffix == ".jsonl":
        try:
            text = path.read_bytes().decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise S4ECollectionError(f"JSONL is not UTF-8: {path}") from exc
        parsed_rows: list[Any] = []
        projected_rows: list[Any] = []
        removed: list[str] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            parsed = strict_json_loads(line)
            parsed_rows.append(parsed)
            projected, line_removed = _redact_label_fields(
                parsed,
                (f"line[{line_number}]",),
            )
            projected_rows.append(projected)
            removed.extend(line_removed)
        if contains_secret_key(parsed_rows):
            return evidence.copy(
                path,
                destination,
                parsed=parsed_rows,
                category="training_scope",
            )
        if removed:
            data = b"".join(
                (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    + "\n"
                ).encode("utf-8")
                for row in projected_rows
            )
            return evidence.write_projection(
                path,
                destination,
                data,
                category="training_scope",
                removed_fields=removed,
            )
        return evidence.copy(
            path,
            destination,
            parsed=parsed_rows,
            category="training_scope",
        )

    if suffix in {".csv", ".tsv"}:
        try:
            text = path.read_bytes().decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise S4ECollectionError(f"delimited evidence is not UTF-8: {path}") from exc
        delimiter = "," if suffix == ".csv" else "\t"
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True))
        if not rows:
            return evidence.copy(path, destination, category="training_scope")
        header = rows[0]
        require(len(header) == len(set(header)), f"duplicate delimited header in {path}")
        header_projection = {field: None for field in header}
        if contains_secret_key(header_projection):
            return evidence.copy(
                path,
                destination,
                parsed=header_projection,
                category="training_scope",
            )
        label_indexes = [index for index, field in enumerate(header) if LABEL_KEY.search(field)]
        if not label_indexes:
            return evidence.copy(
                path,
                destination,
                parsed=header_projection,
                category="training_scope",
            )
        for row_number, row in enumerate(rows[1:], start=2):
            require(
                len(row) == len(header),
                f"delimited evidence row {row_number} has {len(row)} fields; expected {len(header)}",
            )
        keep_indexes = [index for index in range(len(header)) if index not in label_indexes]
        output = io.StringIO(newline="")
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
        for row in rows:
            writer.writerow([row[index] for index in keep_indexes])
        return evidence.write_projection(
            path,
            destination,
            output.getvalue().encode("utf-8"),
            category="training_scope",
            removed_fields=[header[index] for index in label_indexes],
        )

    return evidence.copy(path, destination, category="training_scope")


def resolve_reference_path(reference: Any, repo_root: Path, field: str) -> Path:
    text = _strict_string(reference, field)
    if re.match(r"^[A-Za-z]:[\\/]", text):
        path = Path(text)
    elif PurePosixPath(text).is_absolute():
        if os.name == "nt":
            raise RunCollectionError(
                "POSIX_PATH_NOT_LOCAL",
                f"{field} is an absolute Linux path on Windows; refusing to join it to repo-root: {text}",
            )
        path = Path(text)
    else:
        candidate = Path(text)
        path = candidate if candidate.is_absolute() else repo_root / candidate
    return path.resolve(strict=False)


def _nested(document: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = document
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _nested_present(document: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = document
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _reference_candidates(
    sources: Sequence[tuple[str, Mapping[str, Any], Sequence[Sequence[str]]]],
    repo_root: Path,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for source_name, document, paths in sources:
        for path in paths:
            value = _nested(document, path)
            if value is None:
                continue
            field = source_name + "." + ".".join(path)
            resolved = resolve_reference_path(value, repo_root, field)
            candidates.append(
                {"source": field, "declared": str(value), "resolved": str(resolved)}
            )
    return candidates


def require_single_reference(candidates: Sequence[Mapping[str, str]], label: str) -> Path:
    require(bool(candidates), f"{label} reference is missing")
    values = {candidate["resolved"] for candidate in candidates}
    if len(values) != 1:
        details = "; ".join(
            f"{candidate['source']}={candidate['declared']}" for candidate in candidates
        )
        raise RunCollectionError(
            "REFERENCE_CONFLICT",
            f"{label} references disagree: {details}",
            status="CONFLICT",
        )
    return Path(next(iter(values)))


class EvidenceWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.records: list[dict[str, Any]] = []
        self._destinations: set[str] = set()

    def copy(
        self,
        source: Path,
        relative_destination: str,
        *,
        parsed: Any | None = None,
        category: str,
    ) -> dict[str, Any]:
        source = source.resolve()
        destination = self.root / PurePosixPath(relative_destination)
        destination_key = destination.relative_to(self.root).as_posix()
        if destination_key in self._destinations:
            previous = next(r for r in self.records if r.get("evidence_path") == "evidence/" + destination_key)
            require(previous.get("source_path") == str(source), f"evidence source identity changed: {destination_key}")
            require(sha256_file(source) == previous["sha256"] == sha256_file(destination),
                    f"evidence bytes changed: {destination_key}")
            return dict(previous, reused=True)
        record: dict[str, Any] = {
            "category": category,
            "source_path": str(source),
            "evidence_path": "evidence/" + destination_key,
        }
        if parsed is not None and contains_secret_key(parsed):
            record.update(
                status="BLOCKED_SENSITIVE",
                sha256=sha256_file(source),
                size_bytes=source.stat().st_size,
                reason="credential-like JSON key detected; bytes not copied",
            )
            self.records.append(record)
            return record
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
        source_sha = sha256_file(source)
        copied_sha = sha256_file(destination)
        require(source_sha == copied_sha, f"evidence copy changed bytes: {source}")
        record.update(
            status="COPIED",
            sha256=source_sha,
            size_bytes=source.stat().st_size,
        )
        self._destinations.add(destination_key)
        self.records.append(record)
        return record

    def write_projection(
        self,
        source: Path,
        relative_destination: str,
        data: bytes,
        *,
        category: str,
        removed_fields: Sequence[str],
    ) -> dict[str, Any]:
        source = source.resolve()
        destination = self.root / PurePosixPath(relative_destination)
        destination_key = destination.relative_to(self.root).as_posix()
        require(destination_key not in self._destinations, f"duplicate evidence destination: {destination_key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(data)
        record = {
            "category": category,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "source_size_bytes": source.stat().st_size,
            "evidence_path": "evidence/" + destination_key,
            "status": "COPIED_PROJECTED_WITHOUT_LABEL_VALUES",
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
            "removed_fields": list(removed_fields),
        }
        self._destinations.add(destination_key)
        self.records.append(record)
        return record


def read_json_evidence(
    path: Path,
    evidence: EvidenceWriter,
    destination: str,
    *,
    category: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise RunCollectionError("MISSING_METADATA", f"required JSON file is missing: {path}")
    document, _, _ = read_json_document(path)
    if type(document) is not dict:
        raise RunCollectionError("INVALID_METADATA", f"JSON root must be an object: {path}")
    record = evidence.copy(path, destination, parsed=document, category=category)
    if record["status"] != "COPIED":
        raise RunCollectionError("SENSITIVE_METADATA", record["reason"])
    return document, record


def _verify_file_identity(
    path: Path,
    expected_sha256: str,
    *,
    expected_size_bytes: int | None = None,
    label: str,
) -> tuple[str, int]:
    if not path.is_file():
        raise RunCollectionError("MISSING_ASSET", f"{label} is missing: {path}")
    size = path.stat().st_size
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise RunCollectionError(
            "SIZE_MISMATCH",
            f"{label} size mismatch: {size} != {expected_size_bytes}",
        )
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RunCollectionError(
            "SHA256_MISMATCH",
            f"{label} SHA-256 mismatch: {digest} != {expected_sha256}",
        )
    return digest, size


def _require_asset_within_repository(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RunCollectionError(
            "ASSET_OUTSIDE_REPOSITORY",
            f"{label} is outside the explicitly supplied repository: {resolved}",
        ) from exc
    return resolved


def verify_scoped_asset_path(
    path: Path,
    expected_sha256: str,
    repo_root: Path,
    *,
    expected_size_bytes: int | None = None,
    label: str,
    mismatch_code: str = "SHA256_MISMATCH",
    mismatch_status: str = "ERROR",
) -> dict[str, Any]:
    resolved = _require_asset_within_repository(path, repo_root, label)
    try:
        digest, size = _verify_file_identity(
            resolved,
            expected_sha256,
            expected_size_bytes=expected_size_bytes,
            label=label,
        )
    except RunCollectionError as exc:
        if exc.code in {"SHA256_MISMATCH", "SIZE_MISMATCH"}:
            raise RunCollectionError(
                mismatch_code,
                exc.reason,
                status=mismatch_status,
            ) from exc
        raise
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": size,
        "status": "VERIFIED",
    }


def load_verified_torch(
    path: Path,
    expected_sha256: str,
    repo_root: Path,
    *,
    expected_size_bytes: int | None = None,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    """Hash first, then deserialize on CPU without constructing a model."""

    verification = verify_scoped_asset_path(
        path,
        expected_sha256,
        repo_root,
        expected_size_bytes=expected_size_bytes,
        label=label,
    )
    digest = verification["sha256"]
    size = verification["size_bytes"]
    path = Path(verification["path"])

    import torch

    mode = "weights_only_true"
    safe_error: dict[str, str] | None = None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # controlled compatibility for hash-locked originals
        safe_error = {"type": type(exc).__name__, "message": str(exc)}
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mode = "verified_project_original_weights_only_false"
    if sha256_file(path) != digest:
        raise RunCollectionError("ASSET_CHANGED_DURING_READ", f"{label} changed during deserialization")
    return payload, {
        "load_mode": mode,
        "safe_load_error": safe_error,
        "map_location": "cpu",
        "sha256": digest,
        "size_bytes": size,
        "model_constructed": False,
        "optimizer_projected": False,
    }


def _state_dict(payload: Any, *, raw_allowed: bool, label: str) -> Mapping[str, Any]:
    if raw_allowed and isinstance(payload, Mapping) and payload and all(
        type(key) is str and hasattr(value, "detach") for key, value in payload.items()
    ):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("model_state"), Mapping):
        return payload["model_state"]
    raise RunCollectionError("MISSING_MODEL_STATE", f"{label} does not contain a tensor state mapping")


def tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise RunCollectionError("NON_TENSOR_STATE", f"state value is not a tensor: {name}")
    if tensor.is_sparse or tensor.is_quantized:
        raise RunCollectionError("UNSUPPORTED_TENSOR", f"unsupported sparse/quantized tensor: {name}")
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    if name.endswith(BUFFER_SUFFIXES):
        role = "known_buffer_by_state_name"
    else:
        role = "parameter_or_buffer_not_persisted_by_state_dict"
    return {
        "key": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "tensor_sha256": sha256_bytes(raw),
        "digest_encoding": "torch_contiguous_uint8_bytes_v1",
        "state_role": role,
    }


def tensor_inventory(state: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    records = [tensor_record(key, state[key]) for key in sorted(keys)]
    merged = hashlib.sha256()
    for record in records:
        merged.update(record["key"].encode("utf-8"))
        merged.update(b"\0")
        merged.update(json.dumps(record["shape"], separators=(",", ":")).encode("ascii"))
        merged.update(b"\0")
        merged.update(record["dtype"].encode("ascii"))
        merged.update(b"\0")
        merged.update(record["tensor_sha256"].encode("ascii"))
        merged.update(b"\n")
    return {
        "tensor_count": len(records),
        "merged_sha256": merged.hexdigest(),
        "merged_digest_encoding": "ordered_key_shape_dtype_tensor_sha256_v1",
        "tensors": records,
    }


def compare_tensor_states(
    teacher_state: Mapping[str, Any],
    init_state: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    import torch

    teacher_keys = {key for key in teacher_state if key.startswith(prefix)}
    init_keys = {key for key in init_state if key.startswith(prefix)}
    missing_in_init = sorted(teacher_keys - init_keys)
    unexpected_in_init = sorted(init_keys - teacher_keys)
    rows: list[dict[str, Any]] = []
    for key in sorted(teacher_keys & init_keys):
        teacher = teacher_state[key]
        init = init_state[key]
        if not isinstance(teacher, torch.Tensor) or not isinstance(init, torch.Tensor):
            status = "NON_TENSOR"
        elif tuple(teacher.shape) != tuple(init.shape):
            status = "SHAPE_MISMATCH"
        elif teacher.dtype != init.dtype:
            status = "DTYPE_MISMATCH"
        elif not torch.equal(teacher.detach().cpu(), init.detach().cpu()):
            status = "VALUE_MISMATCH"
        else:
            status = "EQUAL"
        rows.append(
            {
                "key": key,
                "status": status,
                "teacher_shape": list(getattr(teacher, "shape", [])),
                "init_shape": list(getattr(init, "shape", [])),
                "teacher_dtype": str(getattr(teacher, "dtype", type(teacher).__name__)),
                "init_dtype": str(getattr(init, "dtype", type(init).__name__)),
            }
        )
    mismatches = [row for row in rows if row["status"] != "EQUAL"]
    return {
        "prefix": prefix,
        "teacher_key_count": len(teacher_keys),
        "init_key_count": len(init_keys),
        "missing_in_init": missing_in_init,
        "unexpected_in_init": unexpected_in_init,
        "per_key": rows,
        "mismatch_count": len(missing_in_init) + len(unexpected_in_init) + len(mismatches),
        "status": "MATCH" if not missing_in_init and not unexpected_in_init and not mismatches else "CONFLICT",
    }


def _validate_task_names(value: Any, *, expected_count: int, label: str) -> list[str]:
    if type(value) is not list or len(value) != expected_count:
        raise RunCollectionError(
            "TASK_SCOPE_MISMATCH",
            f"{label} must contain exactly {expected_count} ordered task names",
        )
    if any(type(item) is not str or not item for item in value) or len(set(value)) != len(value):
        raise RunCollectionError("TASK_SCOPE_MISMATCH", f"{label} has invalid or duplicate task names")
    return list(value)


def summarize_init_asset(path: Path, digest: str, payload: Any, load: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    state = _state_dict(payload, raw_allowed=True, label="init")
    backbone_keys = [key for key in state if key.startswith(BACKBONE_PREFIX)]
    encoder_other_keys = [
        key for key in state if key.startswith(ENCODER_PREFIX) and not key.startswith(BACKBONE_PREFIX)
    ]
    if not backbone_keys:
        raise RunCollectionError("MISSING_BACKBONE", "init contains no encoder.backbone tensors")
    return (
        {
            "asset_id": "init:" + digest,
            "kind": "init",
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "load": dict(load),
            "backbone": tensor_inventory(state, backbone_keys),
            "encoder_non_backbone": tensor_inventory(state, encoder_other_keys),
            "readout_keys": sorted(key for key in encoder_other_keys if "readout" in key.lower()),
            "state_role_note": "state_dict does not persist parameter-versus-buffer ownership; known buffer suffixes are marked and all others remain explicitly unresolved",
        },
        state,
    )


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def summarize_teacher_asset(
    path: Path,
    digest: str,
    payload: Any,
    load: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise RunCollectionError("INVALID_TEACHER", "teacher checkpoint root is not a mapping")
    state = _state_dict(payload, raw_allowed=False, label="teacher")
    tasks = _validate_task_names(payload.get("task_names"), expected_count=56, label="teacher.task_names")
    canonical_tasks = list(ANIMAL_SOURCE_TASKS)
    if tasks != canonical_tasks:
        missing = [task for task in canonical_tasks if task not in tasks]
        extra = [task for task in tasks if task not in canonical_tasks]
        code = "TASK_ORDER_MISMATCH" if not missing and not extra else "TASK_SCOPE_MISMATCH"
        raise RunCollectionError(
            code,
            "teacher.task_names differs from architecture.toxacute_tasks.ANIMAL_SOURCE_TASKS; "
            f"missing={missing}, extra={extra}, order_matches={not missing and not extra}",
        )
    architecture_config = payload.get("architecture_config")
    if isinstance(architecture_config, Mapping):
        architecture_tasks = _nested_present(architecture_config, ("task_names",))
        if architecture_tasks is not _MISSING:
            persisted_tasks = _validate_task_names(
                architecture_tasks,
                expected_count=56,
                label="teacher.architecture_config.task_names",
            )
            if persisted_tasks != canonical_tasks:
                missing = [task for task in canonical_tasks if task not in persisted_tasks]
                extra = [task for task in persisted_tasks if task not in canonical_tasks]
                raise RunCollectionError(
                    "ARCHITECTURE_TASK_SCOPE_MISMATCH",
                    "teacher.architecture_config.task_names differs from canonical Animal56; "
                    f"missing={missing}, extra={extra}, order_matches={not missing and not extra}",
                )
    decoder_keys = [key for key in state if key.startswith(DECODER_PREFIX)]
    decoder_tasks = {
        key[len(DECODER_PREFIX) :].split(".", 1)[0]
        for key in decoder_keys
        if "." in key[len(DECODER_PREFIX) :]
    }
    if decoder_tasks != set(tasks):
        missing = sorted(set(tasks) - decoder_tasks)
        extra = sorted(decoder_tasks - set(tasks))
        raise RunCollectionError(
            "DECODER_SCOPE_MISMATCH",
            f"teacher decoder task set differs from ordered 56 tasks; missing={missing}, extra={extra}",
        )
    scalers = payload.get("task_scalers")
    if type(scalers) is not dict:
        raise RunCollectionError("MISSING_SCALERS", "teacher.task_scalers is not an object")
    if any(type(task) is not str for task in scalers):
        raise RunCollectionError(
            "SCALER_SCOPE_MISMATCH",
            "teacher scaler task keys must all be strings",
        )
    scaler_tasks = set(scalers)
    if scaler_tasks != set(canonical_tasks):
        missing = sorted(set(canonical_tasks) - scaler_tasks)
        extra = sorted(scaler_tasks - set(canonical_tasks))
        raise RunCollectionError(
            "SCALER_SCOPE_MISMATCH",
            f"teacher scaler missing/extra tasks differ from Animal56; missing={missing}, extra={extra}",
        )
    scaler_rows: list[dict[str, Any]] = []
    for task in tasks:
        scaler = scalers.get(task)
        if type(scaler) is not dict:
            raise RunCollectionError("MISSING_SCALERS", f"teacher scaler missing for {task}")
        mean = scaler.get("mean")
        std = scaler.get("std")
        if not _finite_number(mean) or not _finite_number(std):
            raise RunCollectionError("INVALID_SCALER", f"teacher scaler mean/std invalid for {task}")
        scaler_rows.append(
            {
                "task": task,
                "mean": float(mean),
                "std": float(std),
                "source": "teacher_checkpoint.task_scalers",
                "extra": json_safe({key: value for key, value in scaler.items() if key not in {"mean", "std"}}),
            }
        )
    backbone_keys = [key for key in state if key.startswith(BACKBONE_PREFIX)]
    encoder_other_keys = [
        key for key in state if key.startswith(ENCODER_PREFIX) and not key.startswith(BACKBONE_PREFIX)
    ]
    if not backbone_keys:
        raise RunCollectionError("MISSING_BACKBONE", "teacher contains no encoder.backbone tensors")
    return (
        {
            "asset_id": "teacher:" + digest,
            "kind": "teacher",
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "load": dict(load),
            "epoch": json_safe(payload.get("epoch")),
            "seed": json_safe((payload.get("reproducibility") or {}).get("base_seed") if isinstance(payload.get("reproducibility"), Mapping) else None),
            "task_names": tasks,
            "configuration": json_safe(payload.get("configuration")),
            "architecture_config": json_safe(payload.get("architecture_config")),
            "data_config": json_safe(payload.get("data_config")),
            "split_manifest_hash": json_safe(payload.get("split_manifest_hash")),
            "feature_schema_version": json_safe(payload.get("feature_schema_version")),
            "decoder_inventory": tensor_inventory(state, decoder_keys),
            "scalers": scaler_rows,
            "backbone": tensor_inventory(state, backbone_keys),
            "encoder_non_backbone": tensor_inventory(state, encoder_other_keys),
            "readout_keys": sorted(key for key in encoder_other_keys if "readout" in key.lower()),
            "state_role_note": "state_dict does not persist parameter-versus-buffer ownership; known buffer suffixes are marked and all others remain explicitly unresolved",
        },
        state,
    )


def resolve_teacher_checkpoint(
    sidecar: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Path, str, list[dict[str, Any]]]:
    declared_sha = _strict_sha256(
        sidecar.get("teacher_real_checkpoint_sha256"),
        "provenance.teacher_real_checkpoint_sha256",
    )
    attempts: list[dict[str, Any]] = []
    explicit = sidecar.get("teacher_real_checkpoint")
    if explicit is not None:
        path = resolve_reference_path(explicit, repo_root, "provenance.teacher_real_checkpoint")
        path = _require_asset_within_repository(path, repo_root, "teacher checkpoint")
        directory_reference = sidecar.get("teacher_real_run_dir")
        if directory_reference is not None:
            declared_directory = resolve_reference_path(
                directory_reference,
                repo_root,
                "provenance.teacher_real_run_dir",
            )
            declared_directory = _require_asset_within_repository(
                declared_directory,
                repo_root,
                "teacher run directory",
            )
            if path.parent != declared_directory:
                raise RunCollectionError(
                    "TEACHER_REFERENCE_CONFLICT",
                    "explicit teacher checkpoint is not inside the declared teacher run directory",
                    status="CONFLICT",
                )
        verification = verify_scoped_asset_path(
            path,
            declared_sha,
            repo_root,
            label="teacher checkpoint",
        )
        attempts.append({**verification, "source": "explicit_checkpoint"})
        return path, declared_sha, attempts

    directory_reference = sidecar.get("teacher_real_run_dir")
    if directory_reference is None:
        raise RunCollectionError(
            "MISSING_TEACHER_REFERENCE",
            "provenance has neither teacher_real_checkpoint nor teacher_real_run_dir",
        )
    directory = resolve_reference_path(
        directory_reference,
        repo_root,
        "provenance.teacher_real_run_dir",
    )
    directory = _require_asset_within_repository(
        directory,
        repo_root,
        "teacher run directory",
    )
    if not directory.is_dir():
        raise RunCollectionError("MISSING_TEACHER_DIRECTORY", f"teacher directory is missing: {directory}")
    matches: list[Path] = []
    for candidate in sorted(directory.iterdir(), key=lambda item: item.name):
        if not candidate.is_file() or candidate.suffix.lower() not in {".pt", ".pth", ".ckpt", ".bin"}:
            continue
        candidate = _require_asset_within_repository(
            candidate,
            repo_root,
            "teacher directory candidate",
        )
        digest = sha256_file(candidate)
        attempts.append({"path": str(candidate), "sha256": digest, "source": "referenced_directory"})
        if digest == declared_sha:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise RunCollectionError(
            "TEACHER_CANDIDATE_AMBIGUITY",
            f"declared teacher SHA matched {len(matches)} candidates in the referenced directory",
            status="CONFLICT",
        )
    return matches[0], declared_sha, attempts


def project_scope_evidence(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Project persisted training-scope fields and expose cross-source conflicts."""

    fields: dict[str, Any] = {}
    conflict_fields: list[str] = []
    missing_fields: list[str] = []

    def valid_value(field: str, value: Any) -> bool:
        if field == "task_names":
            return (
                type(value) is list
                and bool(value)
                and all(type(item) is str and bool(item) for item in value)
                and len(value) == len(set(value))
            )
        if field in {"split_seed", "max_nodes_filter"}:
            return type(value) is int and (field != "max_nodes_filter" or value > 0)
        if field == "train_fraction":
            return value is None or (
                type(value) in (int, float)
                and math.isfinite(float(value))
                and 0.0 < float(value) <= 1.0
            )
        if field == "fit_conformal":
            return type(value) is bool
        if field in {"datastore_fingerprint", "split_manifest_hash"}:
            return type(value) is str and bool(re.fullmatch(r"[0-9a-f]{64}", value))
        return type(value) is str and bool(value)

    for field, paths in SCOPE_FIELD_PATHS.items():
        candidates: list[dict[str, Any]] = []
        identities: dict[str, Any] = {}
        invalid_sources: list[str] = []
        for source_name, document in sources:
            for path in paths:
                value = _nested_present(document, path)
                if value is _MISSING:
                    continue
                safe_value = json_safe(value)
                identity = json.dumps(
                    safe_value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                candidates.append(
                    {
                        "source": source_name + "." + ".".join(path),
                        "value": safe_value,
                    }
                )
                if not valid_value(field, value):
                    invalid_sources.append(source_name + "." + ".".join(path))
                identities[identity] = safe_value
        if invalid_sources:
            fields[field] = {
                "status": "INVALID",
                "value": None,
                "reason": "one or more persisted values have the wrong JSON type or format",
                "sources": candidates,
                "invalid_sources": invalid_sources,
            }
            missing_fields.append(field)
        elif not candidates:
            fields[field] = {
                "status": "UNKNOWN",
                "value": None,
                "reason": "field was not persisted in any inspected source",
                "sources": [],
            }
            if field in REQUIRED_SCOPE_FIELDS:
                missing_fields.append(field)
        elif len(identities) != 1:
            fields[field] = {
                "status": "CONFLICT",
                "value": None,
                "reason": "persisted sources disagree",
                "sources": candidates,
            }
            conflict_fields.append(field)
        else:
            fields[field] = {
                "status": "CONSISTENT",
                "value": next(iter(identities.values())),
                "reason": None,
                "sources": candidates,
            }
    return {
        "status": "CONFLICT" if conflict_fields else ("INCOMPLETE" if missing_fields else "COMPLETE"),
        "fields": fields,
        "conflict_fields": conflict_fields,
        "missing_fields": missing_fields,
    }


def require_complete_scope(scope: Mapping[str, Any], label: str, *, source_teacher: bool = False) -> None:
    if scope.get("conflict_fields"):
        raise RunCollectionError(
            "TRAINING_SCOPE_CONFLICT",
            f"{label} training scope conflicts: {scope['conflict_fields']}",
            status="CONFLICT",
        )
    missing = list(scope.get("missing_fields", []))
    # Historical source runs predate label-scaling metadata. Absence is a
    # scientific unknown, never an inferred 100% training fraction.
    fraction = scope.get("fields", {}).get("train_fraction", {})
    if source_teacher and fraction.get("status") == "UNKNOWN" and not fraction.get("sources"):
        missing = [field for field in missing if field != "train_fraction"]
    if missing:
        raise RunCollectionError(
            "TRAINING_SCOPE_INCOMPLETE",
            f"{label} training scope fields are not persisted: {missing}",
        )


def _frozen_value_matches(expected: Any, actual: Any) -> bool:
    if type(expected) is bool:
        return type(actual) is bool and actual is expected
    if type(expected) in (int, float):
        return (
            type(actual) in (int, float)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
        )
    return type(actual) is type(expected) and actual == expected


def validate_run_identity_sources(
    spec: RunSpec,
    args_document: Mapping[str, Any],
    metadata: Mapping[str, Any],
    best_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def values_for(field: str, candidates: Sequence[tuple[str, Any]]) -> list[dict[str, Any]]:
        result = [
            {"source": source, "field": field, "value": json_safe(value)}
            for source, value in candidates
            if value is not None
        ]
        rows.extend(result)
        return result

    seed_rows = values_for(
        "seed",
        [
            ("args.seed", args_document.get("seed")),
            ("run_metadata.seed", metadata.get("seed")),
            (
                "run_metadata.configuration.seed",
                _nested(metadata, ("configuration", "seed")),
            ),
            (
                "best_checkpoint.configuration.seed",
                _nested(best_payload, ("configuration", "seed")),
            ),
        ],
    )
    for row in seed_rows:
        if type(row["value"]) is not int or row["value"] != spec.seed:
            raise RunCollectionError(
                "RUN_SEED_CONFLICT",
                f"{row['source']}={row['value']!r} differs from lock seed {spec.seed}",
                status="CONFLICT",
            )

    adaptation_sources = (
        ("args", args_document),
        ("run_metadata", metadata),
        ("best_checkpoint", best_payload),
    )
    for field, expected in spec.expected_adaptation.items():
        field_rows: list[dict[str, Any]] = []
        for source_name, document in adaptation_sources:
            source_rows: list[dict[str, Any]] = []
            for path in (
                (field,),
                ("configuration", field),
                ("architecture_config", field),
            ):
                value = _nested_present(document, path)
                if value is _MISSING:
                    continue
                row = {
                    "source": source_name + "." + ".".join(path),
                    "field": field,
                    "value": json_safe(value),
                    "expected": json_safe(expected),
                    "expected_source": (
                        "low_policy.expected_adaptation"
                        if spec.lock_source == "low_policy"
                        else "core_lock.config"
                    ),
                }
                rows.append(row)
                source_rows.append(row)
                field_rows.append(row)
                if not _frozen_value_matches(expected, value):
                    code = (
                        "RUN_METHOD_CONFLICT"
                        if field in {"d6_candidate", "d7_candidate"}
                        else "RUN_ADAPTATION_CONFLICT"
                    )
                    raise RunCollectionError(
                        code,
                        f"{row['source']}={value!r} differs from frozen {field}={expected!r}",
                        status="CONFLICT",
                    )
            if (
                not source_rows
                and field
                in {"d6_candidate", "d7_candidate", "freeze_backbone_epochs"}
            ):
                raise RunCollectionError(
                    "RUN_ADAPTATION_MISSING",
                    f"{source_name} does not persist frozen adaptation field {field}",
                )
        if not field_rows:
            raise RunCollectionError(
                "RUN_ADAPTATION_MISSING",
                f"no runtime source persists frozen adaptation field {field}",
            )

    fraction_rows = values_for(
        "train_fraction",
        [
            ("args.train_fraction", args_document.get("train_fraction")),
            ("run_metadata.train_fraction", metadata.get("train_fraction")),
            (
                "run_metadata.configuration.train_fraction",
                _nested(metadata, ("configuration", "train_fraction")),
            ),
            (
                "best_checkpoint.configuration.train_fraction",
                _nested(best_payload, ("configuration", "train_fraction")),
            ),
        ],
    )
    expected_fraction = spec.fraction / 100.0
    for row in fraction_rows:
        value = row["value"]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise RunCollectionError(
                "RUN_FRACTION_CONFLICT",
                f"{row['source']} has invalid train_fraction {value!r}",
                status="CONFLICT",
            )
        if not math.isclose(float(value), expected_fraction, rel_tol=0.0, abs_tol=1e-12):
            raise RunCollectionError(
                "RUN_FRACTION_CONFLICT",
                f"{row['source']}={value!r} differs from lock fraction {expected_fraction}",
                status="CONFLICT",
            )
    return rows


def _walk_string_fields(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                continue
            yield from _walk_string_fields(item, prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_string_fields(item, prefix + (str(index),))
    elif type(value) is str:
        yield prefix, value


def collect_training_references(
    documents: Sequence[tuple[str, Mapping[str, Any]]],
    repo_root: Path,
    evidence: EvidenceWriter,
    run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source_name, document in documents:
        for field_path, value in _walk_string_fields(document):
            key = field_path[-1].lower() if field_path else ""
            if "train" not in key or not any(marker in key for marker in TRAINING_REFERENCE_MARKERS):
                continue
            if not any(token in key for token in ("path", "file", "manifest", "list")):
                continue
            source_field = source_name + "." + ".".join(field_path)
            identity = (source_field, value)
            if identity in seen:
                continue
            seen.add(identity)
            reconstructed = "reconstruct" in key or "reconstruct" in source_field.lower()
            row: dict[str, Any] = {
                "source_field": source_field,
                "declared_path": value,
                "evidence_kind": "reconstructed" if reconstructed else "runtime_record",
            }
            try:
                path = resolve_reference_path(value, repo_root, source_field)
                row["resolved_path"] = str(path)
                if not path.is_file():
                    row.update(status="MISSING", reason="referenced training file is not persisted")
                elif path.suffix.lower() not in SAFE_EVIDENCE_SUFFIXES:
                    row.update(status="BLOCKED", reason="file type is outside the non-sensitive evidence allowlist")
                elif path.stat().st_size > 16 * 1024 * 1024:
                    row.update(status="BLOCKED", reason="evidence file exceeds the 16 MiB bounded-copy limit")
                else:
                    evidence_record = _collect_structured_training_evidence(
                        path,
                        evidence,
                        f"training/{run_id}/{len(rows):03d}_{path.name}",
                    )
                    row.update(
                        status=evidence_record["status"],
                        sha256=evidence_record.get("sha256"),
                        source_sha256=evidence_record.get(
                            "source_sha256",
                            evidence_record.get("sha256"),
                        ),
                        evidence_path=evidence_record.get("evidence_path"),
                        reason=evidence_record.get("reason"),
                        removed_fields=evidence_record.get("removed_fields"),
                    )
            except (OSError, S4ECollectionError) as exc:
                row.update(status="ERROR", reason=str(exc))
            rows.append(row)
    if not rows:
        rows.append(
            {
                "status": "NOT_PERSISTED",
                "declared_path": None,
                "evidence_kind": None,
                "reason": "no explicit runtime training sample/subset file reference was persisted",
            }
        )
    return rows


def _init_reference_sources(
    spec: RunSpec,
    args_document: Mapping[str, Any],
    metadata: Mapping[str, Any],
    best_payload: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, str]]:
    lock_document = {"init_state_path": spec.lock_init_state_path}
    return _reference_candidates(
        [
            ("lock", lock_document, [("init_state_path",)]),
            ("args", args_document, [("init_state_path",)]),
            (
                "run_metadata",
                metadata,
                [
                    ("init_state_path",),
                    ("configuration", "init_state_path"),
                    ("args", "init_state_path"),
                    ("init_overlay", "path"),
                    ("init_overlay", "output"),
                    ("init_overlay", "init_state_path"),
                    ("d7_artifact_contract", "output"),
                    ("d7_artifact_contract", "init_state_path"),
                    ("d6_artifact_contract", "output"),
                    ("d6_artifact_contract", "init_state_path"),
                ],
            ),
            ("best_checkpoint", best_payload, [("configuration", "init_state_path")]),
        ],
        repo_root,
    )


def _sidecar_reference_sources(
    init_path: Path,
    metadata: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, str]]:
    explicit = _reference_candidates(
        [
            (
                "run_metadata",
                metadata,
                [
                    ("init_provenance_path",),
                    ("init_overlay", "provenance"),
                    ("init_overlay", "provenance_path"),
                    ("d7_artifact_contract", "provenance_path"),
                    ("d6_artifact_contract", "provenance_path"),
                ],
            )
        ],
        repo_root,
    )
    default = Path(str(init_path) + ".provenance.json").resolve(strict=False)
    if default.is_file() or not explicit:
        explicit.append(
            {
                "source": "derived_from_exact_init_path",
                "declared": str(default),
                "resolved": str(default),
            }
        )
    return explicit


def _source_sha_candidates(
    metadata: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    args_document: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sources: list[tuple[str, Mapping[str, Any], Sequence[Sequence[str]]]] = [
        (
            "run_metadata",
            metadata,
            [
                ("init_overlay", "output_sha256"),
                ("init_overlay", "artifact_sha256"),
                ("init_overlay", "init_state_sha256"),
                ("d7_artifact_contract", "output_sha256"),
                ("d7_artifact_contract", "artifact_sha256"),
                ("d6_artifact_contract", "output_sha256"),
                ("d6_artifact_contract", "artifact_sha256"),
            ],
        ),
        ("provenance", sidecar, [("output_sha256",)]),
    ]
    if args_document is not None:
        sources.insert(
            0,
            (
                "args",
                args_document,
                [
                    ("init_overlay", "output_sha256"),
                    ("init_overlay", "artifact_sha256"),
                    ("init_overlay", "init_state_sha256"),
                    ("init_overlay_provenance", "init_state_sha256"),
                    ("d7_artifact_contract", "output_sha256"),
                    ("d7_artifact_contract", "artifact_sha256"),
                    ("d6_artifact_contract", "output_sha256"),
                    ("d6_artifact_contract", "artifact_sha256"),
                ],
            ),
        )
    for source_name, document, paths in sources:
        for path in paths:
            value = _nested_present(document, path)
            if value is _MISSING:
                continue
            rows.append(
                {
                    "source": source_name + "." + ".".join(path),
                    "sha256": _strict_sha256(value, source_name + "." + ".".join(path)),
                }
            )
    return rows


def resolve_init_sha_identity(
    metadata: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    args_document: Mapping[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    candidates = _source_sha_candidates(metadata, sidecar, args_document)
    if not candidates:
        raise RunCollectionError("MISSING_INIT_SHA", "no init output SHA was persisted")
    identities = {row["sha256"] for row in candidates}
    if len(identities) != 1:
        raise RunCollectionError(
            "INIT_SHA_CONFLICT",
            f"init SHA references disagree: {candidates}",
            status="CONFLICT",
        )
    return next(iter(identities)), candidates


def validate_init_contract_identity(
    spec: RunSpec,
    args_document: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sidecar_human_seed = _nested_present(sidecar, ("human_seed",))
    sidecar_teacher_epoch = _nested_present(sidecar, ("expected_teacher_epoch",))
    if type(sidecar_human_seed) is not int or sidecar_human_seed != spec.seed:
        raise RunCollectionError(
            "INIT_CONTRACT_IDENTITY_CONFLICT",
            f"provenance.human_seed={sidecar_human_seed!r} differs from run seed {spec.seed}",
            status="CONFLICT",
        )
    if type(sidecar_teacher_epoch) is not int:
        raise RunCollectionError(
            "INIT_CONTRACT_IDENTITY_MISSING",
            "provenance.expected_teacher_epoch is not an integer",
        )
    rows: list[dict[str, Any]] = []
    contracts_found = 0
    for source_name, document in (
        ("args", args_document),
        ("run_metadata", metadata),
    ):
        for contract_name in ("d6_artifact_contract", "d7_artifact_contract"):
            contract = _nested_present(document, (contract_name,))
            if contract is _MISSING or contract is None:
                continue
            if type(contract) is not dict:
                raise RunCollectionError(
                    "INVALID_INIT_CONTRACT",
                    f"{source_name}.{contract_name} must be an object when persisted",
                )
            contracts_found += 1
            for field, expected in (
                ("human_seed", spec.seed),
                ("expected_teacher_epoch", sidecar_teacher_epoch),
            ):
                value = _nested_present(contract, (field,))
                qualified = f"{source_name}.{contract_name}.{field}"
                if value is _MISSING:
                    raise RunCollectionError(
                        "INIT_CONTRACT_IDENTITY_MISSING",
                        f"{qualified} is not persisted",
                    )
                row = {
                    "source": qualified,
                    "value": json_safe(value),
                    "expected": expected,
                }
                rows.append(row)
                if type(value) is not int or value != expected:
                    raise RunCollectionError(
                        "INIT_CONTRACT_IDENTITY_CONFLICT",
                        f"{qualified}={value!r} differs from expected {expected}",
                        status="CONFLICT",
                    )
            contract_seed = contract["human_seed"]
            if contract_seed != sidecar_human_seed:
                raise RunCollectionError(
                    "INIT_CONTRACT_IDENTITY_CONFLICT",
                    f"{source_name}.{contract_name}.human_seed differs from provenance.human_seed",
                    status="CONFLICT",
                )
    if contracts_found == 0:
        raise RunCollectionError(
            "INIT_CONTRACT_IDENTITY_MISSING",
            "no d6/d7 artifact contract was persisted in args or run_metadata",
        )
    return rows


def validate_best_references(
    spec: RunSpec,
    metadata: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (
        ("best_checkpoint_path",),
        ("best_checkpoint", "path"),
        ("checkpoint_manifest", "best_checkpoint_path"),
    ):
        value = _nested(metadata, path)
        if value is None:
            continue
        resolved = resolve_reference_path(
            value,
            repo_root,
            "run_metadata." + ".".join(path),
        )
        row = {
            "source": "run_metadata." + ".".join(path),
            "kind": "path",
            "value": str(resolved),
        }
        rows.append(row)
        locked = resolve_reference_path(spec.checkpoint_path, repo_root, "lock.checkpoint.path")
        if resolved != locked:
            raise RunCollectionError(
                "BEST_REFERENCE_CONFLICT",
                f"{row['source']} differs from the locked best checkpoint path",
                status="CONFLICT",
            )
    for path in (
        ("best_checkpoint_sha256",),
        ("best_checkpoint", "sha256"),
        ("checkpoint_manifest", "best_checkpoint_sha256"),
    ):
        value = _nested(metadata, path)
        if value is None:
            continue
        digest = _strict_sha256(value, "run_metadata." + ".".join(path))
        row = {
            "source": "run_metadata." + ".".join(path),
            "kind": "sha256",
            "value": digest,
        }
        rows.append(row)
        if digest != spec.checkpoint_sha256:
            raise RunCollectionError(
                "BEST_REFERENCE_CONFLICT",
                f"{row['source']} differs from the locked best checkpoint SHA-256",
                status="CONFLICT",
            )
    return rows


def validate_teacher_reference_sources(
    metadata: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sha_sources = [
        ("provenance.teacher_real_checkpoint_sha256", sidecar.get("teacher_real_checkpoint_sha256")),
        (
            "run_metadata.init_overlay.teacher_real_checkpoint_sha256",
            _nested(metadata, ("init_overlay", "teacher_real_checkpoint_sha256")),
        ),
        (
            "run_metadata.d7_artifact_contract.teacher_real_checkpoint_sha256",
            _nested(metadata, ("d7_artifact_contract", "teacher_real_checkpoint_sha256")),
        ),
        (
            "run_metadata.d6_artifact_contract.teacher_real_checkpoint_sha256",
            _nested(metadata, ("d6_artifact_contract", "teacher_real_checkpoint_sha256")),
        ),
    ]
    sha_rows = [
        {"source": source, "kind": "sha256", "value": _strict_sha256(value, source)}
        for source, value in sha_sources
        if value is not None
    ]
    if not sha_rows:
        raise RunCollectionError("MISSING_TEACHER_SHA", "teacher SHA-256 is not persisted")
    if len({row["value"] for row in sha_rows}) != 1:
        raise RunCollectionError(
            "TEACHER_REFERENCE_CONFLICT",
            "teacher SHA-256 sources disagree",
            status="CONFLICT",
        )
    rows.extend(sha_rows)

    for field_name, paths in (
        (
            "checkpoint",
            (
                ("teacher_real_checkpoint",),
                ("init_overlay", "teacher_real_checkpoint"),
                ("d7_artifact_contract", "teacher_real_checkpoint"),
                ("d6_artifact_contract", "teacher_real_checkpoint"),
            ),
        ),
        (
            "run_dir",
            (
                ("teacher_real_run_dir",),
                ("init_overlay", "teacher_real_run_dir"),
                ("d7_artifact_contract", "teacher_real_run_dir"),
                ("d6_artifact_contract", "teacher_real_run_dir"),
            ),
        ),
    ):
        path_rows: list[dict[str, Any]] = []
        for source_name, document in (("provenance", sidecar), ("run_metadata", metadata)):
            for path in paths:
                value = _nested(document, path)
                if value is None:
                    continue
                path_rows.append(
                    {
                        "source": source_name + "." + ".".join(path),
                        "kind": field_name,
                        "value": str(
                            resolve_reference_path(
                                value,
                                repo_root,
                                source_name + "." + ".".join(path),
                            )
                        ),
                    }
                )
        if len({row["value"] for row in path_rows}) > 1:
            raise RunCollectionError(
                "TEACHER_REFERENCE_CONFLICT",
                f"teacher {field_name} sources disagree",
                status="CONFLICT",
            )
        rows.extend(path_rows)
    return rows


def _repository_identity(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "head": head,
        "worktree_status": status,
        "python": sys.executable,
        "platform": sys.platform,
    }


def _write_checksum_manifest(output: Path) -> None:
    rows: list[str] = []
    for path in sorted(output.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    with (output / "checksums.sha256").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")


def verify_checksum_manifest(output: str | Path) -> None:
    root = Path(output)
    checksum_path = root / "checksums.sha256"
    rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2, "invalid checksums.sha256 line")
        digest, relative = parts
        _strict_sha256(digest, f"checksums[{relative}]")
        require(relative != "checksums.sha256", "checksums.sha256 must exclude itself")
        require(relative not in rows, f"duplicate checksum row: {relative}")
        rows[relative] = digest
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    require(set(rows) == actual_files, "checksums.sha256 coverage differs from output files")
    for relative, digest in rows.items():
        require(sha256_file(root / PurePosixPath(relative)) == digest, f"checksum mismatch: {relative}")


def verify_output(output: str | Path, expected_run_ids: set[str]) -> dict[str, Any]:
    """Independently re-read outputs; never trusts a manifest PASS flag."""

    root = Path(output)
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": json_safe(detail)})

    links, _, _ = read_json_document(root / "run_source_links.json")
    assets, _, _ = read_json_document(root / "source_assets.json")
    training, _, _ = read_json_document(root / "training_scope_evidence.json")
    require(type(links) is dict and type(links.get("runs")) is list, "invalid run_source_links output")
    require(type(assets) is dict and type(assets.get("assets")) is list, "invalid source_assets output")
    require(type(training) is dict and type(training.get("targets")) is list, "invalid training_scope output")
    rows = links["runs"]
    row_ids = [row.get("run_id") for row in rows]
    check("exact_50_run_rows", len(rows) == 50, len(rows))
    check("exact_expected_run_ids", set(row_ids) == expected_run_ids and len(row_ids) == len(set(row_ids)), row_ids)
    check("b0_excluded", all(row.get("method") in METHODS for row in rows), sorted({row.get("method") for row in rows}))
    check(
        "matrix_complete",
        {(row.get("method"), row.get("seed"), row.get("fraction")) for row in rows}
        == {
            (method, seed, fraction)
            for method in METHODS
            for seed in SEEDS
            for fraction in (*LOW_FRACTIONS, 100)
        },
        "B1/RPT x seeds42-46 x fractions10/25/50/75/100",
    )
    asset_ids = [asset.get("asset_id") for asset in assets["assets"]]
    check("unique_source_assets", len(asset_ids) == len(set(asset_ids)), asset_ids)
    referenced = {
        value
        for row in rows
        for value in (row.get("init_asset_id"), row.get("teacher_asset_id"))
        if value is not None
    }
    check("source_references_resolve", referenced <= set(asset_ids), sorted(referenced - set(asset_ids)))
    target_ids = [row.get("run_id") for row in training["targets"]]
    check("training_scope_covers_runs", set(target_ids) == expected_run_ids and len(target_ids) == 50, target_ids)
    complete = all(row.get("status") == "COLLECTED" for row in rows)
    check("all_runs_collected", complete, {row["run_id"]: row.get("status") for row in rows if row.get("status") != "COLLECTED"})
    all_checks = all(row["passed"] for row in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "all_structural_checks_pass": all_checks,
        "collection_complete": complete,
        "acceptance_status": "PENDING_REVIEW",
        "verified_at_unix": time.time(),
    }


def collect_source_assets(
    repo_root: str | Path,
    low_policy_path: str | Path,
    core_lock_path: str | Path,
    output_path: str | Path,
) -> int:
    repo_root = Path(repo_root).resolve()
    low_policy_path = Path(low_policy_path).resolve()
    core_lock_path = Path(core_lock_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise S4ECollectionError(f"output already exists; refusing to overwrite: {output}")

    low_policy, low_bytes, low_sha = read_json_document(low_policy_path)
    core_lock, core_bytes, core_sha = read_json_document(core_lock_path)
    specs = build_run_specs(low_policy, core_lock)
    repository = _repository_identity(repo_root)
    started = time.time()
    output.mkdir(parents=True, exist_ok=False)
    evidence = EvidenceWriter(output / "evidence")
    evidence.copy(
        low_policy_path,
        "inputs/low_policy.json",
        parsed=low_policy,
        category="input_lock",
    )
    evidence.copy(
        core_lock_path,
        "inputs/core_lock.json",
        parsed=core_lock,
        category="input_lock",
    )
    require(sha256_bytes(low_bytes) == low_sha, "low policy input changed in memory")
    require(sha256_bytes(core_bytes) == core_sha, "core lock input changed in memory")

    links: list[dict[str, Any]] = []
    source_asset_records: dict[str, dict[str, Any]] = {}
    source_states: dict[str, Mapping[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    source_training: dict[str, dict[str, Any]] = {}
    target_training: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for spec in specs:
        link: dict[str, Any] = {
            "run_id": spec.run_id,
            "method": spec.method,
            "seed": spec.seed,
            "fraction": spec.fraction,
            "lock_source": spec.lock_source,
            "expected_adaptation": json_safe(spec.expected_adaptation),
            "status": "PENDING",
        }
        try:
            run_dir = resolve_reference_path(spec.run_dir, repo_root, f"{spec.run_id}.run_dir")
            checkpoint = resolve_reference_path(
                spec.checkpoint_path,
                repo_root,
                f"{spec.run_id}.checkpoint",
            )
            best_payload, best_load = load_verified_torch(
                checkpoint,
                spec.checkpoint_sha256,
                repo_root,
                expected_size_bytes=spec.checkpoint_size_bytes,
                label=f"{spec.run_id} best checkpoint",
            )
            if not isinstance(best_payload, Mapping):
                raise RunCollectionError("INVALID_BEST", "best checkpoint root is not a mapping")
            observed_epoch = best_payload.get("epoch")
            if type(observed_epoch) is not int or observed_epoch != spec.best_epoch:
                raise RunCollectionError(
                    "BEST_EPOCH_MISMATCH",
                    f"best checkpoint epoch {observed_epoch!r} != lock {spec.best_epoch}",
                )
            link["best"] = {"path": str(checkpoint), "sha256": spec.checkpoint_sha256,
                            "size_bytes": spec.checkpoint_size_bytes, "epoch": observed_epoch, "load": best_load}
            args_document, args_evidence = read_json_evidence(
                run_dir / "args.json",
                evidence,
                f"runs/{spec.run_id}/args.json",
                category="run_metadata",
            )
            metadata, metadata_evidence = read_json_evidence(
                run_dir / "run_metadata.json",
                evidence,
                f"runs/{spec.run_id}/run_metadata.json",
                category="run_metadata",
            )
            best_reference_sources = validate_best_references(
                spec,
                metadata,
                repo_root,
            )
            field_sources = validate_run_identity_sources(
                spec,
                args_document,
                metadata,
                best_payload,
            )
            init_candidates = _init_reference_sources(
                spec,
                args_document,
                metadata,
                best_payload,
                repo_root,
            )
            init_path = require_single_reference(init_candidates, "init")
            init_path = _require_asset_within_repository(
                init_path,
                repo_root,
                f"{spec.run_id} init",
            )
            sidecar_candidates = _sidecar_reference_sources(init_path, metadata, repo_root)
            sidecar_path = require_single_reference(sidecar_candidates, "init provenance sidecar")
            sidecar_path = _require_asset_within_repository(
                sidecar_path,
                repo_root,
                f"{spec.run_id} init provenance sidecar",
            )
            sidecar, sidecar_evidence = read_json_evidence(
                sidecar_path,
                evidence,
                f"runs/{spec.run_id}/init.provenance.json",
                category="init_provenance",
            )
            sidecar_output = sidecar.get("output")
            if sidecar_output is None:
                raise RunCollectionError(
                    "MISSING_INIT_OUTPUT_REFERENCE",
                    "init provenance lacks the exact output path",
                )
            resolved_sidecar_output = resolve_reference_path(
                sidecar_output,
                repo_root,
                "provenance.output",
            )
            if resolved_sidecar_output != init_path:
                raise RunCollectionError(
                    "INIT_OUTPUT_REFERENCE_CONFLICT",
                    "init provenance output path differs from the run's resolved init",
                    status="CONFLICT",
                )
            teacher_reference_sources = validate_teacher_reference_sources(
                metadata,
                sidecar,
                repo_root,
            )
            expected_teacher_epoch = sidecar.get("expected_teacher_epoch")
            if type(expected_teacher_epoch) is not int:
                raise RunCollectionError(
                    "INVALID_TEACHER_EPOCH",
                    "provenance.expected_teacher_epoch must be an integer",
                )
            human_seed = sidecar.get("human_seed")
            if type(human_seed) is not int or human_seed != spec.seed:
                raise RunCollectionError(
                    "INIT_SEED_MISMATCH",
                    f"provenance human_seed {human_seed!r} != run seed {spec.seed}",
                )
            init_contract_identity_sources = validate_init_contract_identity(
                spec,
                args_document,
                metadata,
                sidecar,
            )
            init_sha, init_sha_candidates = resolve_init_sha_identity(
                metadata,
                sidecar,
                args_document,
            )
            init_asset_id = "init:" + init_sha
            cached_init = source_asset_records.get(init_asset_id)
            init_path_verification = verify_scoped_asset_path(
                init_path,
                init_sha,
                repo_root,
                expected_size_bytes=(
                    cached_init.get("size_bytes")
                    if cached_init is not None
                    else None
                ),
                label=f"{spec.run_id} init",
                mismatch_code="INIT_SHA_CONFLICT",
                mismatch_status="CONFLICT",
            )
            init_path_verification["summary_reused"] = cached_init is not None
            link.update(init_asset_id=init_asset_id, init_path_verification=init_path_verification,
                        sidecar_evidence=sidecar_evidence)
            if init_asset_id not in source_asset_records:
                init_payload, init_load = load_verified_torch(
                    init_path,
                    init_sha,
                    repo_root,
                    label=f"{spec.run_id} init",
                )
                init_record, init_state = summarize_init_asset(
                    init_path,
                    init_sha,
                    init_payload,
                    init_load,
                )
                source_asset_records[init_asset_id] = init_record
                source_states[init_asset_id] = init_state

            teacher_path, teacher_sha, teacher_attempts = resolve_teacher_checkpoint(
                sidecar,
                repo_root,
            )
            teacher_asset_id = "teacher:" + teacher_sha
            cached_teacher = source_asset_records.get(teacher_asset_id)
            teacher_path_verification = verify_scoped_asset_path(
                teacher_path,
                teacher_sha,
                repo_root,
                expected_size_bytes=(
                    cached_teacher.get("size_bytes")
                    if cached_teacher is not None
                    else None
                ),
                label=f"{spec.run_id} teacher",
                mismatch_code="TEACHER_SHA_CONFLICT",
                mismatch_status="CONFLICT",
            )
            teacher_path_verification["summary_reused"] = cached_teacher is not None
            link["teacher_path_verification"] = teacher_path_verification
            if teacher_asset_id not in source_asset_records:
                teacher_payload, teacher_load = load_verified_torch(
                    teacher_path,
                    teacher_sha,
                    repo_root,
                    label=f"{spec.run_id} teacher",
                )
                teacher_record, teacher_state = summarize_teacher_asset(
                    teacher_path,
                    teacher_sha,
                    teacher_payload,
                    teacher_load,
                )
                if type(teacher_payload.get("epoch")) is not int or teacher_payload.get("epoch") != expected_teacher_epoch:
                    raise RunCollectionError(
                        "TEACHER_EPOCH_MISMATCH",
                        f"teacher epoch {teacher_payload.get('epoch')!r} != provenance {expected_teacher_epoch}",
                    )
                teacher_seed = (teacher_payload.get("reproducibility") or {}).get("base_seed") if isinstance(teacher_payload.get("reproducibility"), Mapping) else None
                if type(teacher_seed) is not int or teacher_seed != spec.seed:
                    raise RunCollectionError(
                        "TEACHER_SEED_MISMATCH",
                        f"teacher base seed {teacher_seed!r} != run seed {spec.seed}",
                    )
                teacher_record["expected_epoch_source"] = "init_provenance.expected_teacher_epoch"
                teacher_directory_reference = sidecar.get("teacher_real_run_dir")
                teacher_directory = (
                    resolve_reference_path(
                        teacher_directory_reference,
                        repo_root,
                        "provenance.teacher_real_run_dir",
                    )
                    if teacher_directory_reference is not None
                    else teacher_path.parent
                )
                teacher_documents: list[tuple[str, Mapping[str, Any]]] = [
                    ("teacher_provenance", sidecar)
                ]
                teacher_metadata_evidence: list[dict[str, Any]] = []
                for metadata_name in ("args.json", "run_metadata.json"):
                    metadata_path = teacher_directory / metadata_name
                    if metadata_path.is_file():
                        teacher_document, teacher_evidence = read_json_evidence(
                            metadata_path,
                            evidence,
                            f"sources/{teacher_sha}/{metadata_name}",
                            category="teacher_metadata",
                        )
                        teacher_documents.append(
                            ("teacher_" + metadata_name.removesuffix(".json"), teacher_document)
                        )
                        teacher_metadata_evidence.append(teacher_evidence)
                    else:
                        teacher_metadata_evidence.append(
                            {
                                "status": "NOT_PERSISTED",
                                "source_path": str(metadata_path),
                                "reason": "teacher metadata file was not persisted",
                            }
                        )
                teacher_record["metadata_evidence"] = teacher_metadata_evidence
                teacher_scope = project_scope_evidence(
                    [("teacher_checkpoint", teacher_payload), *teacher_documents]
                )
                source_asset_records[teacher_asset_id] = teacher_record
                source_states[teacher_asset_id] = teacher_state
                source_training[teacher_asset_id] = {
                    "asset_id": teacher_asset_id,
                    "kind": "source_teacher",
                    "scope": teacher_scope,
                    "sample_or_subset_evidence": collect_training_references(
                        teacher_documents,
                        repo_root,
                        evidence,
                        spec.run_id + "_teacher",
                    ),
                }

            link["teacher_asset_id"] = teacher_asset_id
            require_complete_scope(source_training[teacher_asset_id]["scope"], "teacher", source_teacher=True)
            cached_teacher = source_asset_records[teacher_asset_id]
            if cached_teacher.get("epoch") != expected_teacher_epoch:
                raise RunCollectionError(
                    "TEACHER_EPOCH_MISMATCH",
                    f"deduplicated teacher epoch {cached_teacher.get('epoch')!r} != provenance {expected_teacher_epoch}",
                )
            if cached_teacher.get("seed") != spec.seed:
                raise RunCollectionError(
                    "TEACHER_SEED_MISMATCH",
                    f"deduplicated teacher seed {cached_teacher.get('seed')!r} != run seed {spec.seed}",
                )

            comparison_id = init_asset_id + "|" + teacher_asset_id
            if comparison_id not in comparisons:
                backbone = compare_tensor_states(
                    source_states[teacher_asset_id],
                    source_states[init_asset_id],
                    prefix=BACKBONE_PREFIX,
                )
                encoder_non_backbone = compare_tensor_states(
                    source_states[teacher_asset_id],
                    source_states[init_asset_id],
                    prefix=ENCODER_PREFIX,
                )
                comparisons[comparison_id] = {
                    "comparison_id": comparison_id,
                    "teacher_asset_id": teacher_asset_id,
                    "init_asset_id": init_asset_id,
                    "backbone": backbone,
                    "encoder_all": encoder_non_backbone,
                    "status": "MATCH" if backbone["status"] == "MATCH" else "CONFLICT",
                }
            if comparisons[comparison_id]["status"] != "MATCH":
                raise RunCollectionError(
                    "BACKBONE_CONFLICT",
                    "teacher and init backbone differ by key, shape, dtype, or value",
                    status="CONFLICT",
                )

            training_references = collect_training_references(
                [
                    ("args", args_document),
                    ("run_metadata", metadata),
                    ("best_checkpoint", best_payload),
                ],
                repo_root,
                evidence,
                spec.run_id,
            )
            target_scope = project_scope_evidence(
                [
                    ("best_checkpoint", best_payload),
                    ("args", args_document),
                    ("run_metadata", metadata),
                ]
            )
            target_training.append(
                {
                    "run_id": spec.run_id,
                    "method": spec.method,
                    "seed": spec.seed,
                    "fraction": spec.fraction,
                    "scope": target_scope,
                    "sample_or_subset_evidence": training_references,
                }
            )
            require_complete_scope(target_scope, "target run")
            link.update(
                status="COLLECTED",
                run_dir=str(run_dir),
                best={
                    "path": str(checkpoint),
                    "sha256": spec.checkpoint_sha256,
                    "size_bytes": spec.checkpoint_size_bytes,
                    "epoch": observed_epoch,
                    "load": best_load,
                },
                args_evidence=args_evidence,
                run_metadata_evidence=metadata_evidence,
                best_reference_sources=best_reference_sources,
                field_sources=field_sources,
                init_references=init_candidates,
                init_sha_references=init_sha_candidates,
                init_contract_identity_sources=init_contract_identity_sources,
                init_path_verification=init_path_verification,
                init_asset_id=init_asset_id,
                sidecar_references=sidecar_candidates,
                sidecar_evidence=sidecar_evidence,
                teacher_asset_id=teacher_asset_id,
                teacher_reference_sources=teacher_reference_sources,
                teacher_resolution_attempts=teacher_attempts,
                teacher_path_verification=teacher_path_verification,
                comparison_id=comparison_id,
            )
        except (OSError, S4ECollectionError) as exc:
            if isinstance(exc, RunCollectionError):
                code = exc.code
                reason = exc.reason
                status = exc.status
            else:
                code = type(exc).__name__
                reason = str(exc)
                status = "ERROR"
            link.update(status=status, reason_code=code, reason=reason)
            issues.append(
                {
                    "run_id": spec.run_id,
                    "code": code,
                    "status": status,
                    "reason": reason,
                }
            )
            if not any(row.get("run_id") == spec.run_id for row in target_training):
                target_training.append(
                    {
                        "run_id": spec.run_id,
                        "method": spec.method,
                        "seed": spec.seed,
                        "fraction": spec.fraction,
                        "scope": None,
                        "sample_or_subset_evidence": [
                            {
                                "status": "UNKNOWN",
                                "declared_path": None,
                                "reason": "run source chain did not validate far enough to project training scope",
                            }
                        ],
                    }
                )
        links.append(link)

    run_links_document = {
        "schema_version": SCHEMA_VERSION,
        "runs": links,
        "issues": issues,
        "acceptance_status": "PENDING_REVIEW",
    }
    source_assets_document = {
        "schema_version": SCHEMA_VERSION,
        "assets": sorted(source_asset_records.values(), key=lambda row: row["asset_id"]),
        "comparisons": sorted(comparisons.values(), key=lambda row: row["comparison_id"]),
        "deduplication": "asset kind plus exact SHA-256; all fifty run links are retained",
        "acceptance_status": "PENDING_REVIEW",
    }
    scientific_unknowns = [
        {"asset_id": asset_id, "field": "train_fraction", "value": None,
         "reason": "not_persisted_in_historical_source",
         "affected_run_ids": [r["run_id"] for r in links if r.get("teacher_asset_id") == asset_id]}
        for asset_id, entry in sorted(source_training.items())
        if entry["scope"]["fields"]["train_fraction"]["status"] == "UNKNOWN"
    ]
    training_document = {
        "schema_version": SCHEMA_VERSION,
        "sources": sorted(source_training.values(), key=lambda row: row["asset_id"]),
        "scientific_unknowns": scientific_unknowns,
        "targets": sorted(target_training, key=lambda row: row["run_id"]),
        "labels_exported": False,
        "missing_runtime_sample_lists_are_not_inferred": True,
        "acceptance_status": "PENDING_REVIEW",
    }
    write_json(output / "run_source_links.json", run_links_document)
    write_json(output / "source_assets.json", source_assets_document)
    write_json(output / "training_scope_evidence.json", training_document)

    expected_run_ids = {spec.run_id for spec in specs}
    verification = verify_output(output, expected_run_ids)
    write_json(output / "verification.json", verification)
    complete = verification["all_structural_checks_pass"] and verification["collection_complete"] and not issues
    finished = time.time()
    existing_files = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "inputs": {
            "low_policy": {"path": str(low_policy_path), "sha256": low_sha},
            "core_lock": {"path": str(core_lock_path), "sha256": core_sha},
        },
        "run_mapping": [
            {
                "run_id": row.run_id,
                "method": row.method,
                "seed": row.seed,
                "fraction": row.fraction,
                "expected_adaptation": json_safe(row.expected_adaptation),
            }
            for row in specs
        ],
        "files": existing_files
        + [
            {
                "path": "collection_manifest.json",
                "sha256": None,
                "size_bytes": None,
                "reason": "self-referential manifest identity is covered by checksums.sha256",
            },
            {
                "path": "checksums.sha256",
                "sha256": None,
                "size_bytes": None,
                "reason": "checksum manifest excludes itself by contract",
            }
        ],
        "started_at_unix": started,
        "finished_at_unix": finished,
        "duration_seconds": finished - started,
        "execution_status": "COMPLETE" if complete else "COMPLETE_WITH_ISSUES",
        "collection_status": "COMPLETE" if complete else "INCOMPLETE",
        "self_check_status": "PASS" if verification["all_structural_checks_pass"] else "FAIL",
        "acceptance_status": "PENDING_REVIEW",
        "predictions_accessed": False,
        "model_constructed": False,
        "forward_called": False,
        "optimizer_constructed": False,
        "training_called": False,
        "gpu_used": False,
        "issue_count": len(issues),
        "scientific_unknowns": scientific_unknowns,
    }
    write_json(output / "collection_manifest.json", manifest)
    _write_checksum_manifest(output)
    verify_checksum_manifest(output)

    if sha256_file(low_policy_path) != low_sha or sha256_file(core_lock_path) != core_sha:
        raise S4ECollectionError("an input lock changed during collection")
    return 0 if complete else 2


__all__ = [
    "ADAPTATION_FIELDS",
    "BACKBONE_PREFIX",
    "LOW_FRACTIONS",
    "METHODS",
    "RunCollectionError",
    "RunSpec",
    "S4ECollectionError",
    "build_run_specs",
    "collect_source_assets",
    "compare_tensor_states",
    "contains_secret_key",
    "json_safe",
    "load_verified_torch",
    "read_json_document",
    "require_single_reference",
    "resolve_init_sha_identity",
    "resolve_reference_path",
    "resolve_teacher_checkpoint",
    "sha256_file",
    "strict_json_loads",
    "summarize_teacher_asset",
    "tensor_record",
    "validate_init_contract_identity",
    "validate_run_identity_sources",
    "verify_output",
    "verify_checksum_manifest",
    "verify_scoped_asset_path",
]
