"""S4B controlled low-fraction validation/test export entry point.

The public workflow is intentionally four-stage and fail-closed:

``preflight --side wsl|server`` -> ``run --split validation|test`` ->
``finalize --core-zip ...`` -> ``verify --archive ...``.

Formal server work additionally requires a separate Codex authorization file;
this repository does not contain or generate one.  The module can be imported
for synthetic tests without accessing checkpoints, a DataStore, CUDA, or any
real test/calibration records.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


# Absolute-script invocation from any cwd must import repository code before
# torch.load or any project import that may be needed to unpickle old payloads.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from s4_low_fraction_control import (  # noqa: E402
    ALIAS_METRICS_SHA256,
    ALIAS_PREDICTION_SHA256,
    ALLOWED_OPERATION,
    CORE_POLICY_SHA256,
    FROZEN_TEST_SAMPLE_PROJECTION_SHA256,
    HUMAN3_TASKS,
    MODEL_STATE_SHA_ALGORITHM,
    RUNTIME_OVERRIDE_NAMES,
    S3C3_CANONICAL_SHA256,
    S3C3_OUTER_SHA256,
    S4BControlError,
    S4B_POLICY_SHA256,
    S4B_RUN_IDS,
    S4B_TASK_ID,
    build_unified_50_metrics,
    build_validation_gate,
    compare_validation_reference,
    identity_sha256,
    load_authorization,
    load_authorization_document,
    load_s4b_policy,
    parse_jsonl,
    prepare_low_fraction_checkpoint,
    read_json,
    read_json_document,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
    validate_authorization,
    validate_datastore_metadata,
    validate_datastore_task_selection,
    validate_frozen_test_table,
    validate_legacy_alias_rows,
    validate_prediction_rows,
    validate_sample_table,
    validate_stored_metrics,
    validate_restoration_evidence,
    verify_037_aliases,
    verify_analysis_manifest,
    verify_validation_gate,
    verify_wsl_receipt,
    tensor_state_sha256,
)


TASK_ID = S4B_TASK_ID
DEFAULT_POLICY = REPOSITORY_ROOT / "configs/s4b_low_fraction_policy.json"
DEFAULT_ANALYSIS_MANIFEST = (
    REPOSITORY_ROOT
    / "Plans&Results/Results/2026-09-14_S3D_Figure2统计与图稿/analysis_manifest.json"
)
def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_exact_json_snapshot(
    path: str | Path,
    source_bytes: bytes,
    expected_sha256: str,
    *,
    label: str,
) -> Any:
    """Create a byte-exact snapshot and independently parse/hash the result."""

    if sha256_bytes(source_bytes) != expected_sha256:
        raise S4BControlError(f"{label} source bytes changed before snapshot")
    snapshot_path = Path(path)
    with snapshot_path.open("xb") as handle:
        handle.write(source_bytes)
    document, written_bytes, written_sha256 = read_json_document(snapshot_path)
    if written_bytes != source_bytes or written_sha256 != expected_sha256:
        raise S4BControlError(f"{label} snapshot bytes differ after write")
    return document


def _require_source_bytes_unchanged(
    path: str | Path,
    source_bytes: bytes,
    *,
    label: str,
) -> None:
    if Path(path).read_bytes() != source_bytes:
        raise S4BControlError(f"{label} input changed after validation")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _validate_repository(repository: Path, implementation_commit: str) -> None:
    if not repository.is_dir():
        raise S4BControlError("repository path is missing")
    if repository.resolve() != REPOSITORY_ROOT.resolve():
        raise S4BControlError("repository argument differs from the runner checkout")
    head = _git(repository, "rev-parse", "HEAD")
    if head != implementation_commit:
        raise S4BControlError("repository HEAD differs from authorized implementation commit")
    if _git(repository, "status", "--porcelain"):
        raise S4BControlError("controlled repository is dirty; preserve changes and stop")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "c4bdca6bead67351c5b7b96213586ffb55d50209", head],
        cwd=repository,
    )
    if ancestor.returncode != 0:
        raise S4BControlError("implementation commit does not descend from the S4B base commit")


def _code_identity(repository: Path, policy_path: Path) -> dict[str, Any]:
    return {
        "repository": str(repository.resolve()),
        "runner_sha256": sha256_file(Path(__file__)),
        "control_sha256": sha256_file(repository / "s4_low_fraction_control.py"),
        "trainer_sha256": sha256_file(repository / "trainer.py"),
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "core_policy_sha256": CORE_POLICY_SHA256,
    }


def _decode_human3_predictions(trainer: Any, batch: Any) -> dict[str, list[float]]:
    import torch

    with torch.no_grad():
        raw = trainer.predict_all_tasks(batch)
        return {
            task: trainer.decode_task_output(task, raw[task], apply_conformal=False)[
                "median"
            ]
            .detach()
            .cpu()
            .reshape(-1)
            .tolist()
            for task in HUMAN3_TASKS
        }


def _synthetic_selftest(policy_path: Path) -> dict[str, Any]:
    policy = load_s4b_policy(policy_path)
    import torch

    model = torch.nn.Linear(2, len(HUMAN3_TASKS) * 3, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        for task_index in range(len(HUMAN3_TASKS)):
            model.weight[task_index * 3, 0] = 1.0
            model.weight[task_index * 3, 1] = float(task_index)
    features = torch.tensor([[-1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    scalers = policy.assets[0].expected_scalers
    class SyntheticTrainer:
        def predict_all_tasks(self, batch: Any) -> dict[str, Any]:
            output = model(batch).reshape(-1, len(HUMAN3_TASKS), 3)
            return {task: output[:, index, :] for index, task in enumerate(HUMAN3_TASKS)}

        def decode_task_output(self, task: str, value: Any, *, apply_conformal: bool) -> dict[str, Any]:
            if apply_conformal:
                raise S4BControlError("synthetic closure must not apply conformal calibration")
            scaler = scalers[task]
            return {"median": value[:, 0] * scaler["std"] + scaler["mean"]}

    decoded = _decode_human3_predictions(SyntheticTrainer(), features)
    for task_index, task in enumerate(HUMAN3_TASKS):
        expected = [
            (-1.0 + task_index) * scalers[task]["std"] + scalers[task]["mean"],
            (1.0 + task_index) * scalers[task]["std"] + scalers[task]["mean"],
        ]
        if any(
            not math.isclose(actual, target, abs_tol=1e-6, rel_tol=1e-7)
            for actual, target in zip(decoded[task], expected)
        ):
            raise S4BControlError("synthetic Human3 decode/de-scale selftest differs")
    return {
        "status": "PASS",
        "synthetic_only": True,
        "human3_tasks": list(HUMAN3_TASKS),
        "batch_size": 2,
        "prediction_mode": "quantile",
        "model_tensors_unchanged": True,
        "training_called": False,
        "worker_shared_decode_path": True,
        "calibration_called": False,
        "real_data_accessed": False,
        "policy_sha256": policy.sha256,
    }


def _sample_table_from_store(
    store: Any,
    split: str,
    *,
    max_nodes_filter: int,
    selected_task_indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    from toxacute_datastore import SPLIT_CODES

    datastore_task_names = list(store.task_names)
    if selected_task_indices is None:
        selected_task_indices = [store.task_index(task) for task in HUMAN3_TASKS]
    selection = validate_datastore_task_selection(
        datastore_task_names,
        list(HUMAN3_TASKS),
        list(selected_task_indices),
    )
    selected_task_indices = selection["selected_task_indices"]

    samples = []
    indices = []
    for index, sample_id in enumerate(store.sample_ids):
        if int(store.split_codes[index]) != SPLIT_CODES[split]:
            continue
        labels = [
            float(store.labels[index, task_index])
            for task_index in selected_task_indices
        ]
        if not any(math.isfinite(value) for value in labels):
            continue
        if int(store.num_nodes[index]) > max_nodes_filter:
            raise S4BControlError("frozen max_nodes_filter would drop an evaluated sample")
        samples.append(
            {
                "sample_id": str(sample_id),
                "row_index": int(store.row_indices[index]),
                "labels": [value if math.isfinite(value) else None for value in labels],
            }
        )
        indices.append(index)
    ordered = sorted(
        zip(samples, indices),
        key=lambda item: (item[0]["row_index"], item[0]["sample_id"]),
    )
    return [item[0] for item in ordered], [item[1] for item in ordered]


def _params_and_store(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    datastore_dir: Path,
    split: str,
    gpu_id: str,
) -> tuple[Any, Any, Any]:
    import main as app

    configuration = copy.deepcopy(dict(payload["configuration"]))
    # This is a runtime cache/object, not a scientific identity and never an
    # S4B payload rewrite.  The verified DataStore below is the active context.
    configuration.pop("datastore_context", None)
    params = app.build_parser().parse_args([])
    vars(params).update(configuration)
    overrides = {
        "mode": ALLOWED_OPERATION,
        "inference_split": split,
        "load_path": str(checkpoint_path),
        "save_path": None,
        "data_store_dir": str(datastore_dir),
        "preprocessed_data_dir": None,
        "gpu_id": gpu_id,
        "num_loader_workers": 0,
        "fit_conformal": False,
    }
    vars(params).update(overrides)
    app.validate_params(params)
    app.seed_everything(params.seed)
    task_names = app.task_names_for_params(params)
    if tuple(task_names) != HUMAN3_TASKS or task_names != payload["task_names"]:
        raise S4BControlError("runtime/checkpoint Human3 task order differs")
    params.effective_rgcer_config = app.resolve_effective_rgcer_config(params)
    store = app._resolve_data_store(params, task_names, required=True)
    return app, params, store


def _receipt_file(preflight_dir: Path) -> Path:
    return preflight_dir / "preflight_receipt.json"


def _historical_reference_posix_path(asset: Any) -> PurePosixPath:
    run_dir = PurePosixPath(asset.run_dir)
    if (
        not run_dir.is_absolute()
        or ".." in run_dir.parts
        or run_dir.as_posix() != asset.run_dir
    ):
        raise S4BControlError(
            f"frozen run_dir is not a canonical absolute POSIX path: {asset.run_id}"
        )
    return run_dir / "diagnostics" / "best_validation_human3_predictions.csv"


def _build_reference_sources(policy: Any) -> list[dict[str, Any]]:
    sources = []
    for asset in policy.assets:
        posix_path = _historical_reference_posix_path(asset)
        live_path = Path(posix_path.as_posix())
        exists = live_path.is_file()
        sources.append(
            {
                "run_id": asset.run_id,
                "historical_csv_path": posix_path.as_posix(),
                "historical_csv_exists": exists,
                "historical_csv_sha256": sha256_file(live_path) if exists else None,
            }
        )
    return sources


def _verify_reference_sources(
    rows: Any, policy: Any, *, check_live_paths: bool
) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(policy.assets):
        raise S4BControlError("preflight historical reference coverage differs")
    result = {}
    for row, asset in zip(rows, policy.assets):
        expected_path = _historical_reference_posix_path(asset).as_posix()
        expected_keys = {
            "run_id",
            "historical_csv_path",
            "historical_csv_exists",
            "historical_csv_sha256",
        }
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise S4BControlError("preflight historical reference record differs")
        if row["run_id"] != asset.run_id or row["historical_csv_path"] != expected_path:
            raise S4BControlError("preflight historical reference identity differs")
        if type(row["historical_csv_exists"]) is not bool:
            raise S4BControlError("preflight historical reference existence fact is invalid")
        digest = row["historical_csv_sha256"]
        if row["historical_csv_exists"]:
            if not re_full_sha(digest):
                raise S4BControlError("preflight historical reference SHA is invalid")
        elif digest is not None:
            raise S4BControlError("missing historical reference unexpectedly has a SHA")
        if check_live_paths:
            path = Path(expected_path)
            if path.is_file() is not row["historical_csv_exists"]:
                raise S4BControlError("historical reference existence changed after preflight")
            if path.is_file() and sha256_file(path) != digest:
                raise S4BControlError("historical reference bytes changed after preflight")
        result[asset.run_id] = dict(row)
    return result


def _validate_recorded_datastore_source(
    datastore_source: Any, policy: Any
) -> dict[str, Any]:
    """Replay full-catalogue identity and Human3 column selection offline."""

    if not isinstance(datastore_source, Mapping):
        raise S4BControlError("server preflight DataStore source is not an object")
    recorded_metadata = {
        "format": datastore_source.get("format"),
        "format_version": datastore_source.get("datastore_format_version"),
        "graph_record_version": datastore_source.get("graph_record_version"),
        "build_id": datastore_source.get("datastore_build_id"),
        "datastore_fingerprint": datastore_source.get("datastore_fingerprint"),
        "raw_csv_sha256": datastore_source.get("raw_csv_sha256"),
        "feature_schema_version": datastore_source.get("feature_schema_version"),
        "max_path_distance": datastore_source.get("max_path_distance"),
        "splitting": datastore_source.get("splitting"),
        "split_seed": datastore_source.get("split_seed"),
        "split_ratios": datastore_source.get("split_ratios"),
        "split_manifest_hash": datastore_source.get("split_manifest_hash"),
        "num_samples": datastore_source.get("num_samples"),
        "num_tasks": datastore_source.get("num_tasks"),
        "labels_shape": datastore_source.get("labels_shape"),
        "task_names": datastore_source.get("datastore_task_names"),
        "build_complete": datastore_source.get("build_complete"),
    }
    replayed_datastore = validate_datastore_metadata(
        recorded_metadata, policy.assets[0]
    )
    for name in (
        "datastore_task_names",
        "requested_task_names",
        "selected_task_indices",
    ):
        if datastore_source.get(name) != replayed_datastore[name]:
            raise S4BControlError(
                f"server preflight DataStore task selection differs: {name}"
            )
    return replayed_datastore


def _verify_server_preflight(
    preflight_dir: Path,
    *,
    implementation_commit: str,
    policy_path: Path,
    authorization_sha256: str,
    output_run_id: str,
    expected_data_store_dir: str | None = None,
    check_live_reference_paths: bool = True,
) -> dict[str, Any]:
    policy = load_s4b_policy(policy_path)
    receipt = read_json(_receipt_file(preflight_dir))
    if not isinstance(receipt, Mapping):
        raise S4BControlError("server preflight receipt is not an object")
    required = {
        "schema_version": 1,
        "receipt_kind": "S4B_SERVER_PREFLIGHT",
        "status": "PASS",
        "task_id": TASK_ID,
        "output_run_id": output_run_id,
        "implementation_commit": implementation_commit,
        "policy_sha256": policy.sha256,
        "core_policy_sha256": CORE_POLICY_SHA256,
        "authorization_sha256": authorization_sha256,
        "runner_sha256": sha256_file(Path(__file__)),
        "control_sha256": sha256_file(REPOSITORY_ROOT / "s4_low_fraction_control.py"),
        "trainer_sha256": sha256_file(REPOSITORY_ROOT / "trainer.py"),
        "verified_asset_count": 40,
        "checkpoint_payload_count": 40,
        "test_sample_identity_sha256": FROZEN_TEST_SAMPLE_PROJECTION_SHA256,
        "test_sample_count": 28,
        "datastore_fingerprint": policy.assets[0].expected_data_config[
            "datastore_fingerprint"
        ],
        "frozen_source_outer_sha256": S3C3_OUTER_SHA256,
        "frozen_source_canonical_sha256": S3C3_CANONICAL_SHA256,
    }
    for name, expected in required.items():
        if name not in receipt or type(receipt[name]) is not type(expected) or receipt[name] != expected:
            raise S4BControlError(f"server preflight receipt field differs: {name}")
    if (
        expected_data_store_dir is not None
        and receipt.get("expected_data_store_dir") != expected_data_store_dir
    ):
        raise S4BControlError("server preflight DataStore path binding differs")
    validation_samples = read_json(preflight_dir / "validation_samples.json")
    test_samples = read_json(preflight_dir / "test_samples.json")
    validation_samples = validate_sample_table(validation_samples, split="validation")
    test_samples = validate_frozen_test_table(test_samples)
    if identity_sha256(validation_samples) != receipt.get("validation_sample_identity_sha256"):
        raise S4BControlError("server preflight validation sample identity differs")
    if identity_sha256(test_samples) != receipt["test_sample_identity_sha256"]:
        raise S4BControlError("server preflight test sample identity differs")
    sample_sources_path = preflight_dir / "sample_sources.json"
    if (
        not sample_sources_path.is_file()
        or sha256_file(sample_sources_path) != receipt.get("sample_sources_sha256")
    ):
        raise S4BControlError("server preflight sample source record differs")
    sample_sources = read_json(sample_sources_path)
    if not isinstance(sample_sources, Mapping):
        raise S4BControlError("server preflight sample source record is not an object")
    datastore_source = sample_sources.get("datastore", {})
    _validate_recorded_datastore_source(datastore_source, policy)
    if (
        datastore_source.get("datastore_fingerprint")
        != policy.assets[0].expected_data_config["datastore_fingerprint"]
        or datastore_source.get("datastore_json_sha256")
        != receipt.get("datastore_json_sha256")
        or sample_sources.get("validation", {}).get("sample_identity_sha256")
        != receipt["validation_sample_identity_sha256"]
        or sample_sources.get("test", {}).get("sample_identity_sha256")
        != FROZEN_TEST_SAMPLE_PROJECTION_SHA256
        or sample_sources.get("test", {}).get("source_outer_sha256")
        != S3C3_OUTER_SHA256
        or sample_sources.get("test", {}).get("source_canonical_sha256")
        != S3C3_CANONICAL_SHA256
    ):
        raise S4BControlError("server preflight sample source identity differs")
    asset_identity = read_json(preflight_dir / "asset_identity.json")
    if not isinstance(asset_identity, list) or len(asset_identity) != 40:
        raise S4BControlError("server preflight asset identity coverage differs")
    for row, asset in zip(asset_identity, policy.assets):
        expected = {
            "run_id": asset.run_id,
            "method": asset.method,
            "seed": asset.seed,
            "fraction_percent": asset.fraction_percent,
            "checkpoint_sha256": asset.checkpoint_sha256,
            "checkpoint_size_bytes": asset.checkpoint_size_bytes,
            "best_epoch": asset.best_epoch,
            "migration_applied": False,
            "model_tensor_count": 170,
        }
        if row != expected:
            raise S4BControlError(f"server preflight asset identity differs: {asset.run_id}")
    authorization_snapshot = preflight_dir / "authorization_snapshot.json"
    if sha256_file(authorization_snapshot) != authorization_sha256:
        raise S4BControlError("server preflight authorization snapshot differs")
    load_authorization(
        authorization_snapshot,
        policy=policy,
        implementation_commit=implementation_commit,
        split="validation",
        output_run_id=output_run_id,
        allow_fixture=False,
    )
    wsl_path = preflight_dir / "wsl_receipt_snapshot.json"
    if sha256_file(wsl_path) != receipt.get("wsl_receipt_sha256"):
        raise S4BControlError("server preflight WSL snapshot differs")
    verify_wsl_receipt(
        read_json(wsl_path),
        implementation_commit=implementation_commit,
        policy_sha256=policy.sha256,
        runner_sha256=receipt["runner_sha256"],
        control_sha256=receipt["control_sha256"],
        trainer_sha256=receipt["trainer_sha256"],
        output_run_id=output_run_id,
    )
    reference_path = preflight_dir / "reference_sources.json"
    if (
        not reference_path.is_file()
        or sha256_file(reference_path) != receipt.get("reference_sources_sha256")
    ):
        raise S4BControlError("server preflight historical reference inventory differs")
    references = read_json(reference_path)
    _verify_reference_sources(
        references, policy, check_live_paths=check_live_reference_paths
    )
    if receipt.get("reference_source_count") != 40:
        raise S4BControlError("server preflight historical reference count differs")
    return receipt


def preflight(arguments: argparse.Namespace) -> int:
    repository = arguments.repository.resolve()
    policy_path = arguments.policy.resolve()
    policy = load_s4b_policy(policy_path)
    _validate_repository(repository, arguments.implementation_commit)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = _code_identity(repository, policy_path)
    command_record = {
        "command": list(sys.argv),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "hostname": socket.gethostname(),
        "started": time.time(),
    }
    write_json(output / "command.json", command_record)
    shutil.copy2(policy_path, output / "s4b_low_fraction_policy.json")

    if arguments.side == "wsl":
        smoke = _synthetic_selftest(policy_path)
        write_json(output / "synthetic_selftest.json", smoke)
        receipt = {
            "schema_version": 1,
            "receipt_id": output.name,
            "receipt_kind": "S4B_WSL_PREFLIGHT",
            "task_id": TASK_ID,
            "implementation_commit": arguments.implementation_commit,
            "output_run_id": arguments.output_run_id,
            "policy_sha256": policy.sha256,
            "core_policy_sha256": CORE_POLICY_SHA256,
            "runner_sha256": code["runner_sha256"],
            "control_sha256": code["control_sha256"],
            "trainer_sha256": code["trainer_sha256"],
            "test_status": "PASS",
            "asset_accessed": False,
            "real_data_accessed": False,
            "training_called": False,
            "completed": time.time(),
        }
        write_json(_receipt_file(output), receipt)
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
        return 0

    if arguments.authorization is None or arguments.wsl_receipt is None:
        raise S4BControlError("server preflight requires authorization and WSL receipt")
    output_run_id = arguments.output_run_id
    authorization_path = arguments.authorization.resolve()
    authorization, authorization_bytes, authorization_sha = load_authorization_document(
        authorization_path,
        policy=policy,
        implementation_commit=arguments.implementation_commit,
        split="validation",
        output_run_id=output_run_id,
        allow_fixture=False,
    )
    wsl_path = arguments.wsl_receipt.resolve()
    wsl_document, wsl_bytes, wsl_sha = read_json_document(wsl_path)
    wsl = verify_wsl_receipt(
        wsl_document,
        implementation_commit=arguments.implementation_commit,
        policy_sha256=policy.sha256,
        runner_sha256=code["runner_sha256"],
        control_sha256=code["control_sha256"],
        trainer_sha256=code["trainer_sha256"],
        output_run_id=output_run_id,
    )
    aliases = verify_037_aliases(arguments.core_zip)
    frozen_test = list(aliases.frozen_samples)
    first = policy.assets[0]
    # Resolve the DataStore without loading a model.  The test projection is
    # immediately checked against accepted 037/034 evidence.
    pseudo_payload = {
        "configuration": {
            **dict(first.expected_configuration),
            **dict(first.expected_adaptation),
        },
        "task_names": list(HUMAN3_TASKS),
    }
    app, params, store = _params_and_store(
        pseudo_payload,
        checkpoint_path=Path(first.checkpoint_path),
        datastore_dir=arguments.datastore_dir.resolve(),
        split="validation",
        gpu_id="0",
    )
    del app
    try:
        datastore_identity = validate_datastore_metadata(store.metadata, first)
        if params.max_nodes_filter != first.expected_data_config["max_nodes_filter"]:
            raise S4BControlError("runtime max_nodes_filter differs from frozen identity")
        datastore_json = store.root / "datastore.json"
        if not datastore_json.is_file():
            raise S4BControlError("verified DataStore has no datastore.json")
        datastore_identity.update(
            {
                "root": str(store.root),
                "datastore_json_sha256": sha256_file(datastore_json),
            }
        )
        validation_samples, _ = _sample_table_from_store(
            store,
            "validation",
            max_nodes_filter=params.max_nodes_filter,
            selected_task_indices=datastore_identity["selected_task_indices"],
        )
        actual_test, _ = _sample_table_from_store(
            store,
            "test",
            max_nodes_filter=params.max_nodes_filter,
            selected_task_indices=datastore_identity["selected_task_indices"],
        )
    finally:
        store.close()
    validation_samples = validate_sample_table(validation_samples, split="validation")
    actual_test = validate_frozen_test_table(actual_test)
    if actual_test != frozen_test:
        raise S4BControlError("server DataStore test mapping differs from frozen 037/034")
    write_json(output / "validation_samples.json", validation_samples)
    write_json(output / "test_samples.json", actual_test)
    sample_sources = {
        "datastore": datastore_identity,
        "validation": {
            "source": "VERIFIED_DATASTORE_V2_BEFORE_WORKERS",
            "sample_identity_sha256": identity_sha256(validation_samples),
            "sample_count": len(validation_samples),
        },
        "test": {
            "source": "VERIFIED_DATASTORE_V2_AND_ACCEPTED_037_034_PROJECTION",
            "sample_identity_sha256": identity_sha256(actual_test),
            "sample_count": len(actual_test),
            "endpoint_n": [14, 13, 13],
            "source_outer_sha256": aliases.outer_sha256,
            "source_canonical_sha256": aliases.canonical_sha256,
        },
    }
    write_json(output / "sample_sources.json", sample_sources)

    asset_identity = []
    for asset in policy.assets:
        prepared = prepare_low_fraction_checkpoint(
            asset.checkpoint_path,
            policy=policy,
            run_id=asset.run_id,
            method=asset.method,
            seed=asset.seed,
            fraction_percent=asset.fraction_percent,
            operation=ALLOWED_OPERATION,
            split="validation",
            expected_policy_sha256=policy.sha256,
        )
        asset_identity.append(
            {
                "run_id": asset.run_id,
                "method": asset.method,
                "seed": asset.seed,
                "fraction_percent": asset.fraction_percent,
                "checkpoint_sha256": prepared.checkpoint_sha256,
                "checkpoint_size_bytes": prepared.checkpoint_size_bytes,
                "best_epoch": asset.best_epoch,
                "migration_applied": prepared.strict_identity["migration_applied"],
                "model_tensor_count": prepared.strict_identity["model_tensor_count"],
            }
        )
        del prepared
    write_json(output / "asset_identity.json", asset_identity)
    reference_sources = _build_reference_sources(policy)
    write_json(output / "reference_sources.json", reference_sources)
    authorization_snapshot = _write_exact_json_snapshot(
        output / "authorization_snapshot.json",
        authorization_bytes,
        authorization_sha,
        label="authorization",
    )
    snapshot_authorization = validate_authorization(
        authorization_snapshot,
        policy=policy,
        implementation_commit=arguments.implementation_commit,
        split="validation",
        output_run_id=output_run_id,
        allow_fixture=False,
    )
    if snapshot_authorization != authorization:
        raise S4BControlError("authorization snapshot content differs")
    wsl_snapshot = _write_exact_json_snapshot(
        output / "wsl_receipt_snapshot.json",
        wsl_bytes,
        wsl_sha,
        label="WSL receipt",
    )
    snapshot_wsl = verify_wsl_receipt(
        wsl_snapshot,
        implementation_commit=arguments.implementation_commit,
        policy_sha256=policy.sha256,
        runner_sha256=code["runner_sha256"],
        control_sha256=code["control_sha256"],
        trainer_sha256=code["trainer_sha256"],
        output_run_id=output_run_id,
    )
    if snapshot_wsl != wsl:
        raise S4BControlError("WSL receipt snapshot content differs")
    receipt = {
        "schema_version": 1,
        "receipt_kind": "S4B_SERVER_PREFLIGHT",
        "status": "PASS",
        "task_id": TASK_ID,
        "output_run_id": output_run_id,
        "implementation_commit": arguments.implementation_commit,
        "policy_sha256": policy.sha256,
        "core_policy_sha256": CORE_POLICY_SHA256,
        "authorization_sha256": authorization_sha,
        "output_run_id": arguments.output_run_id,
        "expected_data_store_dir": str(arguments.datastore_dir.resolve()),
        "wsl_receipt_sha256": wsl_sha,
        "runner_sha256": code["runner_sha256"],
        "control_sha256": code["control_sha256"],
        "trainer_sha256": code["trainer_sha256"],
        "verified_asset_count": 40,
        "checkpoint_payload_count": 40,
        "validation_sample_identity_sha256": identity_sha256(validation_samples),
        "validation_sample_count": len(validation_samples),
        "test_sample_identity_sha256": identity_sha256(actual_test),
        "test_sample_count": len(actual_test),
        "test_endpoint_n": [14, 13, 13],
        "datastore_fingerprint": datastore_identity["datastore_fingerprint"],
        "datastore_json_sha256": datastore_identity["datastore_json_sha256"],
        "sample_sources_sha256": sha256_file(output / "sample_sources.json"),
        "reference_sources_sha256": sha256_file(output / "reference_sources.json"),
        "reference_source_count": len(reference_sources),
        "frozen_source_outer_sha256": aliases.outer_sha256,
        "frozen_source_canonical_sha256": aliases.canonical_sha256,
        "training_called": False,
        "calibration_called": False,
        "completed": time.time(),
    }
    _require_source_bytes_unchanged(
        authorization_path,
        authorization_bytes,
        label="authorization",
    )
    _require_source_bytes_unchanged(
        wsl_path,
        wsl_bytes,
        label="WSL receipt",
    )
    write_json(_receipt_file(output), receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


def _verify_worker_output(
    run_output: Path,
    *,
    asset: Any,
    split: str,
    preflight_dir: Path,
    policy: Any,
    implementation_commit: str,
    authorization_sha256: str,
    output_run_id: str,
    expected_data_store_dir: str,
    check_live_reference_paths: bool = True,
) -> dict[str, Any]:
    independent_samples = read_json(preflight_dir / f"{split}_samples.json")
    worker_samples = read_json(run_output / "expected_samples.json")
    worker_samples = validate_sample_table(worker_samples, split=split)
    independent_samples = validate_sample_table(independent_samples, split=split)
    if identity_sha256(worker_samples) != identity_sha256(independent_samples):
        raise S4BControlError("worker expected_samples differ from preflight source")
    rows = parse_jsonl((run_output / "predictions.jsonl").read_bytes())
    calculated = validate_prediction_rows(
        rows,
        independent_samples,
        asset=asset,
        split=split,
    )
    stored_metrics = read_json(run_output / "metrics.json")
    validate_stored_metrics(stored_metrics, calculated)
    restoration = read_json(run_output / "restoration.json")
    validate_restoration_evidence(
        restoration,
        asset=asset,
        policy=policy,
        implementation_commit=implementation_commit,
        split=split,
        expected_data_store_dir=expected_data_store_dir,
    )
    binding = read_json(run_output / "sample_binding.json")
    expected_sample_sha = identity_sha256(independent_samples)
    if (
        binding.get("status") != "PASS"
        or binding.get("split") != split
        or binding.get("sample_identity_sha256") != expected_sample_sha
        or binding.get("sample_count") != len(independent_samples)
        or binding.get("prediction_row_count") != len(rows)
        or binding.get("source") != "SERVER_PREFLIGHT_INDEPENDENT_DATASTORE_TABLE"
    ):
        raise S4BControlError("worker sample binding differs")
    if split == "test":
        validate_frozen_test_table(independent_samples)
    else:
        references = _verify_reference_sources(
            read_json(preflight_dir / "reference_sources.json"),
            policy,
            check_live_paths=check_live_reference_paths,
        )
        source = read_json(run_output / "reference_source.json")
        if source != references[asset.run_id]:
            raise S4BControlError("worker historical reference source differs from preflight")
        comparison = read_json(run_output / "reference_comparison.json")
        expected_reference_level = (
            "HISTORICAL_SAMPLE_PREDICTIONS"
            if source["historical_csv_exists"]
            else "FROZEN_MACRO_ONLY"
        )
        if (
            comparison.get("status") != "PASS"
            or comparison.get("reference_level") != expected_reference_level
            or comparison.get("historical_csv_exists")
            is not source["historical_csv_exists"]
        ):
            raise S4BControlError("worker validation reference comparison differs")
        historical_copy = run_output / "historical_validation_original.csv"
        if source["historical_csv_exists"]:
            if not historical_copy.is_file():
                raise S4BControlError("worker historical CSV copy is missing")
            copy_sha = sha256_file(historical_copy)
            if copy_sha != source["historical_csv_sha256"]:
                raise S4BControlError(
                    "worker historical CSV copy SHA differs from preflight original"
                )
            if comparison.get("historical_csv_sha256") != copy_sha:
                raise S4BControlError(
                    "worker historical comparison SHA differs from preflight original"
                )
            recalculated_comparison = compare_validation_reference(
                rows, calculated, asset, historical_copy
            )
            if (
                recalculated_comparison.get("historical_csv_sha256") != copy_sha
                or comparison.get("compared_finite_rows")
                != recalculated_comparison.get("compared_finite_rows")
                or comparison.get("sample_equivalence_claimed") is not True
            ):
                raise S4BControlError("worker historical CSV comparison is not reproducible")
        else:
            if source["historical_csv_sha256"] is not None:
                raise S4BControlError("missing historical reference unexpectedly has a SHA")
            if historical_copy.exists():
                raise S4BControlError("macro-only worker unexpectedly copied a historical CSV")
            recalculated_comparison = compare_validation_reference(
                rows,
                calculated,
                asset,
                run_output / "confirmed_missing_historical_validation.csv",
            )
            if (
                comparison.get("expected_macro_rmse")
                != recalculated_comparison["expected_macro_rmse"]
                or comparison.get("actual_macro_rmse")
                != recalculated_comparison["actual_macro_rmse"]
                or comparison.get("sample_equivalence_claimed") is not False
            ):
                raise S4BControlError("worker macro-only comparison is not reproducible")
    evidence_names = [
        "expected_samples.json",
        "predictions.jsonl",
        "metrics.json",
        "restoration.json",
        "sample_binding.json",
    ]
    if split == "validation":
        evidence_names.extend(["reference_comparison.json", "reference_source.json"])
        if source["historical_csv_exists"]:
            evidence_names.append("historical_validation_original.csv")
    evidence_files = {name: sha256_file(run_output / name) for name in evidence_names}
    return {
        "run_id": asset.run_id,
        "method": asset.method,
        "seed": asset.seed,
        "fraction_percent": asset.fraction_percent,
        "split": split,
        "status": "PASS",
        "checkpoint_sha256": asset.checkpoint_sha256,
        "policy_sha256": policy.sha256,
        "implementation_commit": implementation_commit,
        "output_run_id": output_run_id,
        "authorization_sha256": authorization_sha256,
        "preflight_receipt_sha256": sha256_file(_receipt_file(preflight_dir)),
        "prediction_sha256": sha256_file(run_output / "predictions.jsonl"),
        "metrics": calculated,
        **(
            {
                "reference_level": comparison["reference_level"],
                "historical_csv_exists": source["historical_csv_exists"],
                "evidence_files": evidence_files,
                "evidence_identity_sha256": identity_sha256(evidence_files),
            }
            if split == "validation"
            else {}
        ),
    }


def worker(arguments: argparse.Namespace) -> int:
    policy = load_s4b_policy(arguments.policy)
    asset = policy.asset(arguments.asset)
    authorization, authorization_sha = load_authorization(
        arguments.authorization,
        policy=policy,
        implementation_commit=arguments.implementation_commit,
        split=arguments.split,
        output_run_id=arguments.output_run_id,
        allow_fixture=False,
    )
    del authorization
    _validate_repository(arguments.repository.resolve(), arguments.implementation_commit)
    preflight_dir = arguments.preflight.resolve()
    _verify_server_preflight(
        preflight_dir,
        implementation_commit=arguments.implementation_commit,
        policy_path=arguments.policy.resolve(),
        authorization_sha256=authorization_sha,
        output_run_id=arguments.output_run_id,
        expected_data_store_dir=str(arguments.datastore_dir.resolve()),
    )
    if arguments.split == "test":
        if arguments.validation_dir is None or arguments.validation_gate is None:
            raise S4BControlError("test worker requires validation evidence and gate")
        _verify_validation_evidence(
            arguments.validation_dir.resolve(),
            gate_path=arguments.validation_gate.resolve(),
            preflight_dir=preflight_dir,
            policy=policy,
            policy_path=arguments.policy.resolve(),
            implementation_commit=arguments.implementation_commit,
            authorization_sha256=authorization_sha,
            output_run_id=arguments.output_run_id,
            expected_data_store_dir=str(arguments.datastore_dir.resolve()),
            check_live_reference_paths=True,
        )
    prepared = prepare_low_fraction_checkpoint(
        asset.checkpoint_path,
        policy=policy,
        run_id=asset.run_id,
        method=asset.method,
        seed=asset.seed,
        fraction_percent=asset.fraction_percent,
        operation=ALLOWED_OPERATION,
        split=arguments.split,
        expected_policy_sha256=policy.sha256,
    )
    checkpoint_path = Path(prepared.checkpoint_path)
    app, params, store = _params_and_store(
        prepared.payload,
        checkpoint_path=checkpoint_path,
        datastore_dir=arguments.datastore_dir.resolve(),
        split=arguments.split,
        gpu_id="0",
    )
    try:
        datastore_identity = validate_datastore_metadata(store.metadata, asset)
        actual_samples, graph_indices = _sample_table_from_store(
            store,
            arguments.split,
            max_nodes_filter=params.max_nodes_filter,
            selected_task_indices=datastore_identity["selected_task_indices"],
        )
        independent_samples = read_json(
            preflight_dir / f"{arguments.split}_samples.json"
        )
        independent_samples = validate_sample_table(
            independent_samples, split=arguments.split
        )
        if identity_sha256(actual_samples) != identity_sha256(independent_samples):
            raise S4BControlError("worker DataStore mapping differs from preflight samples")
        if arguments.split == "test":
            validate_frozen_test_table(actual_samples)
        else:
            validate_sample_table(actual_samples, split="validation")

        output = arguments.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        # Copy the independently generated table byte-for-byte; never derive an
        # expected table from predictions emitted by this worker.
        shutil.copy2(
            preflight_dir / f"{arguments.split}_samples.json",
            output / "expected_samples.json",
        )
        architecture_kwargs, optimizer_parameters = app.prepare_args(params)
        encoder, architecture, decoders = app._build_model_components(
            params, list(HUMAN3_TASKS), app._device_from_params(params)
        )
        trainer = app.Trainer(
            task_dict=app.build_task_dict(params, list(HUMAN3_TASKS)),
            weighting=app.weighting_method.__dict__[params.weighting],
            architecture=architecture,
            encoder_class=encoder,
            decoders=decoders,
            optim_param=optimizer_parameters,
            args=params,
            save_path=None,
            load_path=str(checkpoint_path),
            **architecture_kwargs,
        )
        if trainer.loaded_epoch != asset.best_epoch:
            raise S4BControlError("Trainer loaded epoch differs from frozen best")
        import torch

        if set(trainer.model.state_dict()) != set(prepared.payload["model_state"]):
            raise S4BControlError("restored model tensor names differ")
        for name, tensor in trainer.model.state_dict().items():
            if not torch.equal(
                tensor.detach().cpu(), prepared.payload["model_state"][name].detach().cpu()
            ):
                raise S4BControlError(f"restored model tensor bytes differ: {name}")
        original_model_sha = tensor_state_sha256(prepared.payload["model_state"])
        before_model_sha = tensor_state_sha256(trainer.model.state_dict())
        checkpoint_sha_before = sha256_file(checkpoint_path)
        trainer.model.eval()
        collator = app.DataCollator(
            spatial_pos_max_clip=params.spatial_pos_clip,
            max_node_filter=None,
        )
        rows = []
        for offset in range(0, len(graph_indices), params.bs):
            indices = graph_indices[offset : offset + params.bs]
            batch = collator([store.get_graph_data(index) for index in indices]).to(
                trainer.device
            )
            if batch.get("is_empty", False):
                raise S4BControlError("collator unexpectedly produced an empty batch")
            expected_ids = [
                actual_samples[offset + local]["sample_id"]
                for local in range(len(indices))
            ]
            if list(batch.sample_id) != expected_ids:
                raise S4BControlError("collated sample order differs")
            decoded = _decode_human3_predictions(trainer, batch)
            for local in range(len(indices)):
                sample = actual_samples[offset + local]
                for task_index, task in enumerate(HUMAN3_TASKS):
                    rows.append(
                        {
                            "run_id": asset.run_id,
                            "method": asset.method,
                            "seed": asset.seed,
                            "fraction_percent": asset.fraction_percent,
                            "sample_id": sample["sample_id"],
                            "row_index": sample["row_index"],
                            "split": arguments.split,
                            "endpoint": task,
                            "y_true": sample["labels"][task_index],
                            "y_pred": float(decoded[task][local]),
                            "mask": sample["labels"][task_index] is not None,
                        }
                    )
        metrics = validate_prediction_rows(
            rows,
            independent_samples,
            asset=asset,
            split=arguments.split,
        )
        write_jsonl(output / "predictions.jsonl", rows)
        write_json(output / "metrics.json", metrics)

        if arguments.split == "validation":
            references = _verify_reference_sources(
                read_json(preflight_dir / "reference_sources.json"),
                policy,
                check_live_paths=True,
            )
            reference_source = references[asset.run_id]
            historical_csv = Path(reference_source["historical_csv_path"])
            comparison = compare_validation_reference(
                rows, metrics, asset, historical_csv
            )
            if reference_source["historical_csv_exists"]:
                shutil.copy2(historical_csv, output / "historical_validation_original.csv")
            write_json(output / "reference_source.json", reference_source)
            write_json(output / "reference_comparison.json", comparison)

        after_model_sha = tensor_state_sha256(trainer.model.state_dict())
        checkpoint_sha_after = sha256_file(checkpoint_path)
        if before_model_sha != after_model_sha:
            raise S4BControlError("model tensors changed during inference")
        if checkpoint_sha_before != checkpoint_sha_after or checkpoint_sha_after != asset.checkpoint_sha256:
            raise S4BControlError("checkpoint bytes changed during inference")
        write_json(
            output / "sample_binding.json",
            {
                "status": "PASS",
                "source": "SERVER_PREFLIGHT_INDEPENDENT_DATASTORE_TABLE",
                "split": arguments.split,
                "sample_identity_sha256": identity_sha256(independent_samples),
                "sample_count": len(independent_samples),
                "prediction_row_count": len(rows),
                "exact_ordered_match": True,
                "frozen_test_projection_sha256": (
                    FROZEN_TEST_SAMPLE_PROJECTION_SHA256
                    if arguments.split == "test"
                    else None
                ),
            },
        )
        write_json(
            output / "restoration.json",
            {
                "schema_version": 1,
                "run_id": asset.run_id,
                "method": asset.method,
                "seed": asset.seed,
                "fraction_percent": asset.fraction_percent,
                "policy_sha256": policy.sha256,
                "implementation_commit": arguments.implementation_commit,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": asset.checkpoint_sha256,
                "checkpoint_size_bytes": asset.checkpoint_size_bytes,
                "loaded_epoch": asset.best_epoch,
                "checkpoint_version": 6,
                "scaler_identity_sha256": identity_sha256(asset.expected_scalers),
                "task_scalers": dict(asset.expected_scalers),
                "effective_configuration": prepared.strict_identity[
                    "effective_configuration"
                ],
                "architecture_config": prepared.strict_identity[
                    "architecture_config"
                ],
                "runtime_overrides": {
                    name: vars(params)[name] for name in RUNTIME_OVERRIDE_NAMES
                },
                "runtime_override_names": list(RUNTIME_OVERRIDE_NAMES),
                "model_state_sha_algorithm": MODEL_STATE_SHA_ALGORITHM,
                "original_model_state_sha256": original_model_sha,
                "model_state_sha256_before": before_model_sha,
                "model_state_sha256_after": after_model_sha,
                "checkpoint_sha256_before": checkpoint_sha_before,
                "checkpoint_sha256_after": checkpoint_sha_after,
                "migration_applied": False,
                "architecture_additions": [],
                "init_overlay_applied": False,
                "training_called": False,
                "calibration_called": False,
                "model_unchanged": True,
                "checkpoint_unchanged": True,
                "routing_enabled": trainer.loaded_routing_enabled,
            },
        )
        print("S4B_WORKER_COMPLETE", asset.run_id, arguments.split)
        return 0
    finally:
        store.close()


def _gpu_uuid_if_idle(gpu_index: str) -> str:
    if gpu_index not in {"0", "1", "2", "3"}:
        raise S4BControlError("S4B GPU index must be one of 0,1,2,3")
    identity = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_index,
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if identity.returncode not in {0, 14} or not identity.stdout.strip().startswith("GPU-"):
        raise S4BControlError(f"GPU {gpu_index} UUID binding failed")
    occupancy = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_index,
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    pids = [
        line.strip()
        for line in occupancy.stdout.splitlines()
        if line.strip() and line.strip() != "No running processes found"
    ]
    if occupancy.returncode not in {0, 14} or pids:
        raise S4BControlError(f"GPU {gpu_index} is occupied or unverifiable")
    return identity.stdout.strip()


def _run_worker_process(
    arguments: argparse.Namespace,
    *,
    asset: Any,
    gpu_index: str,
    output: Path,
) -> None:
    uuid = _gpu_uuid_if_idle(gpu_index)
    run_output = output / f"{arguments.split}_{asset.run_id}"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--policy",
        str(arguments.policy.resolve()),
        "--authorization",
        str(arguments.authorization.resolve()),
        "--implementation-commit",
        arguments.implementation_commit,
        "--output-run-id",
        arguments.output_run_id,
        "--repository",
        str(arguments.repository.resolve()),
        "--preflight",
        str(arguments.preflight.resolve()),
        "--datastore-dir",
        str(arguments.datastore_dir.resolve()),
        "--split",
        arguments.split,
        "--asset",
        asset.run_id,
        "--output",
        str(run_output),
    ]
    if arguments.split == "test":
        command.extend(
            [
                "--validation-dir",
                str(arguments.validation_dir.resolve()),
                "--validation-gate",
                str(arguments.validation_gate.resolve()),
            ]
        )
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=uuid,
        PYTHONPATH=str(REPOSITORY_ROOT),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    log_root = output / "logs"
    started = time.time()
    with (log_root / f"{asset.run_id}.out").open("xb") as stdout_handle, (
        log_root / f"{asset.run_id}.err"
    ).open("xb") as stderr_handle:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    write_json(
        log_root / f"{asset.run_id}.json",
        {
            "command": command,
            "cwd": str(REPOSITORY_ROOT),
            "gpu_index": gpu_index,
            "gpu_uuid": uuid,
            "started": started,
            "finished": time.time(),
            "exit_code": result.returncode,
        },
    )
    if result.returncode != 0:
        raise S4BControlError(
            f"worker failed: {asset.run_id}, exit={result.returncode}"
        )


def _validation_log_identity(validation_dir: Path) -> str:
    logs = validation_dir / "logs"
    expected = {
        f"{run_id}.{suffix}"
        for run_id in S4B_RUN_IDS
        for suffix in ("out", "err", "json")
    }
    actual = {path.name for path in logs.iterdir() if path.is_file()} if logs.is_dir() else set()
    if actual != expected:
        raise S4BControlError("validation worker log coverage differs")
    hashes = {}
    for name in sorted(expected):
        path = logs / name
        hashes[name] = sha256_file(path)
        if name.endswith(".json"):
            record = read_json(path)
            if record.get("exit_code") != 0:
                raise S4BControlError("validation worker log reports a failure")
    return identity_sha256(hashes)


def _verify_validation_evidence(
    validation_dir: Path,
    *,
    gate_path: Path,
    preflight_dir: Path,
    policy: Any,
    policy_path: Path,
    implementation_commit: str,
    authorization_sha256: str,
    output_run_id: str,
    expected_data_store_dir: str,
    check_live_reference_paths: bool,
) -> dict[str, Any]:
    embedded_gate = validation_dir / "validation_gate.json"
    if (
        not embedded_gate.is_file()
        or not gate_path.is_file()
        or embedded_gate.read_bytes() != gate_path.read_bytes()
    ):
        raise S4BControlError("validation gate is not byte-bound to its evidence directory")
    receipt = _verify_server_preflight(
        preflight_dir,
        implementation_commit=implementation_commit,
        policy_path=policy_path,
        authorization_sha256=authorization_sha256,
        output_run_id=output_run_id,
        expected_data_store_dir=expected_data_store_dir,
        check_live_reference_paths=check_live_reference_paths,
    )
    batch_identity = read_json(validation_dir / "batch_identity.json")
    expected_batch = {
        "task_id": TASK_ID,
        "split": "validation",
        "output_run_id": output_run_id,
        "implementation_commit": implementation_commit,
        "policy_sha256": policy.sha256,
        "authorization_sha256": authorization_sha256,
        "preflight_receipt_sha256": sha256_file(_receipt_file(preflight_dir)),
    }
    for name, expected in expected_batch.items():
        if batch_identity.get(name) != expected:
            raise S4BControlError(f"validation batch identity differs: {name}")
    status = read_json(validation_dir / "batch_status.json")
    if (
        status.get("split") != "validation"
        or status.get("execution_status") != "FINISHED"
        or status.get("completed_run_count") != 40
        or status.get("failed_run_count") != 0
        or status.get("not_executed_run_ids") != []
    ):
        raise S4BControlError("validation batch is not a complete forty-run result")
    if read_json(validation_dir / "errors.json") != []:
        raise S4BControlError("validation batch carries errors")
    results = []
    for asset in policy.assets:
        results.append(
            _verify_worker_output(
                validation_dir / f"validation_{asset.run_id}",
                asset=asset,
                split="validation",
                preflight_dir=preflight_dir,
                policy=policy,
                implementation_commit=implementation_commit,
                authorization_sha256=authorization_sha256,
                output_run_id=output_run_id,
                expected_data_store_dir=expected_data_store_dir,
                check_live_reference_paths=check_live_reference_paths,
            )
        )
    stored_results = read_json(validation_dir / "completed_results.json")
    if stored_results != results:
        raise S4BControlError("validation completed results do not replay")
    log_identity = _validation_log_identity(validation_dir)
    gate = read_json(gate_path)
    return verify_validation_gate(
        gate,
        policy=policy,
        implementation_commit=implementation_commit,
        output_run_id=output_run_id,
        authorization_sha256=authorization_sha256,
        preflight_receipt_sha256=sha256_file(_receipt_file(preflight_dir)),
        validation_log_identity_sha256=log_identity,
    )


def run_batch(arguments: argparse.Namespace) -> int:
    repository = arguments.repository.resolve()
    policy = load_s4b_policy(arguments.policy)
    _validate_repository(repository, arguments.implementation_commit)
    authorization, authorization_sha = load_authorization(
        arguments.authorization,
        policy=policy,
        implementation_commit=arguments.implementation_commit,
        split=arguments.split,
        output_run_id=arguments.output_run_id,
        allow_fixture=False,
    )
    del authorization
    preflight_dir = arguments.preflight.resolve()
    _verify_server_preflight(
        preflight_dir,
        implementation_commit=arguments.implementation_commit,
        policy_path=arguments.policy.resolve(),
        authorization_sha256=authorization_sha,
        output_run_id=arguments.output_run_id,
        expected_data_store_dir=str(arguments.datastore_dir.resolve()),
    )
    if arguments.split == "test":
        if arguments.validation_gate is None or arguments.validation_dir is None:
            raise S4BControlError("test run requires validation evidence and gate")
        _verify_validation_evidence(
            arguments.validation_dir.resolve(),
            gate_path=arguments.validation_gate.resolve(),
            preflight_dir=preflight_dir,
            policy=policy,
            policy_path=arguments.policy.resolve(),
            implementation_commit=arguments.implementation_commit,
            authorization_sha256=authorization_sha,
            output_run_id=arguments.output_run_id,
            expected_data_store_dir=str(arguments.datastore_dir.resolve()),
            check_live_reference_paths=True,
        )
    gpus = list(arguments.gpus)
    if not gpus or len(gpus) > 4 or len(set(gpus)) != len(gpus):
        raise S4BControlError("run requires one to four unique GPU indices")
    for gpu in gpus:
        if gpu not in {"0", "1", "2", "3"}:
            raise S4BControlError("GPU indices must be selected from 0..3")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    write_json(
        output / "batch_identity.json",
        {
            "schema_version": 1,
            "task_id": TASK_ID,
            "split": arguments.split,
            "output_run_id": arguments.output_run_id,
            "implementation_commit": arguments.implementation_commit,
            "policy_sha256": policy.sha256,
            "authorization_sha256": authorization_sha,
            "preflight_receipt_sha256": sha256_file(_receipt_file(preflight_dir)),
            "runner_sha256": sha256_file(Path(__file__)),
            "control_sha256": sha256_file(REPOSITORY_ROOT / "s4_low_fraction_control.py"),
            "trainer_sha256": sha256_file(REPOSITORY_ROOT / "trainer.py"),
            "gpu_indices": gpus,
            "max_independent_workers": len(gpus),
            "ddp_used": False,
            "started": time.time(),
        },
    )
    completed = []
    errors = []
    assets = list(policy.assets)
    for offset in range(0, len(assets), len(gpus)):
        wave = assets[offset : offset + len(gpus)]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = {
                pool.submit(
                    _run_worker_process,
                    arguments,
                    asset=asset,
                    gpu_index=gpus[index],
                    output=output,
                ): asset
                for index, asset in enumerate(wave)
            }
            for future in as_completed(futures):
                asset = futures[future]
                try:
                    future.result()
                    completed.append(
                        _verify_worker_output(
                            output / f"{arguments.split}_{asset.run_id}",
                            asset=asset,
                            split=arguments.split,
                            preflight_dir=preflight_dir,
                            policy=policy,
                            implementation_commit=arguments.implementation_commit,
                            authorization_sha256=authorization_sha,
                            output_run_id=arguments.output_run_id,
                            expected_data_store_dir=str(arguments.datastore_dir.resolve()),
                        )
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "run_id": asset.run_id,
                            "split": arguments.split,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        if errors:
            break
    ordered = [
        next(row for row in completed if row["run_id"] == run_id)
        for run_id in S4B_RUN_IDS
        if any(row["run_id"] == run_id for row in completed)
    ]
    write_json(output / "completed_results.json", ordered)
    write_json(output / "errors.json", errors)
    status = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "split": arguments.split,
        "execution_status": "FINISHED" if not errors and len(ordered) == 40 else "PARTIAL_OR_FAILED",
        "acceptance_status": "PENDING_CODEX_REVIEW",
        "completed_run_count": len(ordered),
        "failed_run_count": len(errors),
        "not_executed_run_ids": [
            run_id
            for run_id in S4B_RUN_IDS
            if run_id not in {row["run_id"] for row in ordered}
            and run_id not in {row["run_id"] for row in errors}
        ],
        "finished": time.time(),
    }
    write_json(output / "batch_status.json", status)
    if arguments.split == "validation":
        gate = build_validation_gate(
            ordered,
            policy=policy,
            implementation_commit=arguments.implementation_commit,
            output_run_id=arguments.output_run_id,
            authorization_sha256=authorization_sha,
            preflight_receipt_sha256=sha256_file(_receipt_file(preflight_dir)),
            validation_log_identity_sha256=_validation_log_identity(output),
        )
        write_json(output / "validation_gate.json", gate)
    print(json.dumps(status, ensure_ascii=False, allow_nan=False))
    return 0 if status["execution_status"] == "FINISHED" else 1


def _pack_directory(root: Path, target: Path) -> dict[str, Any]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise S4BControlError("evidence directory contains a symlink")
        if path.is_file():
            if path.suffix.lower() in {".pt", ".pth", ".pkl", ".joblib", ".lmdb"}:
                raise S4BControlError(f"evidence contains a prohibited asset: {path}")
            files.append(path)
    checksum_path = root / "checksums.sha256"
    if checksum_path in files or checksum_path.exists():
        raise S4BControlError("evidence checksum manifest already exists")
    checksums = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)
    }
    with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
        for name, digest in checksums.items():
            handle.write(f"{digest}  {name}\n")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [checksum_path]:
            archive.write(path, path.relative_to(root).as_posix())
    sidecar = Path(str(target) + ".sha256")
    sidecar.write_text(f"{sha256_file(target)}  {target.name}\n", encoding="utf-8")
    return {
        "archive": str(target),
        "archive_sha256": sha256_file(target),
        "member_count": len(files) + 1,
        "sidecar": str(sidecar),
    }


def finalize(arguments: argparse.Namespace) -> int:
    policy = load_s4b_policy(arguments.policy)
    _validate_repository(arguments.repository.resolve(), arguments.implementation_commit)
    authorization, authorization_sha = load_authorization(
        arguments.authorization,
        policy=policy,
        implementation_commit=arguments.implementation_commit,
        split="test",
        output_run_id=arguments.output_run_id,
        allow_fixture=False,
    )
    del authorization
    preflight_dir = arguments.preflight.resolve()
    _verify_server_preflight(
        preflight_dir,
        implementation_commit=arguments.implementation_commit,
        policy_path=arguments.policy.resolve(),
        authorization_sha256=authorization_sha,
        output_run_id=arguments.output_run_id,
        expected_data_store_dir=str(arguments.datastore_dir.resolve()),
    )
    _verify_validation_evidence(
        arguments.validation_dir.resolve(),
        gate_path=arguments.validation_gate.resolve(),
        preflight_dir=preflight_dir,
        policy=policy,
        policy_path=arguments.policy.resolve(),
        implementation_commit=arguments.implementation_commit,
        authorization_sha256=authorization_sha,
        output_run_id=arguments.output_run_id,
        expected_data_store_dir=str(arguments.datastore_dir.resolve()),
        check_live_reference_paths=True,
    )
    test_root = arguments.test_run_dir.resolve()
    batch_status = read_json(test_root / "batch_status.json")
    if (
        batch_status.get("split") != "test"
        or batch_status.get("execution_status") != "FINISHED"
        or batch_status.get("completed_run_count") != 40
        or batch_status.get("failed_run_count") != 0
    ):
        raise S4BControlError("test batch is not a complete forty-run result")
    low_results = []
    for asset in policy.assets:
        low_results.append(
            _verify_worker_output(
                test_root / f"test_{asset.run_id}",
                asset=asset,
                split="test",
                preflight_dir=preflight_dir,
                policy=policy,
                implementation_commit=arguments.implementation_commit,
                authorization_sha256=authorization_sha,
                output_run_id=arguments.output_run_id,
                expected_data_store_dir=str(arguments.datastore_dir.resolve()),
            )
        )
    test_identity = read_json(test_root / "batch_identity.json")
    expected_test_identity = {
        "task_id": TASK_ID,
        "split": "test",
        "output_run_id": arguments.output_run_id,
        "implementation_commit": arguments.implementation_commit,
        "policy_sha256": policy.sha256,
        "authorization_sha256": authorization_sha,
        "preflight_receipt_sha256": sha256_file(_receipt_file(preflight_dir)),
    }
    for name, expected in expected_test_identity.items():
        if test_identity.get(name) != expected:
            raise S4BControlError(f"test batch identity differs: {name}")
    if read_json(test_root / "errors.json") != []:
        raise S4BControlError("test batch carries errors")
    if read_json(test_root / "completed_results.json") != low_results:
        raise S4BControlError("test completed results do not replay")
    aliases = verify_037_aliases(arguments.core_zip)
    analysis = verify_analysis_manifest(arguments.analysis_manifest, aliases)
    unified = build_unified_50_metrics(low_results, aliases, policy=policy)

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "low_fraction_test").mkdir()
    (output / "aliases_100").mkdir()
    (output / "preflight_snapshots").mkdir()
    (output / "batch_evidence").mkdir()
    shutil.copytree(arguments.validation_dir.resolve(), output / "validation_evidence")
    for asset in policy.assets:
        shutil.copytree(
            test_root / f"test_{asset.run_id}",
            output / "low_fraction_test" / asset.run_id,
        )
    shutil.copy2(
        preflight_dir / "validation_samples.json",
        output / "preflight_snapshots/validation_samples.json",
    )
    shutil.copy2(
        preflight_dir / "test_samples.json",
        output / "preflight_snapshots/test_samples.json",
    )
    shutil.copy2(arguments.validation_gate, output / "validation_gate.json")
    for name in (
        "batch_identity.json",
        "batch_status.json",
        "completed_results.json",
        "errors.json",
    ):
        shutil.copy2(test_root / name, output / "batch_evidence" / name)
    if (test_root / "logs").is_dir():
        shutil.copytree(test_root / "logs", output / "batch_evidence/logs")
    for name in (
        "preflight_receipt.json",
        "sample_sources.json",
        "reference_sources.json",
        "asset_identity.json",
        "authorization_snapshot.json",
        "wsl_receipt_snapshot.json",
        "command.json",
    ):
        shutil.copy2(preflight_dir / name, output / "preflight_snapshots" / name)
    alias_index = []
    for alias in aliases.aliases:
        alias_dir = output / "aliases_100" / alias.run_id
        alias_dir.mkdir()
        prediction_path = alias_dir / "predictions.jsonl"
        metrics_path = alias_dir / "metrics.json"
        prediction_path.write_bytes(alias.prediction_bytes)
        metrics_path.write_bytes(alias.metrics_bytes)
        alias_record = {
            "run_id": alias.run_id,
            "method": alias.method,
            "seed": alias.seed,
            "fraction_percent": 100,
            "source": "ALIAS_037_NO_RERUN",
            "source_outer_sha256": aliases.outer_sha256,
            "source_canonical_sha256": aliases.canonical_sha256,
            "source_prediction_member": alias.prediction_member,
            "source_metrics_member": alias.metrics_member,
            "prediction_sha256": alias.prediction_sha256,
            "metrics_sha256": alias.metrics_sha256,
            "local_prediction_path": prediction_path.relative_to(output).as_posix(),
            "local_metrics_path": metrics_path.relative_to(output).as_posix(),
            "rerun": False,
        }
        write_json(alias_dir / "alias.json", alias_record)
        alias_index.append(alias_record)
    write_json(output / "aliases_100.json", alias_index)
    write_json(output / "unified_50_test_metrics.json", unified)
    write_json(
        output / "source_archives.json",
        {
            "s3c3_outer_sha256": aliases.outer_sha256,
            "s3c3_canonical_sha256": aliases.canonical_sha256,
            "analysis_manifest": analysis,
            "frozen_test_sample_projection_sha256": FROZEN_TEST_SAMPLE_PROJECTION_SHA256,
        },
    )
    final_status = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "COMPLETE_PENDING_CODEX_REVIEW",
        "acceptance_status": "PENDING_CODEX_REVIEW",
        "implementation_commit": arguments.implementation_commit,
        "policy_sha256": policy.sha256,
        "authorization_sha256": authorization_sha,
        "output_run_id": arguments.output_run_id,
        "expected_data_store_dir": str(arguments.datastore_dir.resolve()),
        "preflight_receipt_sha256": sha256_file(_receipt_file(preflight_dir)),
        "low_fraction_test_count": 40,
        "alias_100_count": 10,
        "unified_result_count": 50,
        "failed_run_ids": [],
        "not_executed_run_ids": [],
        "training_called": False,
        "test_selected_best": False,
        "alias_rerun_count": 0,
        "completed": time.time(),
    }
    write_json(output / "final_status.json", final_status)
    target = Path(str(output) + ".zip")
    package = _pack_directory(output, target)
    print(json.dumps(package, ensure_ascii=False, allow_nan=False))
    return 0


def verify_archive(arguments: argparse.Namespace) -> int:
    policy = load_s4b_policy(arguments.policy)
    archive_path = arguments.archive.resolve()
    sidecar = Path(str(archive_path) + ".sha256")
    if not sidecar.is_file():
        raise S4BControlError("archive SHA sidecar is missing")
    sidecar_fields = sidecar.read_text(encoding="utf-8").split()
    if len(sidecar_fields) != 2 or sidecar_fields[1] != archive_path.name:
        raise S4BControlError("archive SHA sidecar format differs")
    archive_sha = sha256_file(archive_path)
    if sidecar_fields[0] != archive_sha:
        raise S4BControlError("archive external SHA differs")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise S4BControlError("final archive duplicate/CRC validation failed")
        if "checksums.sha256" not in names:
            raise S4BControlError("final archive checksum manifest is missing")
        checksums = {}
        for line in archive.read("checksums.sha256").decode("utf-8").splitlines():
            match = line.split("  ", 1)
            if len(match) != 2 or re_full_sha(match[0]) is False or match[1] in checksums:
                raise S4BControlError("final archive checksum line is invalid")
            checksums[match[1]] = match[0]
        if set(checksums) != set(names) - {"checksums.sha256"}:
            raise S4BControlError("final archive checksum coverage differs")
        for name, digest in checksums.items():
            if sha256_bytes(archive.read(name)) != digest:
                raise S4BControlError(f"final archive member SHA differs: {name}")

        def member_json(name: str) -> Any:
            if name not in names:
                raise S4BControlError(f"final archive member missing: {name}")
            return strict_json_loads(archive.read(name).decode("utf-8-sig"))

        status = member_json("final_status.json")
        if (
            status.get("schema_version") != 1
            or status.get("task_id") != TASK_ID
            or status.get("status") != "COMPLETE_PENDING_CODEX_REVIEW"
            or status.get("acceptance_status") != "PENDING_CODEX_REVIEW"
            or status.get("policy_sha256") != policy.sha256
            or not re_full_commit(status.get("implementation_commit"))
            or not isinstance(status.get("output_run_id"), str)
            or not status.get("output_run_id")
            or not re_full_sha(status.get("authorization_sha256"))
            or not re_full_sha(status.get("preflight_receipt_sha256"))
            or not isinstance(status.get("expected_data_store_dir"), str)
            or status.get("low_fraction_test_count") != 40
            or status.get("alias_100_count") != 10
            or status.get("unified_result_count") != 50
            or status.get("failed_run_ids") != []
            or status.get("alias_rerun_count") != 0
        ):
            raise S4BControlError("final archive status differs")
        for name in names:
            pure = Path(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise S4BControlError("final archive member path is unsafe")
        with tempfile.TemporaryDirectory(prefix="s4br1_verify_") as temporary:
            extracted = Path(temporary)
            archive.extractall(extracted)
            preflight_root = extracted / "preflight_snapshots"
            if sha256_file(preflight_root / "preflight_receipt.json") != status[
                "preflight_receipt_sha256"
            ]:
                raise S4BControlError("archived preflight receipt identity differs")
            _verify_validation_evidence(
                extracted / "validation_evidence",
                gate_path=extracted / "validation_gate.json",
                preflight_dir=preflight_root,
                policy=policy,
                policy_path=arguments.policy.resolve(),
                implementation_commit=status["implementation_commit"],
                authorization_sha256=status["authorization_sha256"],
                output_run_id=status["output_run_id"],
                expected_data_store_dir=status["expected_data_store_dir"],
                check_live_reference_paths=False,
            )
        sources = member_json("source_archives.json")
        if (
            sources.get("s3c3_outer_sha256") != S3C3_OUTER_SHA256
            or sources.get("s3c3_canonical_sha256") != S3C3_CANONICAL_SHA256
            or sources.get("frozen_test_sample_projection_sha256")
            != FROZEN_TEST_SAMPLE_PROJECTION_SHA256
            or sources.get("analysis_manifest", {}).get("status") != "PASS"
            or sources.get("analysis_manifest", {}).get("verified_alias_count") != 10
        ):
            raise S4BControlError("final archive frozen source identity differs")
        test_samples = member_json("preflight_snapshots/test_samples.json")
        validate_frozen_test_table(test_samples)
        low_results = []
        for asset in policy.assets:
            prefix = f"low_fraction_test/{asset.run_id}/"
            worker_samples = member_json(prefix + "expected_samples.json")
            worker_samples = validate_frozen_test_table(worker_samples)
            if identity_sha256(worker_samples) != identity_sha256(test_samples):
                raise S4BControlError("archived worker expected samples differ")
            rows = parse_jsonl(archive.read(prefix + "predictions.jsonl"))
            metrics = validate_prediction_rows(
                rows, test_samples, asset=asset, split="test"
            )
            validate_stored_metrics(member_json(prefix + "metrics.json"), metrics)
            restoration = member_json(prefix + "restoration.json")
            validate_restoration_evidence(
                restoration,
                asset=asset,
                policy=policy,
                implementation_commit=status["implementation_commit"],
                split="test",
                expected_data_store_dir=status["expected_data_store_dir"],
            )
            binding = member_json(prefix + "sample_binding.json")
            if (
                binding.get("status") != "PASS"
                or binding.get("source")
                != "SERVER_PREFLIGHT_INDEPENDENT_DATASTORE_TABLE"
                or binding.get("split") != "test"
                or binding.get("sample_identity_sha256")
                != FROZEN_TEST_SAMPLE_PROJECTION_SHA256
                or binding.get("sample_count") != 28
                or binding.get("prediction_row_count") != 84
            ):
                raise S4BControlError("archived low-fraction sample binding differs")
            low_results.append(
                {
                    "run_id": asset.run_id,
                    "method": asset.method,
                    "seed": asset.seed,
                    "fraction_percent": asset.fraction_percent,
                    "split": "test",
                    "status": "PASS",
                    "checkpoint_sha256": asset.checkpoint_sha256,
                    "policy_sha256": policy.sha256,
                    "implementation_commit": status["implementation_commit"],
                    "output_run_id": status["output_run_id"],
                    "authorization_sha256": status["authorization_sha256"],
                    "preflight_receipt_sha256": status["preflight_receipt_sha256"],
                    "prediction_sha256": sha256_bytes(
                        archive.read(prefix + "predictions.jsonl")
                    ),
                    "metrics": metrics,
                }
            )
        test_batch_status = member_json("batch_evidence/batch_status.json")
        if (
            test_batch_status.get("split") != "test"
            or test_batch_status.get("execution_status") != "FINISHED"
            or test_batch_status.get("completed_run_count") != 40
            or test_batch_status.get("failed_run_count") != 0
            or test_batch_status.get("not_executed_run_ids") != []
        ):
            raise S4BControlError("archived test batch status differs")
        test_batch_identity = member_json("batch_evidence/batch_identity.json")
        for name, expected_value in {
            "task_id": TASK_ID,
            "split": "test",
            "output_run_id": status["output_run_id"],
            "implementation_commit": status["implementation_commit"],
            "policy_sha256": policy.sha256,
            "authorization_sha256": status["authorization_sha256"],
            "preflight_receipt_sha256": status["preflight_receipt_sha256"],
        }.items():
            if test_batch_identity.get(name) != expected_value:
                raise S4BControlError(f"archived test batch identity differs: {name}")
        if member_json("batch_evidence/errors.json") != []:
            raise S4BControlError("archived test batch carries errors")
        if member_json("batch_evidence/completed_results.json") != low_results:
            raise S4BControlError("archived test completed results do not replay")
        expected_test_logs = {
            f"batch_evidence/logs/{run_id}.{suffix}"
            for run_id in S4B_RUN_IDS
            for suffix in ("out", "err", "json")
        }
        actual_test_logs = {
            name for name in names if name.startswith("batch_evidence/logs/")
        }
        if actual_test_logs != expected_test_logs:
            raise S4BControlError("archived test worker log coverage differs")
        for run_id in S4B_RUN_IDS:
            if member_json(f"batch_evidence/logs/{run_id}.json").get("exit_code") != 0:
                raise S4BControlError("archived test worker log reports a failure")
        alias_records = member_json("aliases_100.json")
        if not isinstance(alias_records, list) or len(alias_records) != 10:
            raise S4BControlError("archived 100% alias index differs")
        alias_identities = set()
        alias_metrics_by_identity = {}
        for row in alias_records:
            identity = (row.get("method"), row.get("seed"))
            if identity in alias_identities or identity not in {
                (method, seed) for method in ("B1", "RPT") for seed in range(42, 47)
            }:
                raise S4BControlError("archived 100% alias identity differs")
            alias_identities.add(identity)
            prediction_path = row.get("local_prediction_path")
            metrics_path = row.get("local_metrics_path")
            if (
                row.get("source") != "ALIAS_037_NO_RERUN"
                or row.get("rerun") is not False
                or row.get("source_outer_sha256") != S3C3_OUTER_SHA256
                or row.get("source_canonical_sha256") != S3C3_CANONICAL_SHA256
                or prediction_path not in names
                or metrics_path not in names
                or sha256_bytes(archive.read(prediction_path)) != row.get("prediction_sha256")
                or sha256_bytes(archive.read(metrics_path)) != row.get("metrics_sha256")
                or row.get("prediction_sha256") != ALIAS_PREDICTION_SHA256[identity]
                or row.get("metrics_sha256") != ALIAS_METRICS_SHA256[identity]
            ):
                raise S4BControlError("archived 100% alias byte identity differs")
            alias_rows = parse_jsonl(archive.read(prediction_path))
            alias_metrics = validate_legacy_alias_rows(
                alias_rows,
                test_samples,
                method=identity[0],
                seed=identity[1],
            )
            validate_stored_metrics(member_json(metrics_path), alias_metrics)
            alias_metrics_by_identity[(identity[0], identity[1], 100)] = alias_metrics
        unified = member_json("unified_50_test_metrics.json")
        if not isinstance(unified, list) or len(unified) != 50:
            raise S4BControlError("archived unified result count differs")
        identities = [
            (row.get("method"), row.get("seed"), row.get("fraction_percent"))
            for row in unified
        ]
        expected = {
            (method, seed, fraction)
            for method in ("B1", "RPT")
            for seed in range(42, 47)
            for fraction in (10, 25, 50, 75, 100)
        }
        if len(set(identities)) != 50 or set(identities) != expected:
            raise S4BControlError("archived unified identity matrix differs")
        low_metrics_by_identity = {
            (row["method"], row["seed"], row["fraction_percent"]): row["metrics"]
            for row in low_results
        }
        unified_by_identity = {identity: row for identity, row in zip(identities, unified)}
        for identity, metrics in low_metrics_by_identity.items():
            row = unified_by_identity[identity]
            if (
                row.get("source") != "S4B_NEW_LOW_FRACTION_TEST"
                or row.get("metrics") != metrics
            ):
                raise S4BControlError("archived unified low-fraction metrics differ")
        for identity, metrics in alias_metrics_by_identity.items():
            row = unified_by_identity[identity]
            if (
                row.get("source") != "ALIAS_037_NO_RERUN"
                or row.get("metrics") != metrics
                or row.get("source_canonical_sha256") != S3C3_CANONICAL_SHA256
            ):
                raise S4BControlError("archived unified alias metrics differ")
    receipt = {
        "status": "PASS",
        "archive_sha256": archive_sha,
        "policy_sha256": policy.sha256,
        "low_fraction_count": 40,
        "alias_count": 10,
        "unified_count": 50,
        "acceptance_status": "PENDING_CODEX_REVIEW",
    }
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


def re_full_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def re_full_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="S4B low-fraction strict inference/export control"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    common.add_argument("--implementation-commit", required=True)
    common.add_argument("--output-run-id", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", parents=[common], help="create a separate WSL or server receipt"
    )
    preflight_parser.add_argument("--side", choices=["wsl", "server"], required=True)
    preflight_parser.add_argument("--repository", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)
    preflight_parser.add_argument("--authorization", type=Path)
    preflight_parser.add_argument("--wsl-receipt", type=Path)
    preflight_parser.add_argument("--core-zip", type=Path)
    preflight_parser.add_argument("--datastore-dir", type=Path)
    preflight_parser.set_defaults(handler=preflight)

    run_parser = subparsers.add_parser(
        "run", parents=[common], help="run all forty validation or test workers"
    )
    run_parser.add_argument("--repository", type=Path, required=True)
    run_parser.add_argument("--authorization", type=Path, required=True)
    run_parser.add_argument("--preflight", type=Path, required=True)
    run_parser.add_argument("--datastore-dir", type=Path, required=True)
    run_parser.add_argument("--split", choices=["validation", "test"], required=True)
    run_parser.add_argument("--validation-gate", type=Path)
    run_parser.add_argument("--validation-dir", type=Path)
    run_parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(handler=run_batch)

    finalize_parser = subparsers.add_parser(
        "finalize", parents=[common], help="add verified 037 aliases and build fifty-run archive"
    )
    finalize_parser.add_argument("--authorization", type=Path, required=True)
    finalize_parser.add_argument("--repository", type=Path, required=True)
    finalize_parser.add_argument("--preflight", type=Path, required=True)
    finalize_parser.add_argument("--datastore-dir", type=Path, required=True)
    finalize_parser.add_argument("--validation-dir", type=Path, required=True)
    finalize_parser.add_argument("--validation-gate", type=Path, required=True)
    finalize_parser.add_argument("--test-run-dir", type=Path, required=True)
    finalize_parser.add_argument("--core-zip", type=Path, required=True)
    finalize_parser.add_argument(
        "--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS_MANIFEST
    )
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.set_defaults(handler=finalize)

    verify_parser = subparsers.add_parser(
        "verify", help="independently verify a finalized evidence archive"
    )
    verify_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.set_defaults(handler=verify_archive)

    selftest_parser = subparsers.add_parser(
        "selftest", help="run synthetic Human3 CPU decode/de-scale smoke only"
    )
    selftest_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    selftest_parser.set_defaults(
        handler=lambda arguments: (
            print(
                json.dumps(
                    _synthetic_selftest(arguments.policy.resolve()),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            or 0
        )
    )

    # Internal worker remains authorization- and preflight-bound even when
    # invoked manually; it is public in --help so the exact command is auditable.
    worker_parser = subparsers.add_parser("worker", parents=[common])
    worker_parser.add_argument("--repository", type=Path, required=True)
    worker_parser.add_argument("--authorization", type=Path, required=True)
    worker_parser.add_argument("--preflight", type=Path, required=True)
    worker_parser.add_argument("--datastore-dir", type=Path, required=True)
    worker_parser.add_argument("--split", choices=["validation", "test"], required=True)
    worker_parser.add_argument("--validation-dir", type=Path)
    worker_parser.add_argument("--validation-gate", type=Path)
    worker_parser.add_argument("--asset", choices=list(S4B_RUN_IDS), required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.set_defaults(handler=worker)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.command == "preflight" and arguments.side == "server":
        if arguments.core_zip is None or arguments.datastore_dir is None:
            parser.error("server preflight requires --core-zip and --datastore-dir")
    try:
        return int(arguments.handler(arguments))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
