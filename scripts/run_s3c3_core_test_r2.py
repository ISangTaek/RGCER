"""S3C3-R2 controlled inference export with historical validation reuse.

This is a versioned successor to the immutable 035 evidence script.  It runs
the nine still-required validation exports through the explicit inference
compatibility view, unlocks all fifteen core test exports only after the full
validation gate, and independently verifies the immutable 035/034 ZIPs before
copying their accepted members.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Importing by absolute script path from an evidence/output directory must use
# repository code, not depend on the caller's current working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from checkpoint_inference_compatibility import (  # noqa: E402
    PRODUCTION_POLICY,
    PRODUCTION_POLICY_SHA256,
    prepare_inference_checkpoint,
    sha256_file,
)
from s3c3_r2_control import (  # noqa: E402
    BASELINE_TEST_ZIP_SHA256,
    CORE_RUN_IDS,
    FrozenTestSampleBinding,
    HISTORICAL_VALIDATION_ZIP_SHA256,
    HUMAN3_TASKS,
    NEW_VALIDATION_RUN_IDS,
    plan_after_validation,
    strict_json_loads,
    validate_core_rows,
    verify_baseline_tests,
    verify_frozen_test_sample_binding,
    verify_historical_validations,
    write_verified_reuse,
)


BASE_ASSET_COMMIT = "55eb8b187642beebd95d5b188f3955031e5b6d54"
LOCK_SHA256 = "d6ce984ead87b6d54c11f43d2b45b333e5afa8b90465626a09954c69af3f9a23"
TASK_ID = "S3C3_CORE_AND_BASELINES_TEST_20260913_R2"
SERVER_ROOT = Path("/home/shangzeli/RGCER")
SERVER_PYTHON = SERVER_ROOT / ".tmp/baseline_envs/S3A3_20260909T044302Z/bin/python"
SERVER_DATASTORE = SERVER_ROOT / "data/toxacute_datastore_v2/builds/toxacute-v2-7b7bd62a6457"
WSL_ROOT = Path("/home/soengzaak/PROJECT/RGCER")
WSL_PYTHON = Path("/home/soengzaak/miniconda3/envs/icl/bin/python")


def read_json(path: str | Path):
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def load_lock(path: str | Path):
    path = Path(path).resolve()
    if sha256_file(path) != LOCK_SHA256:
        raise ValueError("S3C3 lock bytes changed")
    lock = read_json(path)
    if lock.get("commit") != BASE_ASSET_COMMIT:
        raise ValueError("S3C3 lock base asset commit changed")
    if lock.get("allowed_splits") != ["validation", "test"]:
        raise ValueError("S3C3 lock split authorization changed")
    assets = lock.get("assets")
    if not isinstance(assets, list) or len(assets) != 15:
        raise ValueError("S3C3 lock must contain exactly 15 core assets")
    by_run = {asset.get("run_id"): asset for asset in assets}
    if set(by_run) != set(CORE_RUN_IDS):
        raise ValueError("S3C3 lock core run coverage changed")
    for run_id, locked in by_run.items():
        policy = PRODUCTION_POLICY.asset(run_id)
        checkpoint = locked.get("best_checkpoint", {})
        if (
            locked.get("method") != policy.method
            or type(locked.get("seed")) is not int
            or locked["seed"] != policy.seed
            or locked.get("best_epoch") != policy.best_epoch
            or checkpoint.get("sha256") != policy.checkpoint_sha256
            or checkpoint.get("size_bytes") != policy.checkpoint_size_bytes
        ):
            raise ValueError(f"S3C3 lock differs from compatibility policy: {run_id}")
    if lock.get("baseline_zip_sha256") != BASELINE_TEST_ZIP_SHA256:
        raise ValueError("S3C3 lock baseline ZIP identity changed")
    return lock


def _rows_and_metrics(
    output: Path,
    asset,
    split: str,
    *,
    frozen_test_binding: FrozenTestSampleBinding | None = None,
):
    rows = [
        strict_json_loads(line)
        for line in (output / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected = read_json(output / "expected_samples.json")
    frozen_verification = None
    validation_samples = expected
    if split == "test":
        if frozen_test_binding is None:
            raise ValueError(
                "core test requires the independently verified 034 sample binding"
            )
        frozen_verification = verify_frozen_test_sample_binding(
            expected,
            frozen_test_binding,
        )
        validation_samples = frozen_test_binding.as_json_samples()
    calculated = validate_core_rows(
        rows,
        validation_samples,
        method=asset["method"],
        seed=asset["seed"],
        split=split,
    )
    if calculated != read_json(output / "metrics.json"):
        raise ValueError("stored core metrics differ from independent calculation")
    return rows, expected, calculated, frozen_verification


def _verified_worker_result(
    output: Path,
    asset,
    split: str,
    *,
    frozen_test_binding: FrozenTestSampleBinding | None = None,
):
    _, _, metrics, frozen_verification = _rows_and_metrics(
        output,
        asset,
        split,
        frozen_test_binding=frozen_test_binding,
    )
    result = {
        "run_id": asset["run_id"],
        "method": asset["method"],
        "seed": asset["seed"],
        "split": split,
        "metrics": metrics,
    }
    if split == "test":
        if frozen_verification is None:
            raise AssertionError("test sample verification was not produced")
        verification_name = "frozen_test_sample_verification.json"
        write_json(output / verification_name, frozen_verification)
        result["frozen_test_sample_verification"] = frozen_verification
        result["frozen_test_sample_verification_file"] = (
            f"{output.name}/{verification_name}"
        )
    return result


def _core_test_binding_summary(
    core_tests,
    frozen_test_binding: FrozenTestSampleBinding,
):
    by_run = {}
    expected_verification = {
        "status": "VERIFIED",
        "source_archive_sha256": frozen_test_binding.source_archive_sha256,
        "frozen_test_sample_identity_sha256": (
            frozen_test_binding.sample_identity_sha256
        ),
        "actual_test_sample_identity_sha256": (
            frozen_test_binding.sample_identity_sha256
        ),
        "sample_count": len(frozen_test_binding.samples),
        "exact_ordered_match": True,
    }
    for row in core_tests:
        run_id = row.get("run_id")
        expected_verification_file = (
            f"test_{run_id}/frozen_test_sample_verification.json"
        )
        if (
            run_id not in CORE_RUN_IDS
            or run_id in by_run
            or row.get("split") != "test"
            or row.get("frozen_test_sample_verification") != expected_verification
            or row.get("frozen_test_sample_verification_file")
            != expected_verification_file
        ):
            raise ValueError("core test frozen sample verification coverage differs")
        by_run[run_id] = row
    if set(by_run) != set(CORE_RUN_IDS):
        raise ValueError("core test frozen sample verification coverage is not 15")
    return {
        "status": "VERIFIED",
        "source_archive_sha256": frozen_test_binding.source_archive_sha256,
        "frozen_test_sample_identity_sha256": (
            frozen_test_binding.sample_identity_sha256
        ),
        "sample_count": len(frozen_test_binding.samples),
        "verified_run_count": len(by_run),
        "runs": [
            {
                "run_id": run_id,
                "status": "VERIFIED",
                "verification_file": by_run[run_id][
                    "frozen_test_sample_verification_file"
                ],
            }
            for run_id in CORE_RUN_IDS
        ],
    }


def _worker(arguments, lock) -> None:
    if Path.cwd().resolve() != SERVER_ROOT.resolve():
        raise RuntimeError("core worker must run from the server repository")
    import torch
    import main as app
    from toxacute_datastore import SPLIT_CODES

    if not torch.cuda.is_available():
        raise RuntimeError("core worker requires the assigned GPU")
    asset = next(item for item in lock["assets"] if item["run_id"] == arguments.asset)
    checkpoint_path = Path(asset["best_checkpoint"]["path"])
    prepared = prepare_inference_checkpoint(
        checkpoint_path,
        run_id=asset["run_id"],
        method=asset["method"],
        seed=asset["seed"],
        operation="batch_inference",
        split=arguments.split,
        expected_policy_sha256=PRODUCTION_POLICY_SHA256,
    )
    payload = prepared.payload
    configuration = payload["configuration"]
    params = app.build_parser().parse_args([])
    vars(params).update(copy.deepcopy(configuration))
    overrides = {
        "mode": "batch_inference",
        "inference_split": arguments.split,
        "load_path": str(checkpoint_path),
        "save_path": None,
        "data_store_dir": str(SERVER_DATASTORE),
        "preprocessed_data_dir": None,
        "gpu_id": "0",
        "num_loader_workers": 0,
        "fit_conformal": False,
    }
    vars(params).update(overrides)
    app.validate_params(params)
    app.seed_everything(params.seed)
    task_names = app.task_names_for_params(params)
    if task_names != payload["task_names"] or tuple(task_names) != HUMAN3_TASKS:
        raise ValueError("checkpoint/runtime Human3 task order differs")
    params.effective_rgcer_config = app.resolve_effective_rgcer_config(params)
    store = app._resolve_data_store(params, task_names, required=True)

    expected_samples = []
    graph_indices = []
    for index, sample_id in enumerate(store.sample_ids):
        if int(store.split_codes[index]) != SPLIT_CODES[arguments.split]:
            continue
        labels = [float(store.labels[index, store.task_index(task)]) for task in HUMAN3_TASKS]
        if not any(math.isfinite(label) for label in labels):
            continue
        if int(store.num_nodes[index]) > params.max_nodes_filter:
            raise ValueError("frozen node filter would drop an evaluated sample")
        graph_indices.append(index)
        expected_samples.append(
            {
                "sample_id": str(sample_id),
                "row_index": int(store.row_indices[index]),
                "labels": [label if math.isfinite(label) else None for label in labels],
            }
        )
    ordered = sorted(
        zip(expected_samples, graph_indices),
        key=lambda item: (item[0]["row_index"], item[0]["sample_id"]),
    )
    expected_samples = [item[0] for item in ordered]
    graph_indices = [item[1] for item in ordered]

    output = arguments.output.resolve()
    output.mkdir(exist_ok=False)
    write_json(output / "expected_samples.json", expected_samples)
    architecture_kwargs, optimizer_parameters = app.prepare_args(params)
    encoder, architecture, decoders = app._build_model_components(
        params, task_names, app._device_from_params(params)
    )
    trainer = app.Trainer(
        task_dict=app.build_task_dict(params, task_names),
        weighting=app.weighting_method.__dict__[params.weighting],
        architecture=architecture,
        encoder_class=encoder,
        decoders=decoders,
        optim_param=optimizer_parameters,
        args=params,
        save_path=None,
        load_path=str(checkpoint_path),
        inference_compatibility=prepared.compatibility,
        **architecture_kwargs,
    )
    if trainer.loaded_epoch != asset["best_epoch"]:
        raise ValueError("loaded checkpoint epoch differs")
    for name, tensor in trainer.model.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), payload["model_state"][name]):
            raise ValueError(f"loaded model tensor differs: {name}")

    before_model_sha = app.state_dict_sha256(trainer.model)
    checkpoint_sha_before = sha256_file(checkpoint_path)
    trainer.model.eval()
    collator = app.DataCollator(
        spatial_pos_max_clip=params.spatial_pos_clip, max_node_filter=None
    )
    rows = []
    for offset in range(0, len(graph_indices), params.bs):
        indices = graph_indices[offset : offset + params.bs]
        batch = collator([store.get_graph_data(index) for index in indices]).to(trainer.device)
        if batch.get("is_empty", False):
            raise ValueError("collator unexpectedly produced an empty batch")
        expected_ids = [expected_samples[offset + index]["sample_id"] for index in range(len(indices))]
        if list(batch.sample_id) != expected_ids:
            raise ValueError("collated sample order differs")
        with torch.no_grad():
            predictions = trainer.predict_all_tasks(batch)
            decoded = {
                task: trainer.decode_task_output(
                    task, predictions[task], apply_conformal=False
                )["median"]
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
                for task in HUMAN3_TASKS
            }
        for local_index in range(len(indices)):
            sample = expected_samples[offset + local_index]
            for task_index, task in enumerate(HUMAN3_TASKS):
                rows.append(
                    {
                        "method": asset["method"],
                        "seed": asset["seed"],
                        "sample_id": sample["sample_id"],
                        "row_index": sample["row_index"],
                        "split": arguments.split,
                        "endpoint": task,
                        "y_true": sample["labels"][task_index],
                        "y_pred": float(decoded[task][local_index]),
                        "mask": sample["labels"][task_index] is not None,
                    }
                )
    metrics = validate_core_rows(
        rows,
        expected_samples,
        method=asset["method"],
        seed=asset["seed"],
        split=arguments.split,
    )

    if arguments.split == "validation":
        reference = Path(asset["run_dir"]) / "diagnostics/best_validation_human3_predictions.csv"
        if not reference.is_file():
            raise FileNotFoundError("historical validation reference is missing")
        with reference.open(encoding="utf-8-sig") as handle:
            historical_rows = list(csv.DictReader(handle))

        def field(row, names):
            return next(row[name] for name in names if name in row and row[name] != "")

        lookup = {}
        for row in historical_rows:
            key = (field(row, ["sample_id", "id"]), field(row, ["endpoint", "task"]))
            if key in lookup:
                raise ValueError("historical validation reference contains duplicates")
            lookup[key] = (
                float(field(row, ["y_true", "label", "target"])),
                float(field(row, ["y_pred", "final_prediction", "prediction"])),
            )
        observed = [row for row in rows if row["mask"]]
        if set(lookup) != {(row["sample_id"], row["endpoint"]) for row in observed}:
            raise ValueError("historical validation coverage differs")
        max_abs_difference = 0.0
        for row in observed:
            target, prediction = lookup[(row["sample_id"], row["endpoint"])]
            if not math.isclose(target, row["y_true"], abs_tol=1e-6, rel_tol=1e-7):
                raise ValueError("historical validation target differs")
            if not math.isclose(prediction, row["y_pred"], abs_tol=1e-6, rel_tol=1e-7):
                raise ValueError("historical validation prediction differs")
            max_abs_difference = max(max_abs_difference, abs(prediction - row["y_pred"]))
        shutil.copy2(reference, output / "original_validation.csv")
        write_json(
            output / "validation_comparison.json",
            {
                "status": "PASS",
                "reference_path": str(reference),
                "reference_sha256": sha256_file(reference),
                "max_abs_difference": max_abs_difference,
            },
        )

    after_model_sha = app.state_dict_sha256(trainer.model)
    checkpoint_sha_after = sha256_file(checkpoint_path)
    if before_model_sha != after_model_sha:
        raise ValueError("model tensors changed during inference")
    if checkpoint_sha_before != checkpoint_sha_after or checkpoint_sha_after != asset["best_checkpoint"]["sha256"]:
        raise ValueError("checkpoint bytes changed during inference")
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "metrics.json", metrics)
    code_identity = {
        "implementation_git_head": _git_head(REPOSITORY_ROOT),
        "runner_sha256": sha256_file(Path(__file__)),
        "compatibility_module_sha256": sha256_file(
            REPOSITORY_ROOT / "checkpoint_inference_compatibility.py"
        ),
        "trainer_sha256": sha256_file(REPOSITORY_ROOT / "trainer.py"),
        "lock_sha256": LOCK_SHA256,
    }
    audit = prepared.compatibility.audit_record(code_identity=code_identity)
    audit["checkpoint"]["sha256_before"] = checkpoint_sha_before
    audit["checkpoint"]["sha256_after"] = checkpoint_sha_after
    audit["model_state_sha256_before"] = before_model_sha
    audit["model_state_sha256_after"] = after_model_sha
    write_json(output / "compatibility_audit.json", audit)
    write_json(
        output / "restoration.json",
        {
            "asset": asset,
            "checkpoint_version": 6,
            "loaded_epoch": trainer.loaded_epoch,
            "loaded_routing_enabled": trainer.loaded_routing_enabled,
            "model_tensor_identity": before_model_sha,
            "model_unchanged": True,
            "checkpoint_unchanged": True,
            "init_overlay_applied": False,
            "training_called": False,
            "calibration_called": False,
            "compatibility_policy_version": prepared.compatibility.policy_version,
            "compatibility_policy_sha256": prepared.compatibility.policy_sha256,
            "compatibility_migration_applied": prepared.compatibility.migration_applied,
            "overrides": overrides,
        },
    )
    store.close()
    print("CORE_EXPORT_COMPLETE", asset["run_id"], arguments.split)


def _selftest() -> dict:
    samples = [
        {"sample_id": "a", "row_index": 1, "labels": [1.0, 2.0, 3.0]},
        {"sample_id": "b", "row_index": 2, "labels": [2.0, 3.0, 4.0]},
    ]
    rows = [
        {
            "method": "B0",
            "seed": 42,
            "sample_id": sample["sample_id"],
            "row_index": sample["row_index"],
            "split": "validation",
            "endpoint": task,
            "y_true": sample["labels"][task_index],
            "y_pred": sample["labels"][task_index] + 1.0,
            "mask": True,
        }
        for sample in samples
        for task_index, task in enumerate(HUMAN3_TASKS)
    ]
    metrics = validate_core_rows(
        rows, samples, method="B0", seed=42, split="validation"
    )
    if metrics["human3_macro_rmse"] != 1.0:
        raise AssertionError("independent metric selftest failed")
    gate = plan_after_validation(
        historical_validation_verified=True,
        new_validation_results={run_id: True for run_id in NEW_VALIDATION_RUN_IDS},
    )
    if len(gate["core_tests_to_schedule"]) != 15:
        raise AssertionError("two-stage gate selftest failed")
    return {
        "status": "PASS",
        "synthetic_only": True,
        "repository_root": str(REPOSITORY_ROOT),
        "policy_sha256": PRODUCTION_POLICY_SHA256,
    }


def _pack(output: Path) -> None:
    if not (output / "batch_status.json").is_file():
        raise FileNotFoundError("batch_status.json is missing")
    files = []
    for path in output.rglob("*"):
        if any(part in {"__pycache__", "pytest_tmp", ".pytest_cache"} for part in path.parts):
            continue
        if path.is_symlink():
            raise ValueError("evidence package contains a symlink")
        if path.is_file():
            if path.suffix.lower() in {".zip", ".pt", ".pth", ".joblib", ".pkl", ".lmdb"}:
                raise ValueError(f"evidence package contains a prohibited asset: {path}")
            files.append(path)
    checksums = {path.relative_to(output).as_posix(): sha256_file(path) for path in sorted(files)}
    checksum_path = output / "checksums.sha256"
    with checksum_path.open("x", encoding="utf-8") as handle:
        for name, digest in checksums.items():
            handle.write(f"{digest}  {name}\n")
    target = output.parent / f"{output.name}.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [checksum_path]:
            archive.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("packed ZIP CRC check failed")
        for name, digest in checksums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"packed ZIP member checksum differs: {name}")
    sidecar = Path(str(target) + ".sha256")
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{sha256_file(target)}  {target.name}\n")
    print(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["wsl", "server", "worker", "pack", "selftest"], required=True
    )
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wsl-evidence", type=Path)
    parser.add_argument("--historical-validation-zip", type=Path)
    parser.add_argument("--baseline-zip", type=Path)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--asset")
    parser.add_argument("--split", choices=["validation", "test"])
    arguments = parser.parse_args()
    if arguments.mode == "selftest":
        print(json.dumps(_selftest(), ensure_ascii=False, allow_nan=False))
        return 0
    if arguments.mode == "pack":
        _pack(arguments.output.resolve())
        return 0
    if arguments.lock is None:
        parser.error("--lock is required")
    lock = load_lock(arguments.lock)
    if arguments.mode == "worker":
        if arguments.asset not in CORE_RUN_IDS or arguments.output is None:
            parser.error("worker requires an authorized --asset and --output")
        _worker(arguments, lock)
        return 0
    if (
        type(arguments.implementation_commit) is not str
        or len(arguments.implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in arguments.implementation_commit)
    ):
        parser.error("wsl/server require the review-approved 40-hex --implementation-commit")
    if arguments.output is None:
        parser.error("wsl/server require --output")

    server = arguments.mode == "server"
    repository = SERVER_ROOT if server else WSL_ROOT
    expected_python = SERVER_PYTHON if server else WSL_PYTHON
    if Path(sys.executable).absolute() != expected_python:
        raise RuntimeError("wrong controlled Python interpreter")
    output = arguments.output.resolve()
    expected_output_parent = (
        Path("/home/shangzeli/baseline_runs")
        if server
        else Path("/home/soengzaak/EXP_LOGs/RGCER")
    )
    if output.parent != expected_output_parent or not output.name.startswith("S3C3R2_"):
        raise ValueError("output must be a new S3C3R2_* directory under the controlled evidence root")
    output.mkdir(exist_ok=False)
    (output / "logs").mkdir()
    write_json(
        output / "execution_identity.json",
        {
            "task_id": TASK_ID,
            "mode": arguments.mode,
            "repository": str(repository),
            "python": sys.executable,
            "hostname": socket.gethostname(),
            "started": time.time(),
            "base_asset_commit": BASE_ASSET_COMMIT,
            "implementation_commit": arguments.implementation_commit,
            "lock_sha256": LOCK_SHA256,
            "compatibility_policy_sha256": PRODUCTION_POLICY_SHA256,
        },
    )
    source_files = (
        Path(__file__),
        REPOSITORY_ROOT / "checkpoint_inference_compatibility.py",
        REPOSITORY_ROOT / "s3c3_r2_control.py",
        REPOSITORY_ROOT / "trainer.py",
    )
    for source in source_files:
        shutil.copy2(source, output / source.name)
    shutil.copy2(arguments.lock, output / "s3c3_core_test_lock.json")

    def command(name, command_args, *, gpu=None, cwd=None):
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command_args = [str(value) for value in command_args]
        started = time.time()
        with (output / "logs" / f"{name}.out").open("xb") as stdout_handle, (
            output / "logs" / f"{name}.err"
        ).open("xb") as stderr_handle:
            result = subprocess.run(
                command_args,
                cwd=cwd or repository,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        write_json(
            output / "logs" / f"{name}.json",
            {
                "command": command_args,
                "cwd": str(cwd or repository),
                "started": started,
                "ended": time.time(),
                "exit_code": result.returncode,
                "cuda_visible_devices": gpu,
            },
        )
        if result.returncode != 0:
            raise RuntimeError(f"command failed: {name} exit={result.returncode}")
        return (output / "logs" / f"{name}.out").read_text(encoding="utf-8")

    completed = []
    errors = []
    core_test_binding_summary = None
    try:
        if command("status", ["git", "status", "--porcelain"]).strip():
            raise RuntimeError("controlled repository is not clean")
        command("fetch", ["git", "fetch", "origin", "master"])
        if command("remote", ["git", "rev-parse", "origin/master"]).strip() != arguments.implementation_commit:
            raise RuntimeError("origin/master differs from the approved implementation commit")
        command("merge", ["git", "merge", "--ff-only", arguments.implementation_commit])
        if command("head", ["git", "rev-parse", "HEAD"]).strip() != arguments.implementation_commit:
            raise RuntimeError("HEAD differs from the approved implementation commit")
        command("selftest", [sys.executable, Path(__file__), "--mode", "selftest"])
        command(
            "import_path_regression",
            [sys.executable, Path(__file__).resolve(), "--mode", "selftest"],
            cwd=output,
        )
        test_files = [
            "tests/test_s3c3_inference_compatibility.py",
            "tests/test_s3c3_r2_control.py",
            "tests/test_checkpoint_ablation_config.py",
            "tests/test_checkpoint_v6_contract.py",
        ]
        command(
            "focused_tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *test_files,
                "--basetemp",
                output / "pytest_tmp",
                "--junitxml=" + str(output / "pytest.xml"),
            ],
        )
        xml = ET.parse(output / "pytest.xml")
        if not xml.findall(".//testcase") or xml.findall(".//failure") or xml.findall(".//error") or xml.findall(".//skipped"):
            raise RuntimeError("focused JUnit contains failure/error/skip or no tests")

        if server:
            if arguments.wsl_evidence is None:
                raise ValueError("server mode requires --wsl-evidence")
            wsl_evidence = arguments.wsl_evidence.resolve()
            wsl_status = read_json(wsl_evidence / "batch_status.json")
            wsl_identity = read_json(wsl_evidence / "execution_identity.json")
            server_identity = read_json(output / "execution_identity.json")
            if (
                wsl_status.get("execution_status") != "FINISHED"
                or wsl_status.get("mode") != "wsl"
                or wsl_status.get("task_id") != TASK_ID
                or wsl_identity.get("implementation_commit")
                != arguments.implementation_commit
                or wsl_status.get("finished", float("inf"))
                > server_identity["started"]
                or (wsl_evidence / "logs/head.out").read_text(encoding="utf-8").strip()
                != arguments.implementation_commit
            ):
                raise ValueError("WSL preflight evidence is not an exact completed R2 run")
            for source in source_files:
                if sha256_file(wsl_evidence / source.name) != sha256_file(source):
                    raise ValueError(f"WSL/server code bytes differ: {source.name}")
            if sha256_file(wsl_evidence / "s3c3_core_test_lock.json") != LOCK_SHA256:
                raise ValueError("WSL/server lock bytes differ")
            shutil.copytree(
                wsl_evidence,
                output / "wsl_preflight",
                ignore=shutil.ignore_patterns("__pycache__", "pytest_tmp", ".pytest_cache"),
            )
            if arguments.historical_validation_zip is None or arguments.baseline_zip is None:
                raise ValueError("server mode requires both frozen reuse ZIP paths")
            historical = verify_historical_validations(
                arguments.historical_validation_zip,
                expected_sha256=HISTORICAL_VALIDATION_ZIP_SHA256,
            )
            baseline = verify_baseline_tests(
                arguments.baseline_zip,
                expected_sha256=BASELINE_TEST_ZIP_SHA256,
            )
            frozen_test_binding = baseline.frozen_test_sample_binding
            if frozen_test_binding is None:
                raise AssertionError("verified 034 archive did not provide a test binding")
            write_verified_reuse(historical, output / "validation_reused_035")
            write_verified_reuse(baseline, output / "baseline_reused_034")
            write_json(
                output / "historical_validation_reuse.json",
                {
                    "source_path": historical.source_path,
                    "sha256": historical.archive_sha256,
                    "runs": [list(identity) for identity in historical.verified_identities],
                    "member_sha256": {
                        name: hashlib.sha256(data).hexdigest()
                        for name, data in historical.selected_members.items()
                    },
                    "retrained": False,
                },
            )
            write_json(
                output / "baseline_reuse.json",
                {
                    "source_path": baseline.source_path,
                    "sha256": baseline.archive_sha256,
                    "runs": [list(identity) for identity in baseline.verified_identities],
                    "member_sha256": {
                        name: hashlib.sha256(data).hexdigest()
                        for name, data in baseline.selected_members.items()
                    },
                    "frozen_test_sample_identity_sha256": (
                        frozen_test_binding.sample_identity_sha256
                    ),
                    "frozen_test_sample_count": len(frozen_test_binding.samples),
                    "retrained": False,
                },
            )
            by_run = {asset["run_id"]: asset for asset in lock["assets"]}
            for asset in lock["assets"]:
                checkpoint = Path(asset["best_checkpoint"]["path"])
                if (
                    checkpoint.stat().st_size != asset["best_checkpoint"]["size_bytes"]
                    or sha256_file(checkpoint) != asset["best_checkpoint"]["sha256"]
                ):
                    raise ValueError(f"core checkpoint identity differs: {asset['run_id']}")

            def execute(run_id, split, gpu_index):
                gpu = subprocess.run(
                    ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=uuid", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                )
                if gpu.returncode not in {0, 14} or not gpu.stdout.strip().startswith("GPU-"):
                    raise RuntimeError(f"GPU {gpu_index} UUID binding failed")
                gpu_uuid = gpu.stdout.strip()
                occupancy = subprocess.run(
                    ["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid", "--format=csv"],
                    capture_output=True,
                    text=True,
                )
                if occupancy.returncode not in {0, 14} or [
                    line.strip() for line in occupancy.stdout.splitlines() if line.strip()
                ] != ["pid"]:
                    raise RuntimeError(f"GPU {gpu_index} is occupied or unverifiable")
                key = f"{split}_{run_id}"
                write_json(
                    output / "logs" / f"{key}.gpu.json",
                    {
                        "physical_gpu": gpu_index,
                        "uuid": gpu_uuid,
                        "occupancy": occupancy.stdout,
                    },
                )
                run_output = output / key
                command(
                    key,
                    [
                        "timeout",
                        "--kill-after=60",
                        "1800",
                        sys.executable,
                        Path(__file__).resolve(),
                        "--mode",
                        "worker",
                        "--lock",
                        arguments.lock.resolve(),
                        "--asset",
                        run_id,
                        "--split",
                        split,
                        "--output",
                        run_output,
                    ],
                    gpu=gpu_uuid,
                )
                return _verified_worker_result(
                    run_output,
                    by_run[run_id],
                    split,
                    frozen_test_binding=(
                        frozen_test_binding if split == "test" else None
                    ),
                )

            validation_results = {}
            for offset in range(0, len(NEW_VALIDATION_RUN_IDS), 4):
                wave = NEW_VALIDATION_RUN_IDS[offset : offset + 4]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    jobs = {
                        pool.submit(execute, run_id, "validation", gpu_index): run_id
                        for gpu_index, run_id in enumerate(wave)
                    }
                    for job in as_completed(jobs):
                        run_id = jobs[job]
                        try:
                            row = job.result()
                            completed.append(row)
                            validation_results[run_id] = True
                        except Exception as exc:
                            validation_results[run_id] = False
                            errors.append(
                                {
                                    "run_id": run_id,
                                    "split": "validation",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                if errors:
                    break
            gate = plan_after_validation(
                historical_validation_verified=True,
                new_validation_results=validation_results,
            )
            write_json(output / "validation_gate.json", gate)
            if not gate["validation_gate_passed"]:
                raise RuntimeError("validation gate failed; zero core tests were scheduled")

            for offset in range(0, len(gate["core_tests_to_schedule"]), 4):
                wave = gate["core_tests_to_schedule"][offset : offset + 4]
                with ThreadPoolExecutor(max_workers=4) as pool:
                    jobs = {
                        pool.submit(execute, run_id, "test", gpu_index): run_id
                        for gpu_index, run_id in enumerate(wave)
                    }
                    for job in as_completed(jobs):
                        run_id = jobs[job]
                        try:
                            completed.append(job.result())
                        except Exception as exc:
                            errors.append(
                                {
                                    "run_id": run_id,
                                    "split": "test",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                if errors:
                    break
            if errors:
                raise RuntimeError("core test export stopped after the first failed wave")
            core_tests = [row for row in completed if row["split"] == "test"]
            if len(core_tests) != 15:
                raise RuntimeError("core test export count is not 15")
            core_test_binding_summary = _core_test_binding_summary(
                core_tests,
                frozen_test_binding,
            )
            write_json(
                output / "core_test_sample_binding.json",
                core_test_binding_summary,
            )

            unified = []
            baseline_root = output / "baseline_reused_034"
            for manifest_path in sorted(baseline_root.rglob("inference_manifest.json")):
                manifest = read_json(manifest_path)
                metrics = read_json(manifest_path.with_name("metrics.json"))
                unified.append(
                    {
                        "method": manifest["method"],
                        "seed": manifest["seed"],
                        "source": "REUSE_034",
                        "path": str(manifest_path.with_name("metrics.json").relative_to(output)),
                        "per_endpoint": metrics["per_endpoint"],
                        "human3_macro_rmse": metrics["human3_macro_rmse"],
                    }
                )
            for row in core_tests:
                unified.append(
                    {
                        "method": row["method"],
                        "seed": row["seed"],
                        "source": "NEW_CORE_TEST_R2",
                        "path": f"test_{row['run_id']}/metrics.json",
                        **row["metrics"],
                    }
                )
            if len(unified) != 40 or len(
                {(row["method"], row["seed"]) for row in unified}
            ) != 40:
                raise RuntimeError("unified 40-run identity coverage differs")
            if any(
                [row["per_endpoint"][task]["n"] for task in HUMAN3_TASKS]
                != [14, 13, 13]
                for row in unified
            ):
                raise RuntimeError("unified test endpoint counts differ")
            write_json(output / "unified_40_test_metrics.json", unified)

        status = {
            "task_id": TASK_ID,
            "mode": arguments.mode,
            "execution_status": "FINISHED",
            "content_status": "PASS" if server else "NOT_CHECKED_WSL_PREFLIGHT_ONLY",
            "acceptance_status": "PENDING_CODEX_REVIEW",
            "new_validation_exports": sum(
                row["split"] == "validation" for row in completed
            ),
            "historical_validation_reused": 6 if server else 0,
            "core_test_exports": sum(row["split"] == "test" for row in completed),
            "baseline_test_reused": 25 if server else 0,
            "finished": time.time(),
        }
        if server:
            if core_test_binding_summary is None:
                raise AssertionError("core test sample binding summary was not produced")
            status["core_test_sample_binding"] = {
                "status": core_test_binding_summary["status"],
                "source_archive_sha256": core_test_binding_summary[
                    "source_archive_sha256"
                ],
                "frozen_test_sample_identity_sha256": core_test_binding_summary[
                    "frozen_test_sample_identity_sha256"
                ],
                "verified_run_count": core_test_binding_summary[
                    "verified_run_count"
                ],
                "record": "core_test_sample_binding.json",
            }
    except Exception as exc:
        errors.append({"error": f"{type(exc).__name__}: {exc}"})
        status = {
            "task_id": TASK_ID,
            "mode": arguments.mode,
            "execution_status": "PARTIAL_OR_FAILED",
            "content_status": "FAIL_OR_NOT_CHECKED",
            "acceptance_status": "PENDING_CODEX_REVIEW",
            "new_validation_exports": sum(
                row["split"] == "validation" for row in completed
            ),
            "core_test_exports": sum(row["split"] == "test" for row in completed),
            "finished": time.time(),
        }
    write_json(output / "summary.json", completed)
    write_json(output / "errors.json", errors)
    write_json(output / "batch_status.json", status)
    print(json.dumps(status, ensure_ascii=False, allow_nan=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
