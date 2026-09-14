"""S4B strict-control regression tests.

All generated data are synthetic.  The two accepted ZIPs are opened only as
already-reviewed metadata/result fixtures; no checkpoint, DataStore, real test
CSV, calibration data, training path, CUDA path, server, or network is used.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import checkpoint_inference_compatibility as core_compatibility
import s4_low_fraction_control as control
from run_diagnostics import RunDiagnosticsWriter
from scripts import run_s4_low_fraction_export as runner


ROOT = Path(__file__).resolve().parents[1]
POLICY_ATTACHMENT = ROOT / "Plans&Results/Plans/s4b_low_fraction_policy.json"
POLICY_PATH = ROOT / "configs/s4b_low_fraction_policy.json"
DATASTORE_METADATA_FIXTURE = (
    ROOT / "tests/fixtures/s4c_datastore_metadata_toxacute_v2_7b7bd62a6457.json"
)
S4A_FIXTURE = ROOT / "Plans&Results/Results/038_review_20260914/canonical.zip"
S3C3_FIXTURE = (
    ROOT / "Plans&Results/Results/037_2026-09-14_S3C3R2核心恢复与test闭环.zip"
)
ANALYSIS_MANIFEST = (
    ROOT
    / "Plans&Results/Results/2026-09-14_S3D_Figure2统计与图稿/analysis_manifest.json"
)
IMPLEMENTATION_COMMIT = "a" * 40
OUTPUT_RUN_ID = "S4B_SYNTHETIC_FIXTURE_001"
AUTHORIZATION_SHA = "1" * 64
PREFLIGHT_SHA = "2" * 64
VALIDATION_LOG_SHA = "3" * 64
LOCAL_FIXTURES = os.environ.get("S4B_LOCAL_FIXTURES") == "1"


def gate_kwargs():
    return {
        "output_run_id": OUTPUT_RUN_ID,
        "authorization_sha256": AUTHORIZATION_SHA,
        "preflight_receipt_sha256": PREFLIGHT_SHA,
        "validation_log_identity_sha256": VALIDATION_LOG_SHA,
    }


@pytest.fixture(scope="session")
def policy():
    return control.load_s4b_policy(POLICY_PATH)


@pytest.fixture(scope="session")
def aliases():
    if not S3C3_FIXTURE.is_file():
        pytest.skip("accepted 037 archive is not installed in this workspace")
    return control.verify_037_aliases(S3C3_FIXTURE)


def first_metadata(policy):
    if not S4A_FIXTURE.is_file():
        pytest.skip("accepted 038 metadata fixture is not installed in this workspace")
    asset = policy.assets[0]
    with zipfile.ZipFile(S4A_FIXTURE) as archive:
        metadata = control.strict_json_loads(
            archive.read(asset.provenance["metadata_member"]).decode("utf-8-sig")
        )
    return asset, metadata


def full_datastore_metadata():
    return copy.deepcopy(control.read_json(DATASTORE_METADATA_FIXTURE))


def datastore_source_identity(policy):
    identity = control.validate_datastore_metadata(
        full_datastore_metadata(), policy.assets[0]
    )
    identity.update(
        {
            "root": "synthetic-datastore",
            "datastore_json_sha256": "4" * 64,
        }
    )
    return identity


def synthetic_samples():
    return [
        {"sample_id": "s0", "row_index": 10, "labels": [0.0, 1.0, 2.0]},
        {"sample_id": "s1", "row_index": 11, "labels": [1.0, 2.0, 3.0]},
        {"sample_id": "s2", "row_index": 12, "labels": [2.0, None, 4.0]},
    ]


def prediction_rows(asset, samples, *, split="validation", delta=0.25):
    return [
        {
            "run_id": asset.run_id,
            "method": asset.method,
            "seed": asset.seed,
            "fraction_percent": asset.fraction_percent,
            "sample_id": sample["sample_id"],
            "row_index": sample["row_index"],
            "split": split,
            "endpoint": task,
            "y_true": sample["labels"][task_index],
            "y_pred": (
                float(sample["labels"][task_index] + delta)
                if sample["labels"][task_index] is not None
                else float(delta)
            ),
            "mask": sample["labels"][task_index] is not None,
        }
        for sample in samples
        for task_index, task in enumerate(control.HUMAN3_TASKS)
    ]


def synthetic_payload(asset):
    return {
        "checkpoint_version": 6,
        "epoch": asset.best_epoch,
        "configuration": {
            **copy.deepcopy(dict(asset.expected_configuration)),
            **copy.deepcopy(dict(asset.expected_adaptation)),
        },
        "architecture_config": copy.deepcopy(dict(asset.expected_architecture)),
        "task_names": list(control.HUMAN3_TASKS),
        "prediction_mode": "quantile",
        "quantile_config": {"lower": 0.05, "upper": 0.95},
        "task_scalers": copy.deepcopy(dict(asset.expected_scalers)),
        "data_config": copy.deepcopy(dict(asset.expected_data_config)),
        "feature_schema_version": asset.expected_data_config["feature_schema_version"],
        "routing_enabled": True,
        "selection_state": {},
        "reproducibility": copy.deepcopy(dict(asset.expected_reproducibility)),
        "split_manifest_hash": asset.expected_data_config["split_manifest_hash"],
        "rgcer_config": {},
        "effective_rgcer_config": {},
        "model_state": {
            name: torch.zeros(tuple(signature["shape"]), dtype=torch.float32)
            for name, signature in asset.expected_model_tensor_shapes.items()
        },
    }


def fixture_authorization(policy, *, fixture_only=True):
    return {
        "schema_version": 1,
        "authorization_id": "S4B_SYNTHETIC_TEST_AUTH",
        "status": (
            "SYNTHETIC_FIXTURE_ONLY"
            if fixture_only
            else "READY_FOR_S4B_SERVER_EXECUTION"
        ),
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "policy_sha256": policy.sha256,
        "authorized_run_ids": list(control.S4B_RUN_IDS),
        "allowed_splits": ["validation", "test"],
        "output_run_id": OUTPUT_RUN_ID,
        "fixture_only": fixture_only,
    }


def validation_result(asset, policy, *, status="PASS"):
    evidence_files = {
        name: hashlib.sha256(f"{asset.run_id}:{name}".encode()).hexdigest()
        for name in control.VALIDATION_EVIDENCE_FILES
    }
    return {
        "run_id": asset.run_id,
        "method": asset.method,
        "seed": asset.seed,
        "fraction_percent": asset.fraction_percent,
        "split": "validation",
        "status": status,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "policy_sha256": policy.sha256,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "output_run_id": OUTPUT_RUN_ID,
        "authorization_sha256": AUTHORIZATION_SHA,
        "preflight_receipt_sha256": PREFLIGHT_SHA,
        "prediction_sha256": hashlib.sha256(asset.run_id.encode()).hexdigest(),
        "metrics": {"synthetic": True},
        "reference_level": "FROZEN_MACRO_ONLY",
        "historical_csv_exists": False,
        "evidence_files": evidence_files,
        "evidence_identity_sha256": control.identity_sha256(evidence_files),
    }


def low_fraction_test_result(asset, policy):
    return {
        "run_id": asset.run_id,
        "method": asset.method,
        "seed": asset.seed,
        "fraction_percent": asset.fraction_percent,
        "split": "test",
        "status": "PASS",
        "checkpoint_sha256": asset.checkpoint_sha256,
        "policy_sha256": policy.sha256,
        "prediction_sha256": hashlib.sha256(asset.run_id.encode()).hexdigest(),
        "metrics": {
            "per_endpoint": {
                task: {
                    "n": 2,
                    "rmse": 0.5,
                    "mae": 0.5,
                    "r2": 0.0,
                    "r2_reason": None,
                }
                for task in control.HUMAN3_TASKS
            },
            "human3_macro_rmse": 0.5,
        },
    }


def valid_restoration(asset, policy, *, split="validation", datastore="synthetic-datastore"):
    state_sha = hashlib.sha256(asset.run_id.encode()).hexdigest()
    return {
        "schema_version": 1,
        "run_id": asset.run_id,
        "method": asset.method,
        "seed": asset.seed,
        "fraction_percent": asset.fraction_percent,
        "policy_sha256": policy.sha256,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "checkpoint_path": asset.checkpoint_path,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "checkpoint_size_bytes": asset.checkpoint_size_bytes,
        "loaded_epoch": asset.best_epoch,
        "checkpoint_version": 6,
        "scaler_identity_sha256": control.identity_sha256(asset.expected_scalers),
        "task_scalers": copy.deepcopy(dict(asset.expected_scalers)),
        "effective_configuration": {
            **copy.deepcopy(dict(asset.expected_configuration)),
            **copy.deepcopy(dict(asset.expected_adaptation)),
        },
        "architecture_config": copy.deepcopy(dict(asset.expected_architecture)),
        "runtime_overrides": {
            "mode": control.ALLOWED_OPERATION,
            "inference_split": split,
            "load_path": asset.checkpoint_path,
            "save_path": None,
            "data_store_dir": datastore,
            "preprocessed_data_dir": None,
            "gpu_id": "0",
            "num_loader_workers": 0,
            "fit_conformal": False,
        },
        "runtime_override_names": list(control.RUNTIME_OVERRIDE_NAMES),
        "model_state_sha_algorithm": control.MODEL_STATE_SHA_ALGORITHM,
        "original_model_state_sha256": state_sha,
        "model_state_sha256_before": state_sha,
        "model_state_sha256_after": state_sha,
        "checkpoint_sha256_before": asset.checkpoint_sha256,
        "checkpoint_sha256_after": asset.checkpoint_sha256,
        "migration_applied": False,
        "architecture_additions": [],
        "init_overlay_applied": False,
        "training_called": False,
        "calibration_called": False,
        "model_unchanged": True,
        "checkpoint_unchanged": True,
        "routing_enabled": True,
    }


def valid_wsl_receipt(policy):
    return {
        "schema_version": 1,
        "task_id": control.S4B_TASK_ID,
        "receipt_id": "S4B_WSL_SYNTHETIC",
        "receipt_kind": "S4B_WSL_PREFLIGHT",
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "output_run_id": OUTPUT_RUN_ID,
        "policy_sha256": policy.sha256,
        "core_policy_sha256": control.CORE_POLICY_SHA256,
        "runner_sha256": "1" * 64,
        "control_sha256": "2" * 64,
        "trainer_sha256": "3" * 64,
        "test_status": "PASS",
        "asset_accessed": False,
        "real_data_accessed": False,
        "training_called": False,
    }


def formatted_json_bytes(value, style):
    if style == "trailing_lf":
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if style == "no_trailing_lf":
        return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    if style == "crlf":
        text = json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\r\n")
        return (text + "\r\n").encode("utf-8")
    if style == "reordered_compact":
        reordered = dict(reversed(list(value.items())))
        return json.dumps(
            reordered,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    raise AssertionError(f"unknown JSON byte style: {style}")


def install_server_preflight_fakes(monkeypatch, tmp_path, policy, aliases):
    from toxacute_datastore import SPLIT_CODES

    validation_samples = synthetic_samples()
    test_samples = list(aliases.frozen_samples)
    all_samples = [*validation_samples, *test_samples]
    labels = torch.full((len(all_samples), 59), float("nan"), dtype=torch.float64)
    for row_number, sample in enumerate(all_samples):
        for task_offset, label in enumerate(sample["labels"]):
            if label is not None:
                labels[row_number, 35 + task_offset] = float(label)

    store_root = tmp_path / "synthetic_datastore"
    store_root.mkdir()
    (store_root / "datastore.json").write_bytes(b"synthetic datastore identity")

    class SyntheticStore:
        metadata = full_datastore_metadata()
        task_names = metadata["task_names"]
        sample_ids = [sample["sample_id"] for sample in all_samples]
        row_indices = [sample["row_index"] for sample in all_samples]
        split_codes = [
            *([SPLIT_CODES["validation"]] * len(validation_samples)),
            *([SPLIT_CODES["test"]] * len(test_samples)),
        ]
        num_nodes = [1] * len(all_samples)
        root = store_root

        def task_index(self, task):
            return self.task_names.index(task)

        def close(self):
            return None

    store = SyntheticStore()
    store.labels = labels

    def repository_check(repository, implementation_commit):
        assert Path(repository).resolve() == ROOT.resolve()
        assert implementation_commit == IMPLEMENTATION_COMMIT

    def params_and_store(*args, **kwargs):
        del args, kwargs
        return (
            object(),
            SimpleNamespace(
                max_nodes_filter=policy.assets[0].expected_data_config[
                    "max_nodes_filter"
                ]
            ),
            store,
        )

    def prepare_checkpoint(*args, policy, run_id, **kwargs):
        del args, kwargs
        asset = policy.asset(run_id)
        return SimpleNamespace(
            checkpoint_sha256=asset.checkpoint_sha256,
            checkpoint_size_bytes=asset.checkpoint_size_bytes,
            strict_identity={"migration_applied": False, "model_tensor_count": 170},
        )

    monkeypatch.setattr(runner, "_validate_repository", repository_check)
    monkeypatch.setattr(runner, "verify_037_aliases", lambda path: aliases)
    monkeypatch.setattr(runner, "_params_and_store", params_and_store)
    monkeypatch.setattr(runner, "prepare_low_fraction_checkpoint", prepare_checkpoint)
    return store_root


def make_server_preflight_inputs(
    tmp_path,
    monkeypatch,
    policy,
    aliases,
    *,
    authorization_style="trailing_lf",
    wsl_style="crlf",
):
    datastore_dir = install_server_preflight_fakes(
        monkeypatch, tmp_path, policy, aliases
    )
    authorization = fixture_authorization(policy, fixture_only=False)
    authorization_bytes = formatted_json_bytes(authorization, authorization_style)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_bytes(authorization_bytes)

    code = runner._code_identity(ROOT, POLICY_PATH)  # noqa: SLF001
    wsl = valid_wsl_receipt(policy)
    wsl.update(
        {
            "runner_sha256": code["runner_sha256"],
            "control_sha256": code["control_sha256"],
            "trainer_sha256": code["trainer_sha256"],
        }
    )
    wsl_bytes = formatted_json_bytes(wsl, wsl_style)
    wsl_path = tmp_path / "wsl_receipt.json"
    wsl_path.write_bytes(wsl_bytes)

    output = tmp_path / "server_preflight"
    arguments = Namespace(
        side="server",
        repository=ROOT,
        policy=POLICY_PATH,
        implementation_commit=IMPLEMENTATION_COMMIT,
        output_run_id=OUTPUT_RUN_ID,
        authorization=authorization_path,
        wsl_receipt=wsl_path,
        core_zip=tmp_path / "mocked_037.zip",
        datastore_dir=datastore_dir,
        output=output,
    )
    return arguments, authorization_bytes, wsl_bytes


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_frozen_policy_copy_is_byte_identical():
    assert POLICY_ATTACHMENT.read_bytes() == POLICY_PATH.read_bytes()
    assert control.sha256_file(POLICY_PATH) == control.S4B_POLICY_SHA256


def test_policy_has_exact_matrix_and_no_migration(policy):
    assert len(policy.assets) == 40
    assert {asset.identity for asset in policy.assets} == control.S4B_MATRIX
    assert [asset.run_id for asset in policy.assets] == list(control.S4B_RUN_IDS)
    assert all(asset.raw["migration_required"] is False for asset in policy.assets)
    assert all(len(asset.expected_model_tensor_shapes) == 170 for asset in policy.assets)
    assert all(
        asset.expected_adaptation["feature_drift_epochs"] == "0,5,10,15,19"
        for asset in policy.assets
    )


def test_policy_rejects_any_byte_change(tmp_path):
    changed = bytearray(POLICY_PATH.read_bytes())
    changed[-2] = ord(" ") if changed[-2] != ord(" ") else ord("\t")
    path = tmp_path / "changed_policy.json"
    path.write_bytes(changed)
    with pytest.raises(control.S4BControlError, match="policy bytes"):
        control.load_s4b_policy(path)


def test_original_core_policy_identity_and_allowlist_unchanged():
    assert core_compatibility.PRODUCTION_POLICY_SHA256 == control.CORE_POLICY_SHA256
    assert core_compatibility.PRODUCTION_POLICY.computed_sha256() == control.CORE_POLICY_SHA256
    assert len(core_compatibility.PRODUCTION_POLICY.assets) == 15
    assert not set(control.S4B_RUN_IDS) & {
        asset.run_id for asset in core_compatibility.PRODUCTION_POLICY.assets
    }
    control.assert_core_policy_unchanged()


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_actual_038_metadata_fixture_all_forty(policy):
    if not S4A_FIXTURE.is_file():
        pytest.skip("accepted 038 metadata fixture is not installed")
    receipt = control.verify_s4a_metadata_archive(S4A_FIXTURE, policy)
    assert receipt == {
        "status": "PASS",
        "fixture_kind": "REAL_038_METADATA_NOT_REAL_CHECKPOINT_LOAD",
        "canonical_sha256": control.S4A_CANONICAL_SHA256,
        "verified_asset_count": 40,
        "migration_applied_count": 0,
        "architecture_addition_count": 0,
        "tensor_signatures_per_asset": 170,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda metadata: metadata.update(sha256="0" * 64), "sha256"),
        (
            lambda metadata: metadata["fields"]["epoch"].update(value=True),
            "epoch",
        ),
        (
            lambda metadata: metadata["fields"]["configuration"]["value"].update(seed=43),
            "seed",
        ),
        (
            lambda metadata: metadata["fields"]["configuration"]["value"].update(train_fraction=0.25),
            "train_fraction",
        ),
        (
            lambda metadata: metadata["fields"]["task_scalers"]["value"]["man_oral_TDLo"].update(std=True),
            "scaler",
        ),
        (
            lambda metadata: metadata["fields"]["architecture_config"]["value"].pop("d6_candidate"),
            "d6_candidate",
        ),
        (
            lambda metadata: metadata["fields"]["architecture_config"]["value"].update(d6_candidate="none"),
            "d6_candidate",
        ),
        (
            lambda metadata: metadata["fields"]["split_manifest_sha256"].update(present=True, value="0" * 64),
            "split_manifest_sha256",
        ),
        (
            lambda metadata: metadata["model_tensor_shapes"].pop(next(iter(metadata["model_tensor_shapes"]))),
            "tensor",
        ),
    ],
)
def test_metadata_negative_cases_fail_closed(policy, mutation, match):
    asset, metadata = first_metadata(policy)
    mutation(metadata)
    with pytest.raises(control.S4BControlError, match=match):
        control.validate_s4a_metadata(metadata, asset)


def test_strict_payload_positive_has_no_architecture_addition(policy):
    asset = policy.assets[0]
    result = control._validate_payload_identity(  # noqa: SLF001 - isolated contract test
        synthetic_payload(asset), asset, torch_module=torch
    )
    assert result["migration_applied"] is False
    assert result["architecture_additions"] == []
    assert result["model_tensor_count"] == 170
    assert result["task_scalers"] == asset.expected_scalers


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(epoch=True), "epoch"),
        (
            lambda payload: payload["configuration"].update(train_fraction=0.25),
            "train_fraction",
        ),
        (
            lambda payload: payload["configuration"].update(seed=True),
            "seed",
        ),
        (
            lambda payload: payload["task_scalers"]["man_oral_TDLo"].update(mean=True),
            "scaler",
        ),
        (
            lambda payload: payload["configuration"].pop("d6_candidate"),
            "d6_candidate",
        ),
        (
            lambda payload: payload["architecture_config"].update(d6_candidate="none"),
            "d6_candidate",
        ),
        (
            lambda payload: payload.update(split_manifest_sha256="0" * 64),
            "split_manifest_sha256",
        ),
        (
            lambda payload: payload["configuration"].update(
                datastore_context={"unsupported_type": "synthetic.Placeholder"}
            ),
            "unsupported_type",
        ),
        (
            lambda payload: payload["model_state"].pop(next(iter(payload["model_state"]))),
            "tensor name",
        ),
    ],
)
def test_payload_negative_cases_fail_closed(policy, mutation, match):
    asset = policy.assets[0]
    payload = synthetic_payload(asset)
    mutation(payload)
    with pytest.raises(control.S4BControlError, match=match):
        control._validate_payload_identity(  # noqa: SLF001
            payload, asset, torch_module=torch
        )


def test_checkpoint_file_size_then_sha_validation(tmp_path):
    path = tmp_path / "synthetic.pt"
    path.write_bytes(b"not a checkpoint and never unpickled")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert control.verify_file_identity(
        path, expected_size_bytes=path.stat().st_size, expected_sha256=digest
    ) == {"size_bytes": path.stat().st_size, "sha256": digest}
    with pytest.raises(control.S4BControlError, match="size"):
        control.verify_file_identity(
            path, expected_size_bytes=path.stat().st_size + 1, expected_sha256=digest
        )
    with pytest.raises(control.S4BControlError, match="SHA"):
        control.verify_file_identity(
            path, expected_size_bytes=path.stat().st_size, expected_sha256="0" * 64
        )


def test_datastore_metadata_binds_full_catalogue_to_human3_selection(policy):
    asset = policy.assets[0]
    metadata = full_datastore_metadata()
    result = control.validate_datastore_metadata(metadata, asset)
    assert result["datastore_fingerprint"] == asset.expected_data_config[
        "datastore_fingerprint"
    ]
    assert len(result["datastore_task_names"]) == 59
    assert result["requested_task_names"] == list(control.HUMAN3_TASKS)
    assert result["selected_task_indices"] == [35, 36, 37]
    assert result["num_tasks"] == 59
    assert result["labels_shape"] == [79721, 59]
    assert control.sha256_file(POLICY_PATH) == control.S4B_POLICY_SHA256


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing_human3", "missing requested Human3"),
        ("duplicate_task", "duplicates"),
        ("non_string_task", "non-empty strings"),
        ("wrong_num_tasks", "num_tasks"),
        ("wrong_label_columns", "labels_shape columns"),
        ("reordered_full_catalogue", "ordered task catalogue"),
        ("wrong_build", "build_id"),
        ("wrong_fingerprint", "datastore_fingerprint"),
        ("wrong_split", "splitting"),
        ("collapsed_to_human3", "ordered task catalogue"),
        ("bool_format_version", "format_version"),
    ],
)
def test_datastore_metadata_catalogue_negative_cases_fail_closed(policy, case, match):
    metadata = full_datastore_metadata()
    if case == "missing_human3":
        metadata["task_names"].remove(control.HUMAN3_TASKS[1])
    elif case == "duplicate_task":
        metadata["task_names"][1] = metadata["task_names"][0]
    elif case == "non_string_task":
        metadata["task_names"][0] = 7
    elif case == "wrong_num_tasks":
        metadata["num_tasks"] = 58
    elif case == "wrong_label_columns":
        metadata["labels_shape"][1] = 58
    elif case == "reordered_full_catalogue":
        metadata["task_names"][0], metadata["task_names"][1] = (
            metadata["task_names"][1],
            metadata["task_names"][0],
        )
    elif case == "wrong_build":
        metadata["build_id"] = "wrong-build"
    elif case == "wrong_fingerprint":
        metadata["datastore_fingerprint"] = "0" * 64
    elif case == "wrong_split":
        metadata["splitting"] = "random"
    elif case == "collapsed_to_human3":
        metadata["task_names"] = list(control.HUMAN3_TASKS)
        metadata["num_tasks"] = 3
        metadata["labels_shape"][1] = 3
    elif case == "bool_format_version":
        metadata["format_version"] = True
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)
    with pytest.raises(control.S4BControlError, match=match):
        control.validate_datastore_metadata(metadata, policy.assets[0])


def test_datastore_task_selection_rejects_recorded_wrong_indices():
    tasks = full_datastore_metadata()["task_names"]
    with pytest.raises(control.S4BControlError, match="name-resolved"):
        control.validate_datastore_task_selection(
            tasks, list(control.HUMAN3_TASKS), [0, 1, 2]
        )


def test_offline_preflight_replay_rejects_self_consistent_wrong_task_mapping(policy):
    source = datastore_source_identity(policy)
    source["selected_task_indices"] = [0, 1, 2]
    with pytest.raises(control.S4BControlError, match="task selection"):
        runner._validate_recorded_datastore_source(source, policy)  # noqa: SLF001


def test_sample_table_uses_human3_columns_from_full_catalogue(policy):
    from toxacute_datastore import SPLIT_CODES

    metadata = full_datastore_metadata()
    identity = control.validate_datastore_metadata(metadata, policy.assets[0])

    class SyntheticFullStore:
        task_names = list(metadata["task_names"])
        sample_ids = ["sample-12", "sample-10", "sentinel-only"]
        row_indices = [12, 10, 11]
        split_codes = [SPLIT_CODES["validation"]] * 3
        num_nodes = [8, 9, 10]
        labels = torch.full((3, 59), float("nan"), dtype=torch.float64)

        def task_index(self, task_name):
            return self.task_names.index(task_name)

    store = SyntheticFullStore()
    store.labels[:, 0:3] = torch.tensor(
        [[9001.0, 9002.0, 9003.0]] * 3, dtype=torch.float64
    )
    store.labels[0, 35:38] = torch.tensor(
        [1.5, float("nan"), 3.5], dtype=torch.float64
    )
    store.labels[1, 35:38] = torch.tensor(
        [4.5, 5.5, float("nan")], dtype=torch.float64
    )

    samples, graph_indices = runner._sample_table_from_store(  # noqa: SLF001
        store,
        "validation",
        max_nodes_filter=512,
        selected_task_indices=identity["selected_task_indices"],
    )
    assert graph_indices == [1, 0]
    assert samples == [
        {
            "sample_id": "sample-10",
            "row_index": 10,
            "labels": [4.5, 5.5, None],
        },
        {
            "sample_id": "sample-12",
            "row_index": 12,
            "labels": [1.5, None, 3.5],
        },
    ]
    assert all(
        value is None or value < 9000
        for sample in samples
        for value in sample["labels"]
    )


def test_checkpoint_task_scope_remains_strict_human3(policy):
    asset = policy.assets[0]
    payload = synthetic_payload(asset)
    assert control._validate_payload_identity(  # noqa: SLF001
        payload, asset, torch_module=torch
    )["task_scalers"] == asset.expected_scalers

    full_catalogue = full_datastore_metadata()["task_names"]
    payload["task_names"] = list(full_catalogue)
    payload["architecture_config"]["task_names"] = list(full_catalogue)
    payload["data_config"]["task_names"] = list(full_catalogue)
    with pytest.raises(control.S4BControlError, match="task_names"):
        control._validate_payload_identity(  # noqa: SLF001
            payload, asset, torch_module=torch
        )


@pytest.mark.parametrize("operation", ["train", "resume", "calibration", "single_inference"])
def test_forbidden_operations_are_rejected(policy, operation):
    asset = policy.assets[0]
    with pytest.raises(control.S4BControlError, match="batch_inference"):
        control.validate_request(
            policy,
            run_id=asset.run_id,
            method=asset.method,
            seed=asset.seed,
            fraction_percent=asset.fraction_percent,
            operation=operation,
            split="validation",
            expected_policy_sha256=policy.sha256,
        )


def test_wrong_request_identity_and_unknown_run_are_rejected(policy):
    asset = policy.assets[0]
    with pytest.raises(control.S4BControlError, match="seed/fraction"):
        control.validate_request(
            policy,
            run_id=asset.run_id,
            method=asset.method,
            seed=True,
            fraction_percent=asset.fraction_percent,
            operation="batch_inference",
            split="validation",
            expected_policy_sha256=policy.sha256,
        )
    with pytest.raises(control.S4BControlError, match="not uniquely authorized"):
        control.validate_request(
            policy,
            run_id="unknown",
            method="B1",
            seed=42,
            fraction_percent=10,
            operation="batch_inference",
            split="validation",
            expected_policy_sha256=policy.sha256,
        )


def test_fixture_authorization_is_explicit_and_not_formal(policy):
    authorization = fixture_authorization(policy)
    accepted = control.validate_authorization(
        authorization,
        policy=policy,
        implementation_commit=IMPLEMENTATION_COMMIT,
        split="validation",
        output_run_id=OUTPUT_RUN_ID,
        allow_fixture=True,
    )
    assert accepted["status"] == "SYNTHETIC_FIXTURE_ONLY"
    with pytest.raises(control.S4BControlError, match="cannot open formal"):
        control.validate_authorization(
            authorization,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            split="validation",
            output_run_id=OUTPUT_RUN_ID,
            allow_fixture=False,
        )


def test_missing_or_mismatched_authorization_is_rejected(policy, tmp_path):
    with pytest.raises(control.S4BControlError, match="requires a separate"):
        control.load_authorization(
            None,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            split="validation",
            output_run_id=OUTPUT_RUN_ID,
        )
    authorization = fixture_authorization(policy)
    authorization["policy_sha256"] = "0" * 64
    with pytest.raises(control.S4BControlError, match="policy_sha256"):
        control.validate_authorization(
            authorization,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            split="validation",
            output_run_id=OUTPUT_RUN_ID,
            allow_fixture=True,
        )


@pytest.mark.parametrize(
    "style",
    ["trailing_lf", "no_trailing_lf", "crlf", "reordered_compact"],
)
def test_authorization_document_reads_one_exact_byte_version(policy, tmp_path, style):
    authorization = fixture_authorization(policy, fixture_only=False)
    source_bytes = formatted_json_bytes(authorization, style)
    path = tmp_path / f"authorization_{style}.json"
    path.write_bytes(source_bytes)

    loaded, loaded_bytes, digest = control.load_authorization_document(
        path,
        policy=policy,
        implementation_commit=IMPLEMENTATION_COMMIT,
        split="validation",
        output_run_id=OUTPUT_RUN_ID,
        allow_fixture=False,
    )
    compatible_loaded, compatible_digest = control.load_authorization(
        path,
        policy=policy,
        implementation_commit=IMPLEMENTATION_COMMIT,
        split="validation",
        output_run_id=OUTPUT_RUN_ID,
        allow_fixture=False,
    )
    assert loaded == authorization == compatible_loaded
    assert loaded_bytes == source_bytes
    assert digest == compatible_digest == hashlib.sha256(source_bytes).hexdigest()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("empty", "authorization"),
        ("commit", "implementation_commit"),
        ("run_id", "output_run_id"),
        ("assets", "authorized_run_ids"),
        ("scope", "does not permit split=validation"),
    ],
)
def test_authorization_document_counterexamples_fail_closed(
    policy, tmp_path, case, match
):
    authorization = fixture_authorization(policy, fixture_only=False)
    if case == "empty":
        authorization = {}
    elif case == "commit":
        authorization["implementation_commit"] = "b" * 40
    elif case == "run_id":
        authorization["output_run_id"] = "wrong-run"
    elif case == "assets":
        authorization["authorized_run_ids"] = authorization["authorized_run_ids"][:-1]
    else:
        authorization["allowed_splits"] = ["test"]
    path = tmp_path / f"bad_{case}.json"
    path.write_bytes(formatted_json_bytes(authorization, "trailing_lf"))
    with pytest.raises(control.S4BControlError, match=match):
        control.load_authorization_document(
            path,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            split="validation",
            output_run_id=OUTPUT_RUN_ID,
            allow_fixture=False,
        )


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
@pytest.mark.parametrize(
    "style",
    ["trailing_lf", "no_trailing_lf", "crlf", "reordered_compact"],
)
def test_production_preflight_preserves_authorization_snapshot_bytes_and_run_replays(
    policy, aliases, tmp_path, monkeypatch, style
):
    arguments, authorization_bytes, wsl_bytes = make_server_preflight_inputs(
        tmp_path,
        monkeypatch,
        policy,
        aliases,
        authorization_style=style,
    )
    assert runner.preflight(arguments) == 0
    authorization_snapshot = arguments.output / "authorization_snapshot.json"
    wsl_snapshot = arguments.output / "wsl_receipt_snapshot.json"
    assert authorization_snapshot.read_bytes() == authorization_bytes
    assert wsl_snapshot.read_bytes() == wsl_bytes
    authorization_sha = hashlib.sha256(authorization_bytes).hexdigest()
    receipt = runner._verify_server_preflight(  # noqa: SLF001
        arguments.output,
        implementation_commit=IMPLEMENTATION_COMMIT,
        policy_path=POLICY_PATH,
        authorization_sha256=authorization_sha,
        output_run_id=OUTPUT_RUN_ID,
        expected_data_store_dir=str(arguments.datastore_dir.resolve()),
        check_live_reference_paths=False,
    )
    assert receipt["status"] == "PASS"
    assert receipt["authorization_sha256"] == authorization_sha
    assert receipt["wsl_receipt_sha256"] == hashlib.sha256(wsl_bytes).hexdigest()

    if style == "trailing_lf":
        dispatches = []

        def dispatch_sentinel(namespace, *, asset, gpu_index, output):
            del namespace, gpu_index, output
            dispatches.append(asset.run_id)
            raise RuntimeError("EXPECTED_VALIDATION_DISPATCH_SENTINEL")

        def accept_mocked_live_references(references, policy, *, check_live_paths):
            assert len(references) == len(policy.assets) == 40
            assert check_live_paths is True

        monkeypatch.setattr(runner, "_run_worker_process", dispatch_sentinel)
        monkeypatch.setattr(
            runner, "_verify_reference_sources", accept_mocked_live_references
        )
        run_arguments = Namespace(
            repository=ROOT,
            policy=POLICY_PATH,
            implementation_commit=IMPLEMENTATION_COMMIT,
            authorization=arguments.authorization,
            output_run_id=OUTPUT_RUN_ID,
            preflight=arguments.output,
            datastore_dir=arguments.datastore_dir,
            split="validation",
            gpus=["0"],
            output=tmp_path / "validation_dispatch",
            validation_dir=None,
            validation_gate=None,
        )
        with pytest.raises(control.S4BControlError, match="log coverage"):
            runner.run_batch(run_arguments)
        assert dispatches == [policy.assets[0].run_id]
        assert control.read_json(run_arguments.output / "errors.json")[0][
            "error"
        ].endswith("EXPECTED_VALIDATION_DISPATCH_SENTINEL")


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_run_rejects_tampered_preflight_authorization_snapshot_before_dispatch(
    policy, aliases, tmp_path, monkeypatch
):
    arguments, authorization_bytes, _ = make_server_preflight_inputs(
        tmp_path, monkeypatch, policy, aliases
    )
    assert runner.preflight(arguments) == 0
    snapshot = arguments.output / "authorization_snapshot.json"
    snapshot.write_bytes(snapshot.read_bytes() + b" ")
    touched = []
    monkeypatch.setattr(
        runner,
        "_run_worker_process",
        lambda *args, **kwargs: touched.append("dispatch"),
    )
    with pytest.raises(control.S4BControlError, match="authorization snapshot"):
        runner.run_batch(
            Namespace(
                repository=ROOT,
                policy=POLICY_PATH,
                implementation_commit=IMPLEMENTATION_COMMIT,
                authorization=arguments.authorization,
                output_run_id=OUTPUT_RUN_ID,
                preflight=arguments.output,
                datastore_dir=arguments.datastore_dir,
                split="validation",
                gpus=["0"],
                output=tmp_path / "must_not_dispatch",
                validation_dir=None,
                validation_gate=None,
            )
        )
    assert control.sha256_file(arguments.authorization) == hashlib.sha256(
        authorization_bytes
    ).hexdigest()
    assert touched == []


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
@pytest.mark.parametrize("changed_input", ["authorization", "wsl_receipt"])
def test_preflight_rejects_input_replaced_after_validation_without_pass_receipt(
    policy, aliases, tmp_path, monkeypatch, changed_input
):
    arguments, _, _ = make_server_preflight_inputs(
        tmp_path, monkeypatch, policy, aliases
    )
    original_build = runner._build_reference_sources  # noqa: SLF001

    def replace_after_validation(current_policy):
        path = getattr(arguments, changed_input)
        path.write_bytes(path.read_bytes() + b" ")
        return original_build(current_policy)

    monkeypatch.setattr(runner, "_build_reference_sources", replace_after_validation)
    with pytest.raises(control.S4BControlError, match="input changed after validation"):
        runner.preflight(arguments)
    assert not (arguments.output / "preflight_receipt.json").exists()


def test_wsl_receipt_mismatch_is_rejected(policy):
    receipt = valid_wsl_receipt(policy)
    assert control.verify_wsl_receipt(
        receipt,
        implementation_commit=IMPLEMENTATION_COMMIT,
        policy_sha256=policy.sha256,
        runner_sha256="1" * 64,
        control_sha256="2" * 64,
        trainer_sha256="3" * 64,
        output_run_id=OUTPUT_RUN_ID,
    )["receipt_id"] == "S4B_WSL_SYNTHETIC"
    wrong_task = copy.deepcopy(receipt)
    wrong_task["task_id"] = "OTHER_TASK"
    with pytest.raises(control.S4BControlError, match="task_id"):
        control.verify_wsl_receipt(
            wrong_task,
            implementation_commit=IMPLEMENTATION_COMMIT,
            policy_sha256=policy.sha256,
            runner_sha256="1" * 64,
            control_sha256="2" * 64,
            trainer_sha256="3" * 64,
            output_run_id=OUTPUT_RUN_ID,
        )
    receipt["control_sha256"] = "4" * 64
    with pytest.raises(control.S4BControlError, match="control_sha256"):
        control.verify_wsl_receipt(
            receipt,
            implementation_commit=IMPLEMENTATION_COMMIT,
            policy_sha256=policy.sha256,
            runner_sha256="1" * 64,
            control_sha256="2" * 64,
            trainer_sha256="3" * 64,
            output_run_id=OUTPUT_RUN_ID,
        )


def test_independent_metrics_and_macro_are_recomputed(policy):
    asset = policy.assets[0]
    samples = synthetic_samples()
    rows = prediction_rows(asset, samples, delta=0.5)
    metrics = control.validate_prediction_rows(
        rows, samples, asset=asset, split="validation"
    )
    assert metrics["human3_macro_rmse"] == pytest.approx(0.5)
    assert [metrics["per_endpoint"][task]["n"] for task in control.HUMAN3_TASKS] == [3, 2, 3]
    stored = copy.deepcopy(metrics)
    stored["human3_macro_rmse"] = 0.0
    with pytest.raises(control.S4BControlError, match="macro"):
        control.validate_stored_metrics(stored, metrics)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].update(sample_id="wrong"),
        lambda rows: rows[0].update(row_index=999),
        lambda rows: rows[0].update(y_true=999.0),
        lambda rows: rows[0].update(mask=False),
        lambda rows: rows[0].update(fraction_percent=25),
        lambda rows: rows.__setitem__(slice(0, 2), [rows[1], rows[0]]),
        lambda rows: rows.pop(),
        lambda rows: rows.append(copy.deepcopy(rows[-1])),
    ],
)
def test_prediction_identity_counterexamples(policy, mutation):
    asset = policy.assets[0]
    samples = synthetic_samples()
    rows = prediction_rows(asset, samples)
    mutation(rows)
    with pytest.raises(control.S4BControlError):
        control.validate_prediction_rows(
            rows, samples, asset=asset, split="validation"
        )


def test_worker_self_consistent_wrong_samples_still_rejected(policy, tmp_path):
    """Simulates worker exit 0 with expected+predictions altered together."""

    asset = policy.assets[0]
    preflight = tmp_path / "preflight"
    run_output = tmp_path / "worker"
    preflight.mkdir()
    run_output.mkdir()
    good = synthetic_samples()
    wrong = copy.deepcopy(good)
    wrong[0]["sample_id"] = "self_consistent_wrong"
    runner.write_json(preflight / "validation_samples.json", good)
    runner.write_json(run_output / "expected_samples.json", wrong)
    runner.write_jsonl(run_output / "predictions.jsonl", prediction_rows(asset, wrong))
    with pytest.raises(control.S4BControlError, match="preflight source"):
        runner._verify_worker_output(  # noqa: SLF001
            run_output,
            asset=asset,
            split="validation",
            preflight_dir=preflight,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            authorization_sha256=AUTHORIZATION_SHA,
            output_run_id=OUTPUT_RUN_ID,
            expected_data_store_dir="synthetic-datastore",
        )


@pytest.mark.parametrize("mutation", ["scalers", "scaler_hash", "configuration", "runtime_override"])
def test_strict_restoration_validator_rejects_identity_counterexamples(policy, mutation):
    asset = policy.assets[0]
    restoration = valid_restoration(asset, policy)
    if mutation == "scalers":
        restoration["task_scalers"][control.HUMAN3_TASKS[0]]["mean"] += 1.0
    elif mutation == "scaler_hash":
        restoration["scaler_identity_sha256"] = "0" * 64
    elif mutation == "configuration":
        restoration["effective_configuration"]["epochs"] += 1
    else:
        restoration["runtime_overrides"]["fit_conformal"] = True
    with pytest.raises(control.S4BControlError):
        control.validate_restoration_evidence(
            restoration,
            asset=asset,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            split="validation",
            expected_data_store_dir="synthetic-datastore",
        )


@pytest.mark.parametrize("mutation", ["scalers", "scaler_hash", "configuration", "runtime_override"])
def test_collection_path_uses_strict_restoration_validator(policy, tmp_path, mutation):
    asset = policy.assets[0]
    samples = synthetic_samples()
    preflight = tmp_path / "preflight"
    output = tmp_path / "worker"
    preflight.mkdir()
    output.mkdir()
    rows = prediction_rows(asset, samples)
    metrics = control.validate_prediction_rows(rows, samples, asset=asset, split="validation")
    runner.write_json(preflight / "validation_samples.json", samples)
    runner.write_json(output / "expected_samples.json", samples)
    runner.write_jsonl(output / "predictions.jsonl", rows)
    runner.write_json(output / "metrics.json", metrics)
    restoration = valid_restoration(asset, policy)
    if mutation == "scalers":
        restoration["task_scalers"][control.HUMAN3_TASKS[0]]["std"] += 1.0
    elif mutation == "scaler_hash":
        restoration["scaler_identity_sha256"] = "0" * 64
    elif mutation == "configuration":
        restoration["effective_configuration"]["epochs"] += 1
    else:
        restoration["runtime_overrides"]["load_path"] = "wrong.pt"
    runner.write_json(output / "restoration.json", restoration)
    with pytest.raises(control.S4BControlError):
        runner._verify_worker_output(  # noqa: SLF001
            output,
            asset=asset,
            split="validation",
            preflight_dir=preflight,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            authorization_sha256=AUTHORIZATION_SHA,
            output_run_id=OUTPUT_RUN_ID,
            expected_data_store_dir="synthetic-datastore",
        )


def write_reference(path, rows):
    observed = [row for row in rows if row["mask"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "endpoint", "y_true", "y_pred"]
        )
        writer.writeheader()
        for row in observed:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "endpoint": row["endpoint"],
                    "y_true": row["y_true"],
                    "y_pred": row["y_pred"],
                }
            )


def test_existing_historical_csv_exact_comparison(policy, tmp_path):
    asset = policy.assets[0]
    samples = synthetic_samples()
    rows = prediction_rows(asset, samples)
    metrics = control.validate_prediction_rows(
        rows, samples, asset=asset, split="validation"
    )
    path = tmp_path / "history.csv"
    write_reference(path, rows)
    receipt = control.compare_validation_reference(rows, metrics, asset, path)
    assert receipt["reference_level"] == "HISTORICAL_SAMPLE_PREDICTIONS"
    assert receipt["sample_equivalence_claimed"] is True


def actual_diagnostics_reference(tmp_path, rows):
    sample_ids = [sample["sample_id"] for sample in synthetic_samples()]
    route_records = {}
    for task in control.HUMAN3_TASKS:
        task_rows = [row for row in rows if row["endpoint"] == task and row["mask"]]
        targets = torch.tensor([row["y_true"] for row in task_rows])
        predictions = torch.tensor([row["y_pred"] for row in task_rows])
        route_records[task] = {
            "target": [targets],
            "base": [predictions],
            "route": [predictions],
            "final": [predictions],
            "route_regret": [torch.zeros_like(targets)],
            "final_regret": [torch.zeros_like(targets)],
            "null": [torch.ones_like(targets)],
            "entropy": [torch.zeros_like(targets)],
            "sample_id": [row["sample_id"] for row in task_rows],
        }
    writer = RunDiagnosticsWriter(
        tmp_path,
        control.HUMAN3_TASKS,
        {sample["sample_id"]: sample["row_index"] for sample in synthetic_samples()},
    )
    writer.note_best_epoch(1, route_records)
    writer.write_best_artifacts(1)
    return tmp_path / "best_validation_human3_predictions.csv"


def test_actual_run_diagnostics_endpoint_major_csv_matches_sample_major_export(policy, tmp_path):
    asset = policy.assets[0]
    samples = synthetic_samples()
    rows = prediction_rows(asset, samples)
    metrics = control.validate_prediction_rows(rows, samples, asset=asset, split="validation")
    path = actual_diagnostics_reference(tmp_path, rows)
    receipt = control.compare_validation_reference(rows, metrics, asset, path)
    assert receipt["status"] == "PASS"
    assert receipt["historical_row_order_ignored"] is True
    assert receipt["compared_row_index_count"] == 8


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "label", "prediction", "row_index"])
def test_actual_run_diagnostics_reference_counterexamples(policy, tmp_path, mutation):
    asset = policy.assets[0]
    samples = synthetic_samples()
    rows = prediction_rows(asset, samples)
    metrics = control.validate_prediction_rows(rows, samples, asset=asset, split="validation")
    path = actual_diagnostics_reference(tmp_path, rows)
    with path.open(encoding="utf-8", newline="") as handle:
        historical = list(csv.DictReader(handle))
        fields = list(historical[0])
    if mutation == "duplicate":
        historical.append(copy.deepcopy(historical[0]))
    elif mutation == "missing":
        historical.pop()
    elif mutation == "extra":
        extra = copy.deepcopy(historical[0])
        extra["sample_id"] = "unexpected"
        historical.append(extra)
    elif mutation == "label":
        historical[0]["label"] = "999"
    elif mutation == "prediction":
        historical[0]["final_prediction"] = "999"
    else:
        historical[0]["row_index"] = "999"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(historical)
    with pytest.raises(control.S4BControlError):
        control.compare_validation_reference(rows, metrics, asset, path)


def posix_reference_sources(policy, *, first_exists=False, first_sha=None):
    return [
        {
            "run_id": asset.run_id,
            "historical_csv_path": (
                f"{asset.run_dir}/diagnostics/"
                "best_validation_human3_predictions.csv"
            ),
            "historical_csv_exists": first_exists and index == 0,
            "historical_csv_sha256": first_sha if first_exists and index == 0 else None,
        }
        for index, asset in enumerate(policy.assets)
    ]


def test_offline_reference_sources_use_pure_posix_server_identity(policy):
    references = posix_reference_sources(policy)
    verified = runner._verify_reference_sources(  # noqa: SLF001
        references, policy, check_live_paths=False
    )
    assert len(verified) == 40
    assert all(row["historical_csv_path"].startswith("/home/") for row in references)
    assert all("\\" not in row["historical_csv_path"] for row in references)


@pytest.mark.parametrize("mutation", ["run_dir", "filename", "cross_run"])
def test_offline_reference_sources_reject_wrong_posix_identity(policy, mutation):
    references = posix_reference_sources(policy)
    if mutation == "run_dir":
        references[0]["historical_csv_path"] = references[0][
            "historical_csv_path"
        ].replace("d8_b1_f10_e40", "wrong_run", 1)
    elif mutation == "filename":
        references[0]["historical_csv_path"] = references[0][
            "historical_csv_path"
        ].replace("best_validation_human3_predictions.csv", "predictions.csv")
    else:
        references[0]["historical_csv_path"] = references[1]["historical_csv_path"]
    with pytest.raises(control.S4BControlError, match="reference identity"):
        runner._verify_reference_sources(  # noqa: SLF001
            references, policy, check_live_paths=False
        )


def write_historical_worker_fixture(root, policy, rows, *, frozen_original_bytes):
    asset = policy.assets[0]
    samples = synthetic_samples()
    preflight = root / "preflight"
    output = root / "worker"
    preflight.mkdir(parents=True)
    output.mkdir()
    frozen_sha = hashlib.sha256(frozen_original_bytes).hexdigest()
    references = posix_reference_sources(
        policy, first_exists=True, first_sha=frozen_sha
    )
    runner.write_json(preflight / "validation_samples.json", samples)
    runner.write_json(preflight / "reference_sources.json", references)
    runner.write_json(preflight / "preflight_receipt.json", {"synthetic": True})
    metrics = control.validate_prediction_rows(
        rows, samples, asset=asset, split="validation"
    )
    runner.write_json(output / "expected_samples.json", samples)
    runner.write_jsonl(output / "predictions.jsonl", rows)
    runner.write_json(output / "metrics.json", metrics)
    runner.write_json(output / "restoration.json", valid_restoration(asset, policy))
    runner.write_json(
        output / "sample_binding.json",
        {
            "status": "PASS",
            "source": "SERVER_PREFLIGHT_INDEPENDENT_DATASTORE_TABLE",
            "split": "validation",
            "sample_identity_sha256": control.identity_sha256(samples),
            "sample_count": len(samples),
            "prediction_row_count": len(rows),
        },
    )
    runner.write_json(output / "reference_source.json", references[0])
    historical_copy = output / "historical_validation_original.csv"
    write_reference(historical_copy, rows)
    comparison = control.compare_validation_reference(
        rows, metrics, asset, historical_copy
    )
    runner.write_json(output / "reference_comparison.json", comparison)
    return preflight, output


def test_worker_historical_copy_is_bound_to_frozen_original_sha(policy, tmp_path):
    asset = policy.assets[0]
    rows = prediction_rows(asset, synthetic_samples())
    original_path = tmp_path / "original.csv"
    write_reference(original_path, rows)
    original_bytes = original_path.read_bytes()
    original_path.unlink()
    preflight, output = write_historical_worker_fixture(
        tmp_path / "positive",
        policy,
        rows,
        frozen_original_bytes=original_bytes,
    )
    result = runner._verify_worker_output(  # noqa: SLF001
        output,
        asset=asset,
        split="validation",
        preflight_dir=preflight,
        policy=policy,
        implementation_commit=IMPLEMENTATION_COMMIT,
        authorization_sha256=AUTHORIZATION_SHA,
        output_run_id=OUTPUT_RUN_ID,
        expected_data_store_dir="synthetic-datastore",
        check_live_reference_paths=False,
    )
    assert result["reference_level"] == "HISTORICAL_SAMPLE_PREDICTIONS"

    changed_rows = prediction_rows(asset, synthetic_samples(), delta=0.75)
    bad_preflight, bad_output = write_historical_worker_fixture(
        tmp_path / "self-consistent-tamper",
        policy,
        changed_rows,
        frozen_original_bytes=original_bytes,
    )
    with pytest.raises(control.S4BControlError, match="copy SHA"):
        runner._verify_worker_output(  # noqa: SLF001
            bad_output,
            asset=asset,
            split="validation",
            preflight_dir=bad_preflight,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            authorization_sha256=AUTHORIZATION_SHA,
            output_run_id=OUTPUT_RUN_ID,
            expected_data_store_dir="synthetic-datastore",
            check_live_reference_paths=False,
        )


def test_existing_mismatched_csv_cannot_fall_back_to_macro(policy, tmp_path):
    asset = policy.assets[0]
    samples = synthetic_samples()
    delta = asset.best_validation_macro_rmse
    rows = prediction_rows(asset, samples, delta=delta)
    metrics = control.validate_prediction_rows(
        rows, samples, asset=asset, split="validation"
    )
    path = tmp_path / "history.csv"
    write_reference(path, rows)
    content = path.read_text(encoding="utf-8")
    path.write_text(content.replace(str(rows[0]["y_pred"]), "999.0", 1), encoding="utf-8")
    with pytest.raises(control.S4BControlError, match="prediction differs"):
        control.compare_validation_reference(rows, metrics, asset, path)


def test_missing_csv_macro_only_fallback_positive_and_negative(policy, tmp_path):
    asset = policy.assets[0]
    samples = synthetic_samples()
    rows = prediction_rows(asset, samples, delta=asset.best_validation_macro_rmse)
    metrics = control.validate_prediction_rows(
        rows, samples, asset=asset, split="validation"
    )
    missing = tmp_path / "does_not_exist.csv"
    receipt = control.compare_validation_reference(rows, metrics, asset, missing)
    assert receipt["reference_level"] == "FROZEN_MACRO_ONLY"
    assert receipt["historical_csv_exists"] is False
    assert receipt["sample_equivalence_claimed"] is False
    wrong_rows = prediction_rows(asset, samples, delta=0.01)
    wrong_metrics = control.validate_prediction_rows(
        wrong_rows, samples, asset=asset, split="validation"
    )
    with pytest.raises(control.S4BControlError, match="macro-only"):
        control.compare_validation_reference(wrong_rows, wrong_metrics, asset, missing)


def test_validation_gate_requires_all_forty_same_identity(policy):
    results = [validation_result(asset, policy) for asset in policy.assets]
    gate = control.build_validation_gate(
        results, policy=policy, implementation_commit=IMPLEMENTATION_COMMIT, **gate_kwargs()
    )
    assert gate["status"] == "PASS"
    assert gate["test_unlocked"] is True
    assert gate["passed_validation_count"] == 40
    assert len(gate["validation_results"]) == 40
    control.verify_validation_gate(
        gate, policy=policy, implementation_commit=IMPLEMENTATION_COMMIT, **gate_kwargs()
    )


def test_one_validation_failure_or_missing_run_never_unlocks_test(policy):
    results = [validation_result(asset, policy) for asset in policy.assets]
    results[7]["status"] = "FAIL"
    gate = control.build_validation_gate(
        results, policy=policy, implementation_commit=IMPLEMENTATION_COMMIT, **gate_kwargs()
    )
    assert gate["status"] == "FAIL"
    assert gate["test_unlocked"] is False
    assert gate["authorized_test_run_ids"] == []
    missing = control.build_validation_gate(
        results[:-1], policy=policy, implementation_commit=IMPLEMENTATION_COMMIT, **gate_kwargs()
    )
    assert missing["test_unlocked"] is False
    assert missing["missing_run_ids"] == [policy.assets[-1].run_id]


def test_validation_gate_rejects_duplicate_cross_fraction_and_tampering(policy):
    results = [validation_result(asset, policy) for asset in policy.assets]
    with pytest.raises(control.S4BControlError, match="duplicate"):
        control.build_validation_gate(
            [*results, copy.deepcopy(results[0])],
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            **gate_kwargs(),
        )
    crossed = copy.deepcopy(results)
    crossed[0]["fraction_percent"] = 25
    with pytest.raises(control.S4BControlError, match="fraction"):
        control.build_validation_gate(
            crossed, policy=policy, implementation_commit=IMPLEMENTATION_COMMIT, **gate_kwargs()
        )
    gate = control.build_validation_gate(
        results, policy=policy, implementation_commit=IMPLEMENTATION_COMMIT, **gate_kwargs()
    )
    gate["validation_results"][0]["prediction_sha256"] = "0" * 64
    with pytest.raises(control.S4BControlError, match="reproduce"):
        control.verify_validation_gate(
            gate, policy=policy, implementation_commit=IMPLEMENTATION_COMMIT, **gate_kwargs()
        )


def test_validation_gate_rejects_self_reported_pass_without_file_evidence(policy):
    results = [validation_result(asset, policy) for asset in policy.assets]
    results[0].pop("evidence_files")
    with pytest.raises(control.S4BControlError, match="evidence"):
        control.build_validation_gate(
            results,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            **gate_kwargs(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_run_id", "wrong-output"),
        ("authorization_sha256", "0" * 64),
        ("preflight_receipt_sha256", "0" * 64),
    ],
)
def test_validation_gate_rejects_cross_context_result(policy, field, value):
    results = [validation_result(asset, policy) for asset in policy.assets]
    results[0][field] = value
    with pytest.raises(control.S4BControlError):
        control.build_validation_gate(
            results,
            policy=policy,
            implementation_commit=IMPLEMENTATION_COMMIT,
            **gate_kwargs(),
        )


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_actual_037_ten_aliases_and_original_prediction_bytes(aliases):
    assert aliases.outer_sha256 == control.S3C3_OUTER_SHA256
    assert aliases.canonical_sha256 == control.S3C3_CANONICAL_SHA256
    assert len(aliases.aliases) == 10
    assert len(aliases.frozen_samples) == 28
    for alias in aliases.aliases:
        identity = (alias.method, alias.seed)
        assert hashlib.sha256(alias.prediction_bytes).hexdigest() == control.ALIAS_PREDICTION_SHA256[identity]
        assert hashlib.sha256(alias.metrics_bytes).hexdigest() == control.ALIAS_METRICS_SHA256[identity]
        assert alias.fraction_percent == 100


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_analysis_manifest_locates_exact_100_percent_members(aliases):
    if not ANALYSIS_MANIFEST.is_file():
        pytest.skip("S3D analysis manifest is not installed")
    receipt = control.verify_analysis_manifest(ANALYSIS_MANIFEST, aliases)
    assert receipt["status"] == "PASS"
    assert receipt["verified_alias_count"] == 10


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_037_outer_byte_change_is_rejected(tmp_path):
    if not S3C3_FIXTURE.is_file():
        pytest.skip("accepted 037 archive is not installed")
    changed = bytearray(S3C3_FIXTURE.read_bytes())
    changed[-1] ^= 1
    path = tmp_path / "changed.zip"
    path.write_bytes(changed)
    with pytest.raises(control.S4BControlError, match="outer archive SHA"):
        control.verify_037_aliases(path)


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_unified_matrix_is_exactly_fifty_and_100_percent_is_alias_only(policy, aliases):
    low = [low_fraction_test_result(asset, policy) for asset in policy.assets]
    unified = control.build_unified_50_metrics(low, aliases, policy=policy)
    assert len(unified) == 50
    assert len(
        {(row["method"], row["seed"], row["fraction_percent"]) for row in unified}
    ) == 50
    hundred = [row for row in unified if row["fraction_percent"] == 100]
    assert len(hundred) == 10
    assert all(row["source"] == "ALIAS_037_NO_RERUN" for row in hundred)
    assert all(row["source_canonical_sha256"] == control.S3C3_CANONICAL_SHA256 for row in hundred)


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_unified_matrix_rejects_missing_duplicate_and_cross_fraction(policy, aliases):
    low = [low_fraction_test_result(asset, policy) for asset in policy.assets]
    with pytest.raises(control.S4BControlError, match="exactly forty"):
        control.build_unified_50_metrics(low[:-1], aliases, policy=policy)
    with pytest.raises(control.S4BControlError, match="duplicated"):
        control.build_unified_50_metrics([*low, copy.deepcopy(low[0])], aliases, policy=policy)
    crossed = copy.deepcopy(low)
    crossed[0]["fraction_percent"] = 25
    with pytest.raises(control.S4BControlError, match="identity differs"):
        control.build_unified_50_metrics(crossed, aliases, policy=policy)


def build_full_validation_contract(root, policy, frozen_test_samples):
    preflight = root / "preflight_snapshots"
    validation = root / "validation_evidence"
    logs = validation / "logs"
    logs.mkdir(parents=True)
    validation_samples = synthetic_samples()
    runner.write_json(preflight / "validation_samples.json", validation_samples)
    runner.write_json(preflight / "test_samples.json", frozen_test_samples)
    authorization = fixture_authorization(policy, fixture_only=False)
    runner.write_json(preflight / "authorization_snapshot.json", authorization)
    authorization_sha = control.sha256_file(preflight / "authorization_snapshot.json")
    code = {
        "runner_sha256": control.sha256_file(ROOT / "scripts/run_s4_low_fraction_export.py"),
        "control_sha256": control.sha256_file(ROOT / "s4_low_fraction_control.py"),
        "trainer_sha256": control.sha256_file(ROOT / "trainer.py"),
    }
    wsl = valid_wsl_receipt(policy)
    wsl.update(code)
    runner.write_json(preflight / "wsl_receipt_snapshot.json", wsl)
    sample_sources = {
        "datastore": datastore_source_identity(policy),
        "validation": {
            "sample_identity_sha256": control.identity_sha256(validation_samples),
        },
        "test": {
            "sample_identity_sha256": control.FROZEN_TEST_SAMPLE_PROJECTION_SHA256,
            "source_outer_sha256": control.S3C3_OUTER_SHA256,
            "source_canonical_sha256": control.S3C3_CANONICAL_SHA256,
        },
    }
    runner.write_json(preflight / "sample_sources.json", sample_sources)
    asset_identity = [
        {
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
        for asset in policy.assets
    ]
    runner.write_json(preflight / "asset_identity.json", asset_identity)
    first_reference_rows = prediction_rows(
        policy.assets[0],
        validation_samples,
        delta=policy.assets[0].best_validation_macro_rmse,
    )
    temporary_reference = root / "first_historical_reference.csv"
    write_reference(temporary_reference, first_reference_rows)
    first_reference_bytes = temporary_reference.read_bytes()
    temporary_reference.unlink()
    first_reference_sha = hashlib.sha256(first_reference_bytes).hexdigest()
    references = [
        {
            "run_id": asset.run_id,
            "historical_csv_path": (
                f"{asset.run_dir}/diagnostics/"
                "best_validation_human3_predictions.csv"
            ),
            "historical_csv_exists": index == 0,
            "historical_csv_sha256": first_reference_sha if index == 0 else None,
        }
        for index, asset in enumerate(policy.assets)
    ]
    runner.write_json(preflight / "reference_sources.json", references)
    runner.write_json(preflight / "command.json", {"synthetic": True})
    receipt = {
        "schema_version": 1,
        "receipt_kind": "S4B_SERVER_PREFLIGHT",
        "status": "PASS",
        "task_id": runner.TASK_ID,
        "output_run_id": OUTPUT_RUN_ID,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "policy_sha256": policy.sha256,
        "core_policy_sha256": control.CORE_POLICY_SHA256,
        "authorization_sha256": authorization_sha,
        "expected_data_store_dir": "synthetic-datastore",
        "wsl_receipt_sha256": control.sha256_file(preflight / "wsl_receipt_snapshot.json"),
        **code,
        "verified_asset_count": 40,
        "checkpoint_payload_count": 40,
        "validation_sample_identity_sha256": control.identity_sha256(validation_samples),
        "validation_sample_count": len(validation_samples),
        "test_sample_identity_sha256": control.FROZEN_TEST_SAMPLE_PROJECTION_SHA256,
        "test_sample_count": 28,
        "datastore_fingerprint": policy.assets[0].expected_data_config["datastore_fingerprint"],
        "datastore_json_sha256": "4" * 64,
        "sample_sources_sha256": control.sha256_file(preflight / "sample_sources.json"),
        "reference_sources_sha256": control.sha256_file(preflight / "reference_sources.json"),
        "reference_source_count": 40,
        "frozen_source_outer_sha256": control.S3C3_OUTER_SHA256,
        "frozen_source_canonical_sha256": control.S3C3_CANONICAL_SHA256,
    }
    runner.write_json(preflight / "preflight_receipt.json", receipt)
    preflight_sha = control.sha256_file(preflight / "preflight_receipt.json")
    results = []
    for asset, reference in zip(policy.assets, references):
        run_dir = validation / f"validation_{asset.run_id}"
        run_dir.mkdir()
        rows = prediction_rows(
            asset,
            validation_samples,
            delta=asset.best_validation_macro_rmse,
        )
        metrics = control.validate_prediction_rows(
            rows, validation_samples, asset=asset, split="validation"
        )
        runner.write_json(run_dir / "expected_samples.json", validation_samples)
        runner.write_jsonl(run_dir / "predictions.jsonl", rows)
        runner.write_json(run_dir / "metrics.json", metrics)
        runner.write_json(
            run_dir / "restoration.json",
            valid_restoration(asset, policy),
        )
        runner.write_json(
            run_dir / "sample_binding.json",
            {
                "status": "PASS",
                "source": "SERVER_PREFLIGHT_INDEPENDENT_DATASTORE_TABLE",
                "split": "validation",
                "sample_identity_sha256": control.identity_sha256(validation_samples),
                "sample_count": len(validation_samples),
                "prediction_row_count": len(rows),
                "exact_ordered_match": True,
                "frozen_test_projection_sha256": None,
            },
        )
        runner.write_json(run_dir / "reference_source.json", reference)
        if reference["historical_csv_exists"]:
            (run_dir / "historical_validation_original.csv").write_bytes(
                first_reference_bytes
            )
        comparison = control.compare_validation_reference(
            rows,
            metrics,
            asset,
            (
                run_dir / "historical_validation_original.csv"
                if reference["historical_csv_exists"]
                else run_dir / "confirmed_missing_historical_validation.csv"
            ),
        )
        runner.write_json(run_dir / "reference_comparison.json", comparison)
        evidence_names = list(control.VALIDATION_EVIDENCE_FILES)
        if reference["historical_csv_exists"]:
            evidence_names.append("historical_validation_original.csv")
        evidence = {
            name: control.sha256_file(run_dir / name) for name in evidence_names
        }
        results.append(
            {
                "run_id": asset.run_id,
                "method": asset.method,
                "seed": asset.seed,
                "fraction_percent": asset.fraction_percent,
                "split": "validation",
                "status": "PASS",
                "checkpoint_sha256": asset.checkpoint_sha256,
                "policy_sha256": policy.sha256,
                "implementation_commit": IMPLEMENTATION_COMMIT,
                "output_run_id": OUTPUT_RUN_ID,
                "authorization_sha256": authorization_sha,
                "preflight_receipt_sha256": preflight_sha,
                "prediction_sha256": evidence["predictions.jsonl"],
                "metrics": metrics,
                "reference_level": comparison["reference_level"],
                "historical_csv_exists": reference["historical_csv_exists"],
                "evidence_files": evidence,
                "evidence_identity_sha256": control.identity_sha256(evidence),
            }
        )
        for suffix in ("out", "err"):
            (logs / f"{asset.run_id}.{suffix}").write_bytes(b"")
        runner.write_json(logs / f"{asset.run_id}.json", {"exit_code": 0})
    runner.write_json(validation / "completed_results.json", results)
    runner.write_json(validation / "errors.json", [])
    runner.write_json(
        validation / "batch_identity.json",
        {
            "task_id": runner.TASK_ID,
            "split": "validation",
            "output_run_id": OUTPUT_RUN_ID,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "policy_sha256": policy.sha256,
            "authorization_sha256": authorization_sha,
            "preflight_receipt_sha256": preflight_sha,
        },
    )
    runner.write_json(
        validation / "batch_status.json",
        {
            "split": "validation",
            "execution_status": "FINISHED",
            "completed_run_count": 40,
            "failed_run_count": 0,
            "not_executed_run_ids": [],
        },
    )
    log_sha = runner._validation_log_identity(validation)  # noqa: SLF001
    gate = control.build_validation_gate(
        results,
        policy=policy,
        implementation_commit=IMPLEMENTATION_COMMIT,
        output_run_id=OUTPUT_RUN_ID,
        authorization_sha256=authorization_sha,
        preflight_receipt_sha256=preflight_sha,
        validation_log_identity_sha256=log_sha,
    )
    runner.write_json(validation / "validation_gate.json", gate)
    runner.write_json(root / "validation_gate.json", gate)
    return authorization_sha, preflight_sha


@pytest.mark.skipif(not LOCAL_FIXTURES, reason="local accepted attachments are opt-in")
def test_synthetic_final_archive_reverifies_forty_plus_ten(policy, aliases, tmp_path):
    """End-to-end archive verification with synthetic low-fraction predictions."""

    root = tmp_path / "S4B_SYNTHETIC_FINAL"
    (root / "low_fraction_test").mkdir(parents=True)
    (root / "aliases_100").mkdir()
    (root / "preflight_snapshots").mkdir()
    samples = list(aliases.frozen_samples)
    authorization_sha, preflight_sha = build_full_validation_contract(
        root, policy, samples
    )
    low_results = []
    for asset in policy.assets:
        run_dir = root / "low_fraction_test" / asset.run_id
        run_dir.mkdir()
        rows = prediction_rows(asset, samples, split="test", delta=0.125)
        metrics = control.validate_prediction_rows(
            rows, samples, asset=asset, split="test"
        )
        runner.write_json(run_dir / "expected_samples.json", samples)
        runner.write_jsonl(run_dir / "predictions.jsonl", rows)
        runner.write_json(run_dir / "metrics.json", metrics)
        runner.write_json(
            run_dir / "restoration.json",
            valid_restoration(
                asset, policy, split="test", datastore="synthetic-datastore"
            ),
        )
        runner.write_json(
            run_dir / "sample_binding.json",
            {
                "status": "PASS",
                "source": "SERVER_PREFLIGHT_INDEPENDENT_DATASTORE_TABLE",
                "split": "test",
                "sample_identity_sha256": control.FROZEN_TEST_SAMPLE_PROJECTION_SHA256,
                "sample_count": 28,
                "prediction_row_count": 84,
            },
        )
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
                "implementation_commit": IMPLEMENTATION_COMMIT,
                "output_run_id": OUTPUT_RUN_ID,
                "authorization_sha256": authorization_sha,
                "preflight_receipt_sha256": preflight_sha,
                "prediction_sha256": control.sha256_file(
                    run_dir / "predictions.jsonl"
                ),
                "metrics": metrics,
            }
        )
    (root / "batch_evidence").mkdir()
    (root / "batch_evidence/logs").mkdir()
    for run_id in control.S4B_RUN_IDS:
        for suffix in ("out", "err"):
            (root / "batch_evidence/logs" / f"{run_id}.{suffix}").write_bytes(b"")
        runner.write_json(
            root / "batch_evidence/logs" / f"{run_id}.json",
            {"exit_code": 0},
        )
    runner.write_json(root / "batch_evidence/completed_results.json", low_results)
    runner.write_json(root / "batch_evidence/errors.json", [])
    runner.write_json(
        root / "batch_evidence/batch_identity.json",
        {
            "task_id": runner.TASK_ID,
            "split": "test",
            "output_run_id": OUTPUT_RUN_ID,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "policy_sha256": policy.sha256,
            "authorization_sha256": authorization_sha,
            "preflight_receipt_sha256": preflight_sha,
        },
    )
    runner.write_json(
        root / "batch_evidence/batch_status.json",
        {
            "split": "test",
            "execution_status": "FINISHED",
            "completed_run_count": 40,
            "failed_run_count": 0,
            "not_executed_run_ids": [],
        },
    )
    alias_index = []
    for alias in aliases.aliases:
        alias_dir = root / "aliases_100" / alias.run_id
        alias_dir.mkdir()
        prediction_path = alias_dir / "predictions.jsonl"
        metrics_path = alias_dir / "metrics.json"
        prediction_path.write_bytes(alias.prediction_bytes)
        metrics_path.write_bytes(alias.metrics_bytes)
        alias_index.append(
            {
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
                "local_prediction_path": prediction_path.relative_to(root).as_posix(),
                "local_metrics_path": metrics_path.relative_to(root).as_posix(),
                "rerun": False,
            }
        )
    runner.write_json(root / "aliases_100.json", alias_index)
    runner.write_json(
        root / "unified_50_test_metrics.json",
        control.build_unified_50_metrics(low_results, aliases, policy=policy),
    )
    runner.write_json(
        root / "source_archives.json",
        {
            "s3c3_outer_sha256": control.S3C3_OUTER_SHA256,
            "s3c3_canonical_sha256": control.S3C3_CANONICAL_SHA256,
            "frozen_test_sample_projection_sha256": control.FROZEN_TEST_SAMPLE_PROJECTION_SHA256,
            "analysis_manifest": {"status": "PASS", "verified_alias_count": 10},
        },
    )
    runner.write_json(
        root / "final_status.json",
        {
            "schema_version": 1,
            "task_id": runner.TASK_ID,
            "status": "COMPLETE_PENDING_CODEX_REVIEW",
            "acceptance_status": "PENDING_CODEX_REVIEW",
            "policy_sha256": policy.sha256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "output_run_id": OUTPUT_RUN_ID,
            "authorization_sha256": authorization_sha,
            "preflight_receipt_sha256": preflight_sha,
            "expected_data_store_dir": "synthetic-datastore",
            "low_fraction_test_count": 40,
            "alias_100_count": 10,
            "unified_result_count": 50,
            "failed_run_ids": [],
            "alias_rerun_count": 0,
        },
    )
    archive = tmp_path / "S4B_SYNTHETIC_FINAL.zip"
    runner._pack_directory(root, archive)  # noqa: SLF001
    assert runner.verify_archive(Namespace(policy=POLICY_PATH, archive=archive)) == 0

    tampered_root = tmp_path / "S4B_SELF_CONSISTENT_HISTORY_TAMPER"
    shutil.copytree(root, tampered_root)
    (tampered_root / "checksums.sha256").unlink()
    first_asset = policy.assets[0]
    validation_run = (
        tampered_root
        / "validation_evidence"
        / f"validation_{first_asset.run_id}"
    )
    changed_rows = control.parse_jsonl(
        (validation_run / "predictions.jsonl").read_bytes()
    )
    changed_rows[0]["y_pred"] += 0.5
    changed_metrics = control.validate_prediction_rows(
        changed_rows,
        synthetic_samples(),
        asset=first_asset,
        split="validation",
    )
    for path in (
        validation_run / "predictions.jsonl",
        validation_run / "metrics.json",
        validation_run / "historical_validation_original.csv",
        validation_run / "reference_comparison.json",
    ):
        path.unlink()
    runner.write_jsonl(validation_run / "predictions.jsonl", changed_rows)
    runner.write_json(validation_run / "metrics.json", changed_metrics)
    write_reference(
        validation_run / "historical_validation_original.csv", changed_rows
    )
    changed_comparison = control.compare_validation_reference(
        changed_rows,
        changed_metrics,
        first_asset,
        validation_run / "historical_validation_original.csv",
    )
    runner.write_json(
        validation_run / "reference_comparison.json", changed_comparison
    )
    evidence_names = [
        *control.VALIDATION_EVIDENCE_FILES,
        "historical_validation_original.csv",
    ]
    changed_evidence = {
        name: control.sha256_file(validation_run / name) for name in evidence_names
    }
    completed_path = tampered_root / "validation_evidence/completed_results.json"
    completed = control.read_json(completed_path)
    completed[0].update(
        {
            "prediction_sha256": changed_evidence["predictions.jsonl"],
            "metrics": changed_metrics,
            "evidence_files": changed_evidence,
            "evidence_identity_sha256": control.identity_sha256(changed_evidence),
        }
    )
    completed_path.unlink()
    runner.write_json(completed_path, completed)
    old_gate = control.read_json(tampered_root / "validation_gate.json")
    changed_gate = control.build_validation_gate(
        completed,
        policy=policy,
        implementation_commit=IMPLEMENTATION_COMMIT,
        output_run_id=OUTPUT_RUN_ID,
        authorization_sha256=authorization_sha,
        preflight_receipt_sha256=preflight_sha,
        validation_log_identity_sha256=old_gate[
            "validation_log_identity_sha256"
        ],
    )
    for gate_path in (
        tampered_root / "validation_gate.json",
        tampered_root / "validation_evidence/validation_gate.json",
    ):
        gate_path.unlink()
        runner.write_json(gate_path, changed_gate)
    tampered_archive = tmp_path / "S4B_SELF_CONSISTENT_HISTORY_TAMPER.zip"
    runner._pack_directory(tampered_root, tampered_archive)  # noqa: SLF001
    with pytest.raises(control.S4BControlError, match="copy SHA"):
        runner.verify_archive(
            Namespace(policy=POLICY_PATH, archive=tampered_archive)
        )

    (root / "checksums.sha256").unlink()
    restoration_path = (
        root / "low_fraction_test" / policy.assets[0].run_id / "restoration.json"
    )
    original = control.read_json(restoration_path)
    for index, mutation in enumerate(
        ("scalers", "scaler_hash", "configuration", "runtime_override")
    ):
        changed = copy.deepcopy(original)
        if mutation == "scalers":
            changed["task_scalers"][control.HUMAN3_TASKS[0]]["mean"] += 1.0
        elif mutation == "scaler_hash":
            changed["scaler_identity_sha256"] = "0" * 64
        elif mutation == "configuration":
            changed["effective_configuration"]["epochs"] += 1
        else:
            changed["runtime_overrides"]["fit_conformal"] = True
        restoration_path.unlink()
        runner.write_json(restoration_path, changed)
        changed_archive = tmp_path / f"S4B_SYNTHETIC_FINAL_BAD_{index}.zip"
        runner._pack_directory(root, changed_archive)  # noqa: SLF001
        with pytest.raises(control.S4BControlError):
            runner.verify_archive(
                Namespace(policy=POLICY_PATH, archive=changed_archive)
            )
        (root / "checksums.sha256").unlink()
    restoration_path.unlink()
    runner.write_json(restoration_path, original)
    authorization_snapshot = root / "preflight_snapshots/authorization_snapshot.json"
    authorization_bytes = authorization_snapshot.read_bytes()
    authorization_snapshot.unlink()
    missing_auth_archive = tmp_path / "S4B_SYNTHETIC_FINAL_MISSING_AUTH.zip"
    runner._pack_directory(root, missing_auth_archive)  # noqa: SLF001
    with pytest.raises((control.S4BControlError, FileNotFoundError)):
        runner.verify_archive(Namespace(policy=POLICY_PATH, archive=missing_auth_archive))
    (root / "checksums.sha256").unlink()
    authorization_snapshot.write_bytes(authorization_bytes)
    validation_prediction = (
        root
        / "validation_evidence"
        / f"validation_{policy.assets[0].run_id}"
        / "predictions.jsonl"
    )
    validation_bytes = validation_prediction.read_bytes()
    validation_prediction.write_bytes(validation_bytes + b"\n")
    changed_validation_archive = tmp_path / "S4B_SYNTHETIC_FINAL_BAD_VALIDATION.zip"
    runner._pack_directory(root, changed_validation_archive)  # noqa: SLF001
    with pytest.raises(control.S4BControlError):
        runner.verify_archive(
            Namespace(policy=POLICY_PATH, archive=changed_validation_archive)
        )


def test_synthetic_human3_cpu_decode_descale_and_model_immutability(policy):
    torch.manual_seed(123)
    model = torch.nn.Linear(4, 3)
    before = control.tensor_state_sha256(model.state_dict())
    raw = {
        task: torch.tensor([[float(index), -1.0, -1.0], [float(index + 1), -1.0, -1.0]])
        for index, task in enumerate(control.HUMAN3_TASKS)
    }
    scalers = policy.assets[0].expected_scalers
    decoded = control.decode_synthetic_human3_batch(raw, scalers, model)
    assert control.tensor_state_sha256(model.state_dict()) == before
    for index, task in enumerate(control.HUMAN3_TASKS):
        assert decoded[task] == pytest.approx(
            [
                index * scalers[task]["std"] + scalers[task]["mean"],
                (index + 1) * scalers[task]["std"] + scalers[task]["mean"],
            ]
        )


def test_synthetic_decode_rejects_bool_scaler(policy):
    model = torch.nn.Linear(1, 1)
    raw = {task: torch.zeros((2, 3)) for task in control.HUMAN3_TASKS}
    scalers = copy.deepcopy(dict(policy.assets[0].expected_scalers))
    scalers["man_oral_TDLo"]["mean"] = True
    with pytest.raises(control.S4BControlError, match="finite float"):
        control.decode_synthetic_human3_batch(raw, scalers, model)


def test_cli_help_lists_required_commands_from_arbitrary_cwd(tmp_path):
    script = ROOT / "scripts/run_s4_low_fraction_export.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for command in ("preflight", "run", "finalize", "verify"):
        assert command in result.stdout


def test_cli_synthetic_selftest_from_arbitrary_cwd(tmp_path):
    script = ROOT / "scripts/run_s4_low_fraction_export.py"
    result = subprocess.run(
        [sys.executable, str(script), "selftest", "--policy", str(POLICY_PATH)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "PASS"
    assert receipt["synthetic_only"] is True
    assert receipt["real_data_accessed"] is False
    assert receipt["training_called"] is False
    assert receipt["model_tensors_unchanged"] is True
    assert receipt["worker_shared_decode_path"] is True


@pytest.mark.skipif(
    os.environ.get("S4B_ISOLATED_CHILD") == "1",
    reason="avoid recursively creating another isolated checkout",
)
def test_default_no_asset_group_runs_in_isolated_copy_without_plans(tmp_path):
    isolated = tmp_path / "isolated"
    shutil.copytree(
        ROOT,
        isolated,
        ignore=shutil.ignore_patterns(
            ".git",
            ".tmp",
            "Plans&Results",
            "artifacts",
            "data",
            "outputs",
            "__pycache__",
            "*.pt",
            "*.pth",
            "*.zip",
        ),
    )
    assert not (isolated / "Plans&Results").exists()
    environment = dict(os.environ)
    environment.pop("S4B_LOCAL_FIXTURES", None)
    environment["S4B_ISOLATED_CHILD"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "pytest",
            "-q",
            "tests/test_s4_low_fraction_control.py",
            "--disable-warnings",
            "--basetemp",
            str(isolated / ".isolated-pytest"),
        ],
        cwd=isolated,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "passed" in result.stdout
    assert "Plans&Results" not in result.stderr


def test_cli_formal_worker_cannot_start_without_authorization(policy, tmp_path):
    arguments = Namespace(
        policy=POLICY_PATH,
        asset=policy.assets[0].run_id,
        authorization=tmp_path / "missing_authorization.json",
        implementation_commit=IMPLEMENTATION_COMMIT,
        split="validation",
        output_run_id=OUTPUT_RUN_ID,
    )
    with pytest.raises(control.S4BControlError, match="authorization file is missing"):
        # It fails before inspecting preflight, DataStore, torch, or CUDA.
        runner.worker(arguments)


def test_public_test_worker_replays_repository_and_validation_before_asset(
    policy, tmp_path, monkeypatch
):
    authorization = tmp_path / "authorization.json"
    runner.write_json(authorization, fixture_authorization(policy, fixture_only=False))
    events = []
    monkeypatch.setattr(
        runner,
        "_validate_repository",
        lambda repository, commit: events.append("repository"),
    )
    monkeypatch.setattr(
        runner,
        "_verify_server_preflight",
        lambda *args, **kwargs: events.append("preflight") or {},
    )
    monkeypatch.setattr(
        runner,
        "_verify_validation_evidence",
        lambda *args, **kwargs: events.append("validation") or {},
    )

    def asset_sentinel(*args, **kwargs):
        events.append("asset")
        raise RuntimeError("asset sentinel")

    monkeypatch.setattr(runner, "prepare_low_fraction_checkpoint", asset_sentinel)
    arguments = Namespace(
        policy=POLICY_PATH,
        asset=policy.assets[0].run_id,
        authorization=authorization,
        implementation_commit=IMPLEMENTATION_COMMIT,
        split="test",
        output_run_id=OUTPUT_RUN_ID,
        repository=ROOT,
        preflight=tmp_path / "preflight",
        validation_dir=tmp_path / "validation",
        validation_gate=tmp_path / "validation/validation_gate.json",
        datastore_dir=tmp_path / "datastore",
        output=tmp_path / "output",
    )
    with pytest.raises(RuntimeError, match="asset sentinel"):
        runner.worker(arguments)
    assert events == ["repository", "preflight", "validation", "asset"]


def test_public_test_worker_bad_repository_or_validation_never_touches_asset(
    policy, tmp_path, monkeypatch
):
    authorization = tmp_path / "authorization.json"
    runner.write_json(authorization, fixture_authorization(policy, fixture_only=False))
    touched = []
    monkeypatch.setattr(
        runner,
        "prepare_low_fraction_checkpoint",
        lambda *args, **kwargs: touched.append("asset"),
    )
    base = Namespace(
        policy=POLICY_PATH,
        asset=policy.assets[0].run_id,
        authorization=authorization,
        implementation_commit=IMPLEMENTATION_COMMIT,
        split="test",
        output_run_id=OUTPUT_RUN_ID,
        repository=ROOT,
        preflight=tmp_path / "preflight",
        validation_dir=tmp_path / "validation",
        validation_gate=tmp_path / "validation/validation_gate.json",
        datastore_dir=tmp_path / "datastore",
        output=tmp_path / "output",
    )
    monkeypatch.setattr(
        runner,
        "_validate_repository",
        lambda *args, **kwargs: (_ for _ in ()).throw(control.S4BControlError("bad HEAD")),
    )
    with pytest.raises(control.S4BControlError, match="bad HEAD"):
        runner.worker(base)
    assert touched == []
    monkeypatch.setattr(runner, "_validate_repository", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_server_preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner,
        "_verify_validation_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            control.S4BControlError("bad validation")
        ),
    )
    with pytest.raises(control.S4BControlError, match="bad validation"):
        runner.worker(base)
    assert touched == []


def test_test_run_and_finalize_reject_validation_before_execution_or_packaging(
    policy, tmp_path, monkeypatch
):
    authorization = tmp_path / "authorization.json"
    runner.write_json(authorization, fixture_authorization(policy, fixture_only=False))
    monkeypatch.setattr(runner, "_validate_repository", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_server_preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner,
        "_verify_validation_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            control.S4BControlError("validation replay rejected")
        ),
    )
    touched = []
    monkeypatch.setattr(
        runner,
        "_gpu_uuid_if_idle",
        lambda *args, **kwargs: touched.append("gpu"),
    )
    common = {
        "policy": POLICY_PATH,
        "authorization": authorization,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "output_run_id": OUTPUT_RUN_ID,
        "repository": ROOT,
        "preflight": tmp_path / "preflight",
        "datastore_dir": tmp_path / "datastore",
        "validation_dir": tmp_path / "validation",
        "validation_gate": tmp_path / "validation/validation_gate.json",
    }
    with pytest.raises(control.S4BControlError, match="validation replay"):
        runner.run_batch(
            Namespace(
                **common,
                split="test",
                gpus=["0"],
                output=tmp_path / "run-output",
            )
        )
    assert touched == []
    monkeypatch.setattr(
        runner,
        "_verify_worker_output",
        lambda *args, **kwargs: touched.append("test-artifact"),
    )
    with pytest.raises(control.S4BControlError, match="validation replay"):
        runner.finalize(
            Namespace(
                **common,
                test_run_dir=tmp_path / "test-run",
                core_zip=tmp_path / "core.zip",
                analysis_manifest=tmp_path / "analysis.json",
                output=tmp_path / "final-output",
            )
        )
    assert touched == []
