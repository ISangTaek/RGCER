"""Synthetic tests for the CPU-only S4E source asset collector.

No real project checkpoint, prediction, label table, or server path is read.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "collect_s4e_source_assets.py"
HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import s4e_source_assets as collector  # noqa: E402
from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS  # noqa: E402


ANIMAL_TASKS = list(ANIMAL_SOURCE_TASKS)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_checkpoint(path: Path, value) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return file_sha(path), path.stat().st_size


def git_init(repository: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "s4e@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "S4E Synthetic"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", "synthetic"], cwd=repository, check=True)


def adaptation_config(method: str) -> dict:
    return {
        "d6_candidate": "b1" if method == "B1" else "none",
        "d7_candidate": "none" if method == "B1" else "s1",
        "freeze_backbone_epochs": 0 if method == "B1" else 40,
        "backbone_lr_multiplier": 1.0,
        "trainable_last_blocks": 0,
        "retention_probe_per_task": 16,
        "retention_damage_threshold": 0.02,
        "feature_drift_probe_size": 128,
        "feature_drift_epochs": "0,5,10,15,19",
    }


def teacher_payload(seed: int, epoch: int, *, missing_head=False, missing_scaler=False):
    state = {
        "encoder.backbone.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "encoder.backbone.running_mean": torch.tensor([0.0, 1.0]),
        "encoder.readout.weight": torch.tensor([[5.0, 6.0]]),
    }
    for index, task in enumerate(ANIMAL_TASKS):
        if missing_head and index == 0:
            continue
        state[f"decoders.{task}.weight"] = torch.tensor([[float(index)]])
    scalers = {
        task: {"mean": float(index), "std": float(index + 1), "count": index + 2}
        for index, task in enumerate(ANIMAL_TASKS)
    }
    if missing_scaler:
        scalers.pop(ANIMAL_TASKS[0])
    return {
        "checkpoint_version": 6,
        "epoch": epoch,
        "model_state": state,
        "task_names": list(ANIMAL_TASKS),
        "task_scalers": scalers,
        "configuration": {
            "arch": "Graphormer",
            "dataset": "toxacute",
            "splitting": "scaffold",
            "split_seed": 42,
            "task_sampling": "proportional",
            "train_fraction": 1.0,
            "train_eval_scope": "validation_only",
            "fit_conformal": False,
        },
        "architecture_config": {
            "hidden_dim": 2,
            "a_layers": 1,
            "task_names": list(ANIMAL_TASKS),
        },
        "data_config": {
            "datastore_fingerprint": "d" * 64,
            "max_nodes_filter": 512,
        },
        "split_manifest_hash": "e" * 64,
        "feature_schema_version": "synthetic-v1",
        "reproducibility": {"base_seed": seed},
        "optimizer_state": {"must_not_be_projected": torch.tensor([99.0])},
    }


def init_payload():
    return {
        "encoder.backbone.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "encoder.backbone.running_mean": torch.tensor([0.0, 1.0]),
        "encoder.readout.weight": torch.tensor([[5.0, 6.0]]),
    }


def build_synthetic_repository(tmp_path: Path):
    repository = tmp_path / "仓库 含空格"
    repository.mkdir()
    git_init(repository)
    configs = repository / "配置 锁"
    low_path = configs / "low policy.json"
    core_path = configs / "core lock.json"

    sources = {}
    for context in ("low", "core"):
        for seed in range(42, 47):
            epoch = 29 if context == "low" and seed == 42 else 39
            source_dir = repository / "资产" / context / f"seed_{seed}"
            teacher_path = source_dir / f"teacher_{epoch}.pt"
            teacher_sha, teacher_size = save_checkpoint(
                teacher_path,
                teacher_payload(seed, epoch),
            )
            init_path = source_dir / "初始化 state.pt"
            init_sha, init_size = save_checkpoint(init_path, init_payload())
            sidecar_path = Path(str(init_path) + ".provenance.json")
            write_json(
                sidecar_path,
                {
                    "mode": "b1",
                    "human_seed": seed,
                    "expected_teacher_epoch": epoch,
                    "teacher_real_run_dir": str(source_dir),
                    "teacher_real_checkpoint": str(teacher_path),
                    "teacher_real_checkpoint_sha256": teacher_sha,
                    "teacher_initial_model_sha256": "f" * 64,
                    "output": str(init_path),
                    "output_sha256": init_sha,
                },
            )
            sources[(context, seed)] = {
                "teacher": teacher_path,
                "teacher_sha": teacher_sha,
                "teacher_size": teacher_size,
                "init": init_path,
                "init_sha": init_sha,
                "init_size": init_size,
                "sidecar": sidecar_path,
                "epoch": epoch,
            }

    low_assets = []
    for method in ("B1", "RPT"):
        for fraction in (10, 25, 50, 75):
            for seed in range(42, 47):
                run_id = f"scaling_{method}_f{fraction}_s{seed}"
                run_dir = repository / "运行 目录" / run_id
                source = sources[("low", seed)]
                adaptation = adaptation_config(method)
                contract_name = (
                    "d6_artifact_contract" if method == "B1" else "d7_artifact_contract"
                )
                artifact_contract = {
                    "artifact_sha256": source["init_sha"],
                    "expected_teacher_epoch": source["epoch"],
                    "human_seed": seed,
                    "output": str(source["init"]),
                    "provenance_path": str(source["sidecar"]),
                }
                best_path = run_dir / f"{run_id}_best.pt"
                best_sha, best_size = save_checkpoint(
                    best_path,
                    {
                        "epoch": fraction // 25,
                        "model_state": {"unused": torch.tensor([1.0])},
                        "task_names": list(HUMAN_TASKS),
                        "configuration": {
                            **adaptation,
                            "init_state_path": str(source["init"]),
                            "seed": seed,
                            "dataset": "toxacute",
                            "splitting": "scaffold",
                            "split_seed": 42,
                            "train_fraction": fraction / 100.0,
                            "task_sampling": "proportional",
                            "train_eval_scope": "validation_only",
                            "fit_conformal": False,
                        },
                        "architecture_config": dict(adaptation),
                        "data_config": {
                            "datastore_fingerprint": "d" * 64,
                            "max_nodes_filter": 512,
                        },
                        "split_manifest_hash": "e" * 64,
                        "feature_schema_version": "synthetic-v1",
                    },
                )
                write_json(
                    run_dir / "args.json",
                    {
                        **adaptation,
                        "init_state_path": str(source["init"]),
                        "seed": seed,
                        "train_fraction": fraction / 100.0,
                        contract_name: artifact_contract,
                    },
                )
                write_json(
                    run_dir / "run_metadata.json",
                    {
                        **adaptation,
                        "seed": seed,
                        "train_fraction": fraction / 100.0,
                        "init_overlay": {
                            "init_state_path": str(source["init"]),
                            "init_state_sha256": source["init_sha"],
                        },
                        contract_name: artifact_contract,
                    },
                )
                low_assets.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "seed": seed,
                        "fraction_percent": fraction,
                        "run_dir": str(run_dir),
                        "checkpoint": {
                            "path": str(best_path),
                            "sha256": best_sha,
                            "size_bytes": best_size,
                        },
                        "best_epoch": fraction // 25,
                        "migration_required": False,
                        "expected_configuration": {"init_state_path": str(source["init"])},
                        "expected_adaptation": adaptation,
                    }
                )

    core_assets = []
    for method in ("B0", "B1", "RPT"):
        for seed in range(42, 47):
            run_id = f"D8_formal_{method}_s{seed}"
            if method == "B0":
                core_assets.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "seed": seed,
                        "run_dir": str(repository / "unused_b0" / run_id),
                        "best_checkpoint": {
                            "path": str(repository / "unused_b0" / f"{run_id}.pt"),
                            "sha256": "0" * 64,
                            "size_bytes": 1,
                        },
                        "best_epoch": 0,
                        "config": {"init_state_path": None},
                    }
                )
                continue
            run_dir = repository / "运行 目录" / run_id
            source = sources[("core", seed)]
            adaptation = adaptation_config(method)
            contract_name = (
                "d6_artifact_contract" if method == "B1" else "d7_artifact_contract"
            )
            artifact_contract = {
                "artifact_sha256": source["init_sha"],
                "expected_teacher_epoch": source["epoch"],
                "human_seed": seed,
                "output": str(source["init"]),
                "provenance_path": str(source["sidecar"]),
            }
            best_path = run_dir / f"{run_id}_best.pt"
            best_epoch = 0 if method == "B1" else seed - 40
            best_sha, best_size = save_checkpoint(
                best_path,
                {
                    "epoch": best_epoch,
                    "model_state": {"unused": torch.tensor([1.0])},
                    "task_names": list(HUMAN_TASKS),
                    "configuration": {
                        **adaptation,
                        "init_state_path": str(source["init"]),
                        "seed": seed,
                        "dataset": "toxacute",
                        "splitting": "scaffold",
                        "split_seed": 42,
                        "train_fraction": None,
                        "task_sampling": "proportional",
                        "train_eval_scope": "validation_only",
                        "fit_conformal": False,
                    },
                    "architecture_config": dict(adaptation),
                    "data_config": {
                        "datastore_fingerprint": "d" * 64,
                        "max_nodes_filter": 512,
                    },
                    "split_manifest_hash": "e" * 64,
                    "feature_schema_version": "synthetic-v1",
                },
            )
            write_json(
                run_dir / "args.json",
                {
                    **adaptation,
                    "init_state_path": str(source["init"]),
                    "seed": seed,
                    "train_fraction": None,
                    contract_name: artifact_contract,
                },
            )
            write_json(
                run_dir / "run_metadata.json",
                {
                    **adaptation,
                    "seed": seed,
                    "train_fraction": None,
                    "init_overlay": {
                        "init_state_path": str(source["init"]),
                        "init_state_sha256": source["init_sha"],
                    },
                    contract_name: artifact_contract,
                },
            )
            core_assets.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "best_checkpoint": {
                        "path": str(best_path),
                        "sha256": best_sha,
                        "size_bytes": best_size,
                    },
                    "best_epoch": best_epoch,
                    "config": {
                        **adaptation,
                        "init_state_path": str(source["init"]),
                    },
                }
            )

    low = {"schema_version": 1, "assets": low_assets}
    core = {"task_id": "synthetic", "assets": core_assets}
    write_json(low_path, low)
    write_json(core_path, core)
    input_files = [
        path
        for path in repository.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    original_hashes = {str(path): file_sha(path) for path in input_files}
    return {
        "repo": repository,
        "low": low_path,
        "core": core_path,
        "low_document": low,
        "core_document": core,
        "sources": sources,
        "original_hashes": original_hashes,
    }


def run_cli(fixture, output: Path):
    return subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(SCRIPT),
            "--repo-root",
            str(fixture["repo"]),
            "--low-policy",
            str(fixture["low"]),
            "--core-lock",
            str(fixture["core"]),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def assert_inputs_unchanged(fixture):
    assert {
        path: file_sha(Path(path)) for path in fixture["original_hashes"]
    } == fixture["original_hashes"]


def identity_documents(method: str, *, seed: int = 42, fraction: int = 10):
    adaptation = adaptation_config(method)
    args_document = {
        **adaptation,
        "seed": seed,
        "train_fraction": fraction / 100.0,
    }
    metadata = {
        **adaptation,
        "seed": seed,
        "train_fraction": fraction / 100.0,
    }
    best_payload = {
        "configuration": {
            **adaptation,
            "seed": seed,
            "train_fraction": fraction / 100.0,
        },
        "architecture_config": dict(adaptation),
    }
    return args_document, metadata, best_payload


def identity_spec(method: str, *, seed: int = 42, fraction: int = 10):
    return collector.RunSpec(
        run_id=f"identity_{method}_{seed}_{fraction}",
        method=method,
        seed=seed,
        fraction=fraction,
        run_dir="run",
        checkpoint_path="best.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_size_bytes=1,
        best_epoch=0,
        lock_init_state_path="init.pt",
        expected_adaptation=adaptation_config(method),
        lock_source="low_policy",
    )


@pytest.mark.parametrize("missing", ["train_fraction", "dataset"])
def test_shared_historical_teacher_missing_scope_preserves_facts(tmp_path, monkeypatch, missing):
    original = teacher_payload
    def historical(*a, **kw):
        payload = original(*a, **kw)
        payload["configuration"].pop(missing)
        return payload
    monkeypatch.setattr(sys.modules[__name__], "teacher_payload", historical)
    fixture = build_synthetic_repository(tmp_path)
    output = tmp_path / "historical-output"
    code = collector.collect_source_assets(fixture["repo"], fixture["low"], fixture["core"], output)
    read = lambda n: json.loads((output / n).read_text(encoding="utf-8"))
    links = read("run_source_links.json")
    assets = read("source_assets.json")
    assert len(links["runs"]) == 50
    assert len([x for x in assets["assets"] if x["kind"] == "teacher"]) == 6
    assert all("teacher_asset_id" in x and "best" in x and "init_path_verification" in x for x in links["runs"])
    if missing == "train_fraction":
        assert code == 0 and not links["issues"]
        unknowns = read("collection_manifest.json")["scientific_unknowns"]
        assert len(unknowns) == 6
        assert sum(len(x["affected_run_ids"]) for x in unknowns) == 50
        assert all(x["value"] is None for x in unknowns)
        assert all(x["scope"]["fields"]["train_fraction"]["status"] == "UNKNOWN"
                   for x in read("training_scope_evidence.json")["sources"])
    else:
        assert code == 2
        assert {x["code"] for x in links["issues"]} == {"TRAINING_SCOPE_INCOMPLETE"}
        assert all("dataset" in x["reason"] for x in links["issues"])
    collector.verify_checksum_manifest(output)


def test_source_fraction_unknown_exception_does_not_allow_invalid_or_target():
    payload = teacher_payload(42, 29)
    payload["configuration"].pop("train_fraction")
    scope = collector.project_scope_evidence([("checkpoint", payload)])
    collector.require_complete_scope(scope, "teacher", source_teacher=True)
    with pytest.raises(collector.RunCollectionError):
        collector.require_complete_scope(scope, "target")
    payload["configuration"]["train_fraction"] = "unknown"
    with pytest.raises(collector.RunCollectionError):
        collector.require_complete_scope(collector.project_scope_evidence([("checkpoint", payload)]), "teacher", source_teacher=True)
    payload["configuration"]["train_fraction"] = 1.0
    with pytest.raises(collector.RunCollectionError):
        collector.require_complete_scope(collector.project_scope_evidence([("checkpoint", payload), ("args", {"train_fraction": 0.5})]), "teacher", source_teacher=True)


def test_evidence_copy_reuse_requires_same_source_and_bytes(tmp_path):
    src = tmp_path / "source.json"
    write_json(src, {"a": 1})
    writer = collector.EvidenceWriter(tmp_path / "evidence")
    first = writer.copy(src, "teacher/args.json", parsed={"a": 1}, category="teacher")
    second = writer.copy(src, "teacher/args.json", parsed={"a": 1}, category="teacher")
    assert first["sha256"] == second["sha256"] and second["reused"]
    other = tmp_path / "other.json"
    write_json(other, {"a": 1})
    with pytest.raises(collector.S4ECollectionError, match="identity changed"):
        writer.copy(other, "teacher/args.json", parsed={"a": 1}, category="teacher")
    write_json(src, {"a": 2})
    with pytest.raises(collector.S4ECollectionError, match="bytes changed"):
        writer.copy(src, "teacher/args.json", parsed={"a": 2}, category="teacher")


def test_exact_matrix_excludes_b0_and_rejects_incomplete_or_illegal(tmp_path):
    fixture = build_synthetic_repository(tmp_path)
    specs = collector.build_run_specs(fixture["low_document"], fixture["core_document"])
    assert len(specs) == 50
    assert {row.method for row in specs} == {"B1", "RPT"}
    assert {(row.method, row.seed, row.fraction) for row in specs} == {
        (method, seed, fraction)
        for method in ("B1", "RPT")
        for seed in range(42, 47)
        for fraction in (10, 25, 50, 75, 100)
    }

    incomplete = json.loads(json.dumps(fixture["low_document"]))
    incomplete["assets"].pop()
    with pytest.raises(collector.S4ECollectionError, match="exact 40-run matrix"):
        collector.build_run_specs(incomplete, fixture["core_document"])

    illegal = json.loads(json.dumps(fixture["core_document"]))
    illegal["assets"][0]["method"] = "UNKNOWN"
    with pytest.raises(collector.S4ECollectionError, match="illegal matrix identity"):
        collector.build_run_specs(fixture["low_document"], illegal)


def test_low_and_core_specs_preserve_their_frozen_adaptation_sources(tmp_path):
    fixture = build_synthetic_repository(tmp_path)
    specs = collector.build_run_specs(fixture["low_document"], fixture["core_document"])
    low_rpt = next(
        row for row in specs if row.method == "RPT" and row.fraction == 10 and row.seed == 42
    )
    core_b1 = next(
        row for row in specs if row.method == "B1" and row.fraction == 100 and row.seed == 42
    )
    assert dict(low_rpt.expected_adaptation) == adaptation_config("RPT")
    core_lock_row = next(
        row
        for row in fixture["core_document"]["assets"]
        if row["method"] == "B1" and row["seed"] == 42
    )
    assert dict(core_b1.expected_adaptation) == {
        field: core_lock_row["config"][field]
        for field in collector.ADAPTATION_FIELDS
        if field in core_lock_row["config"]
    }


@pytest.mark.parametrize("method", ["B1", "RPT"])
def test_real_candidate_identity_mapping_accepts_b1_and_rpt(method):
    spec = identity_spec(method)
    args_document, metadata, best_payload = identity_documents(method)
    rows = collector.validate_run_identity_sources(
        spec,
        args_document,
        metadata,
        best_payload,
    )
    by_field = {(row["source"], row["field"]): row for row in rows}
    assert by_field[("args.d6_candidate", "d6_candidate")]["value"] == (
        "b1" if method == "B1" else "none"
    )
    assert by_field[("args.d7_candidate", "d7_candidate")]["value"] == (
        "none" if method == "B1" else "s1"
    )
    assert by_field[("args.freeze_backbone_epochs", "freeze_backbone_epochs")][
        "value"
    ] == (0 if method == "B1" else 40)


@pytest.mark.parametrize(
    ("source", "field", "wrong", "match"),
    [
        ("args", "d6_candidate", "b1", "frozen d6_candidate='none'"),
        ("run_metadata", "d7_candidate", "none", "frozen d7_candidate='s1'"),
        ("best_architecture", "d7_candidate", "none", "frozen d7_candidate='s1'"),
        (
            "best_configuration",
            "freeze_backbone_epochs",
            0,
            "frozen freeze_backbone_epochs=40",
        ),
    ],
)
def test_rpt_candidate_and_frozen_adaptation_conflicts_are_rejected(
    source,
    field,
    wrong,
    match,
):
    spec = identity_spec("RPT")
    args_document, metadata, best_payload = identity_documents("RPT")
    target = {
        "args": args_document,
        "run_metadata": metadata,
        "best_configuration": best_payload["configuration"],
        "best_architecture": best_payload["architecture_config"],
    }[source]
    target[field] = wrong
    with pytest.raises(collector.RunCollectionError, match=match):
        collector.validate_run_identity_sources(
            spec,
            args_document,
            metadata,
            best_payload,
        )


def test_deleting_persisted_adaptation_field_does_not_make_identity_pass():
    spec = identity_spec("RPT")
    args_document, metadata, best_payload = identity_documents("RPT")
    del args_document["d7_candidate"]
    with pytest.raises(collector.RunCollectionError, match="does not persist"):
        collector.validate_run_identity_sources(
            spec,
            args_document,
            metadata,
            best_payload,
        )


def test_cli_positive_collects_50_deduplicates_sources_and_preserves_inputs(tmp_path, monkeypatch):
    fixture = build_synthetic_repository(tmp_path)
    output = tmp_path / "输出 证据"

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("model forward is forbidden")

    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden_forward)
    code = collector.collect_source_assets(
        fixture["repo"], fixture["low"], fixture["core"], output
    )
    assert code == 0
    manifest = json.loads((output / "collection_manifest.json").read_text(encoding="utf-8"))
    links = json.loads((output / "run_source_links.json").read_text(encoding="utf-8"))
    assets = json.loads((output / "source_assets.json").read_text(encoding="utf-8"))
    training = json.loads((output / "training_scope_evidence.json").read_text(encoding="utf-8"))
    verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert len(links["runs"]) == 50
    assert all(row["status"] == "COLLECTED" for row in links["runs"])
    assert all(
        row["init_path_verification"]["status"] == "VERIFIED"
        and row["teacher_path_verification"]["status"] == "VERIFIED"
        for row in links["runs"]
    )
    assert any(
        row["init_path_verification"]["summary_reused"]
        for row in links["runs"]
    )
    assert len(
        {row["init_path_verification"]["path"] for row in links["runs"]}
    ) > 1
    assert {row["method"] for row in links["runs"]} == {"B1", "RPT"}
    # Every init has identical bytes, and low/core teachers for seeds 43-46
    # are byte-identical.  Deduplication is by exact asset SHA, not path.
    assert len(assets["assets"]) == 7  # one init plus six teacher identities
    assert len(assets["comparisons"]) == 6
    assert len(training["targets"]) == 50
    assert all(
        row["sample_or_subset_evidence"][0]["status"] == "NOT_PERSISTED"
        for row in training["targets"]
    )
    seed42_low = next(
        row for row in links["runs"] if row["method"] == "B1" and row["seed"] == 42 and row["fraction"] == 10
    )
    seed42_core = next(
        row for row in links["runs"] if row["method"] == "B1" and row["seed"] == 42 and row["fraction"] == 100
    )
    assert seed42_low["teacher_asset_id"] != seed42_core["teacher_asset_id"]
    teacher_by_id = {row["asset_id"]: row for row in assets["assets"] if row["kind"] == "teacher"}
    assert teacher_by_id[seed42_low["teacher_asset_id"]]["epoch"] == 29
    assert teacher_by_id[seed42_core["teacher_asset_id"]]["epoch"] == 39
    assert all(row["decoder_inventory"]["tensor_count"] == 56 for row in teacher_by_id.values())
    assert all(len(row["scalers"]) == 56 for row in teacher_by_id.values())
    assert manifest["acceptance_status"] == "PENDING_REVIEW"
    assert manifest["model_constructed"] is False
    assert manifest["forward_called"] is False
    assert manifest["optimizer_constructed"] is False
    assert manifest["training_called"] is False
    assert manifest["gpu_used"] is False
    listed_files = {row["path"] for row in manifest["files"]}
    assert {"collection_manifest.json", "checksums.sha256"} <= listed_files
    assert verification["all_structural_checks_pass"] is True
    assert verification["collection_complete"] is True
    checksum_lines = (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert checksum_lines
    assert all(not line.endswith("  checksums.sha256") for line in checksum_lines)
    assert_inputs_unchanged(fixture)


def test_cli_entrypoint_positive_with_spaces_and_chinese(tmp_path):
    fixture = build_synthetic_repository(tmp_path)
    output = tmp_path / "CLI 输出 空格"
    result = run_cli(fixture, output)
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["execution_status"] == "COMPLETE"
    assert response["acceptance_status"] == "PENDING_REVIEW"
    assert (output / "collection_manifest.json").is_file()
    assert_inputs_unchanged(fixture)


def test_cli_failure_missing_sidecar_returns_two_and_keeps_evidence(tmp_path):
    fixture = build_synthetic_repository(tmp_path)
    fixture["sources"][("low", 42)]["sidecar"].unlink()
    output = tmp_path / "CLI failure evidence"
    result = run_cli(fixture, output)
    assert result.returncode == 2, result.stderr
    response = json.loads(result.stdout)
    assert response["execution_status"] == "COMPLETE_WITH_ISSUES"
    links = json.loads((output / "run_source_links.json").read_text(encoding="utf-8"))
    affected = [row for row in links["runs"] if row["seed"] == 42 and row["fraction"] != 100]
    assert affected
    assert all(row["status"] == "ERROR" for row in affected)
    assert all(row["reason_code"] == "MISSING_METADATA" for row in affected)
    assert (output / "verification.json").is_file()
    assert (output / "checksums.sha256").is_file()


def test_sha_mismatch_blocks_deserialization(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    checkpoint = repository / "asset.pt"
    save_checkpoint(checkpoint, {"value": torch.tensor([1.0])})
    calls = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(collector.RunCollectionError, match="SHA-256 mismatch"):
        collector.load_verified_torch(
            checkpoint,
            "0" * 64,
            repository,
            label="synthetic",
        )
    assert calls == []


def test_real_artifact_sha_sources_bind_legacy_overlay_and_sidecar():
    digest = "a" * 64
    metadata = {
        "d6_artifact_contract": {"artifact_sha256": digest},
        "init_overlay": {
            "init_state_sha256": digest,
            "output_sha256": digest,
        },
    }
    args_document = {
        "d6_artifact_contract": {"artifact_sha256": digest},
        "init_overlay_provenance": {"init_state_sha256": digest},
    }
    sidecar = {"output_sha256": digest}
    resolved, rows = collector.resolve_init_sha_identity(
        metadata,
        sidecar,
        args_document,
    )
    assert resolved == digest
    assert {
        "args.d6_artifact_contract.artifact_sha256",
        "args.init_overlay_provenance.init_state_sha256",
        "run_metadata.d6_artifact_contract.artifact_sha256",
        "run_metadata.init_overlay.init_state_sha256",
        "run_metadata.init_overlay.output_sha256",
        "provenance.output_sha256",
    } <= {row["source"] for row in rows}

    metadata["d6_artifact_contract"]["artifact_sha256"] = "b" * 64
    with pytest.raises(collector.RunCollectionError, match="SHA references disagree"):
        collector.resolve_init_sha_identity(metadata, sidecar, args_document)


@pytest.mark.parametrize(
    ("source", "field", "wrong"),
    [
        ("args", "human_seed", 43),
        ("run_metadata", "expected_teacher_epoch", 30),
    ],
)
def test_artifact_contract_seed_and_teacher_epoch_conflicts_are_rejected(
    source,
    field,
    wrong,
):
    spec = identity_spec("B1")
    contract = {"human_seed": 42, "expected_teacher_epoch": 29}
    args_document = {"d6_artifact_contract": dict(contract)}
    metadata = {"d6_artifact_contract": dict(contract)}
    collector.validate_init_contract_identity(
        spec,
        args_document,
        metadata,
        {"human_seed": 42, "expected_teacher_epoch": 29},
    )
    target = args_document if source == "args" else metadata
    target["d6_artifact_contract"][field] = wrong
    with pytest.raises(collector.RunCollectionError, match="differs from expected"):
        collector.validate_init_contract_identity(
            spec,
            args_document,
            metadata,
            {"human_seed": 42, "expected_teacher_epoch": 29},
        )


def test_each_same_sha_path_is_verified_before_summary_reuse(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    first = repository / "first.pt"
    second = repository / "second.pt"
    first.write_bytes(b"same immutable bytes")
    second.write_bytes(first.read_bytes())
    digest = file_sha(first)
    first_result = collector.verify_scoped_asset_path(
        first,
        digest,
        repository,
        expected_size_bytes=first.stat().st_size,
        label="first init",
    )
    second_result = collector.verify_scoped_asset_path(
        second,
        digest,
        repository,
        expected_size_bytes=first_result["size_bytes"],
        label="second init",
    )
    assert first_result["sha256"] == second_result["sha256"] == digest

    second.unlink()
    with pytest.raises(collector.RunCollectionError, match="is missing"):
        collector.verify_scoped_asset_path(
            second,
            digest,
            repository,
            label="missing second init",
        )
    second.write_bytes(b"different bytes")
    with pytest.raises(collector.RunCollectionError) as mismatch:
        collector.verify_scoped_asset_path(
            second,
            digest,
            repository,
            label="changed second init",
            mismatch_code="INIT_SHA_CONFLICT",
            mismatch_status="CONFLICT",
        )
    assert mismatch.value.code == "INIT_SHA_CONFLICT"
    assert mismatch.value.status == "CONFLICT"

    outside = tmp_path / "outside.pt"
    outside.write_bytes(first.read_bytes())
    with pytest.raises(collector.RunCollectionError) as outside_error:
        collector.verify_scoped_asset_path(
            outside,
            digest,
            repository,
            label="outside init",
        )
    assert outside_error.value.code == "ASSET_OUTSIDE_REPOSITORY"


def test_cached_init_path_tamper_returns_two_and_marks_affected_runs(tmp_path):
    fixture = build_synthetic_repository(tmp_path)
    changed = fixture["sources"][("low", 43)]["init"]
    changed.write_bytes(b"tampered second path")
    output = tmp_path / "cached path failure"
    code = collector.collect_source_assets(
        fixture["repo"],
        fixture["low"],
        fixture["core"],
        output,
    )
    assert code == 2
    links = json.loads((output / "run_source_links.json").read_text(encoding="utf-8"))
    affected = [
        row
        for row in links["runs"]
        if row["fraction"] != 100 and row["seed"] == 43
    ]
    assert affected
    assert all(row["status"] == "CONFLICT" for row in affected)
    assert all(row["reason_code"] == "INIT_SHA_CONFLICT" for row in affected)


def test_teacher_directory_zero_or_multiple_sha_matches_is_rejected(tmp_path):
    repository = tmp_path / "repo"
    directory = repository / "teachers"
    directory.mkdir(parents=True)
    original = directory / "a.pt"
    digest, _ = save_checkpoint(original, teacher_payload(42, 29))
    duplicate = directory / "b.pt"
    duplicate.write_bytes(original.read_bytes())
    sidecar = {
        "teacher_real_run_dir": str(directory),
        "teacher_real_checkpoint_sha256": digest,
    }
    with pytest.raises(collector.RunCollectionError, match="matched 2 candidates"):
        collector.resolve_teacher_checkpoint(sidecar, repository)
    duplicate.unlink()
    sidecar["teacher_real_checkpoint_sha256"] = "0" * 64
    with pytest.raises(collector.RunCollectionError, match="matched 0 candidates"):
        collector.resolve_teacher_checkpoint(sidecar, repository)


@pytest.mark.parametrize(
    ("missing_head", "missing_scaler", "match"),
    [(True, False, "decoder task set"), (False, True, "scaler missing")],
)
def test_missing_decoder_or_scaler_is_not_defaulted(tmp_path, missing_head, missing_scaler, match):
    path = tmp_path / "teacher.pt"
    payload = teacher_payload(
        42,
        29,
        missing_head=missing_head,
        missing_scaler=missing_scaler,
    )
    save_checkpoint(path, payload)
    with pytest.raises(collector.RunCollectionError, match=match):
        collector.summarize_teacher_asset(
            path,
            file_sha(path),
            payload,
            {"load_mode": "synthetic"},
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("arbitrary_names", "TASK_SCOPE_MISMATCH"),
        ("human_replacement", "TASK_SCOPE_MISMATCH"),
        ("wrong_order", "TASK_ORDER_MISMATCH"),
        ("architecture_wrong_order", "ARCHITECTURE_TASK_SCOPE_MISMATCH"),
    ],
)
def test_teacher_is_bound_to_exact_canonical_animal56(tmp_path, mutation, match):
    payload = teacher_payload(42, 29)
    if mutation == "arbitrary_names":
        payload["task_names"] = [f"wrong_endpoint_{index:02d}" for index in range(56)]
    elif mutation == "human_replacement":
        payload["task_names"][0] = HUMAN_TASKS[0]
    elif mutation == "wrong_order":
        payload["task_names"][0], payload["task_names"][1] = (
            payload["task_names"][1],
            payload["task_names"][0],
        )
    else:
        architecture_tasks = payload["architecture_config"]["task_names"]
        architecture_tasks[0], architecture_tasks[1] = (
            architecture_tasks[1],
            architecture_tasks[0],
        )
    path = tmp_path / "teacher.pt"
    save_checkpoint(path, payload)
    with pytest.raises(collector.RunCollectionError) as error:
        collector.summarize_teacher_asset(
            path,
            file_sha(path),
            payload,
            {"load_mode": "synthetic"},
        )
    assert error.value.code == match


@pytest.mark.parametrize("kind", ["decoder", "scaler"])
def test_teacher_rejects_extra_decoder_or_scaler_task(tmp_path, kind):
    payload = teacher_payload(42, 29)
    if kind == "decoder":
        payload["model_state"]["decoders.extra_animal_task.weight"] = torch.tensor(
            [[1.0]]
        )
        expected_code = "DECODER_SCOPE_MISMATCH"
    else:
        payload["task_scalers"]["extra_animal_task"] = {
            "mean": 0.0,
            "std": 1.0,
        }
        expected_code = "SCALER_SCOPE_MISMATCH"
    path = tmp_path / "teacher.pt"
    save_checkpoint(path, payload)
    with pytest.raises(collector.RunCollectionError) as error:
        collector.summarize_teacher_asset(
            path,
            file_sha(path),
            payload,
            {"load_mode": "synthetic"},
        )
    assert error.value.code == expected_code


def test_backbone_comparison_reports_missing_shape_dtype_and_value():
    teacher = {
        "encoder.backbone.equal": torch.tensor([1.0]),
        "encoder.backbone.missing": torch.tensor([2.0]),
        "encoder.backbone.shape": torch.tensor([1.0, 2.0]),
        "encoder.backbone.dtype": torch.tensor([1.0], dtype=torch.float32),
        "encoder.backbone.value": torch.tensor([1.0]),
    }
    init = {
        "encoder.backbone.equal": torch.tensor([1.0]),
        "encoder.backbone.unexpected": torch.tensor([3.0]),
        "encoder.backbone.shape": torch.tensor([[1.0, 2.0]]),
        "encoder.backbone.dtype": torch.tensor([1.0], dtype=torch.float64),
        "encoder.backbone.value": torch.tensor([9.0]),
    }
    result = collector.compare_tensor_states(
        teacher,
        init,
        prefix=collector.BACKBONE_PREFIX,
    )
    statuses = {row["key"]: row["status"] for row in result["per_key"]}
    assert result["status"] == "CONFLICT"
    assert result["missing_in_init"] == ["encoder.backbone.missing"]
    assert result["unexpected_in_init"] == ["encoder.backbone.unexpected"]
    assert statuses["encoder.backbone.equal"] == "EQUAL"
    assert statuses["encoder.backbone.shape"] == "SHAPE_MISMATCH"
    assert statuses["encoder.backbone.dtype"] == "DTYPE_MISMATCH"
    assert statuses["encoder.backbone.value"] == "VALUE_MISMATCH"


def test_json_strictness_type_fidelity_and_unknown_projection(tmp_path):
    assert collector.json_safe(object())["value"] is None
    assert collector.json_safe(object())["reason"].startswith("unsupported_type:")
    with pytest.raises(collector.S4ECollectionError, match="non-finite"):
        collector.strict_json_loads('{"value": NaN}')
    with pytest.raises(collector.S4ECollectionError, match="duplicate JSON key"):
        collector.strict_json_loads('{"value": 1, "value": 2}')

    fixture = build_synthetic_repository(tmp_path)
    wrong_type = json.loads(json.dumps(fixture["low_document"]))
    wrong_type["assets"][0]["migration_required"] = "false"
    with pytest.raises(collector.S4ECollectionError, match="JSON boolean"):
        collector.build_run_specs(wrong_type, fixture["core_document"])


def test_existing_output_is_rejected_without_overwrite(tmp_path):
    fixture = build_synthetic_repository(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(collector.S4ECollectionError, match="already exists"):
        collector.collect_source_assets(fixture["repo"], fixture["low"], fixture["core"], output)
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"


def test_absolute_posix_path_is_not_rebased_on_windows(tmp_path):
    if os.name == "nt":
        with pytest.raises(collector.RunCollectionError, match="absolute Linux path"):
            collector.resolve_reference_path(
                "/home/example/RGCER/asset.pt",
                tmp_path,
                "asset",
            )
    else:
        assert collector.resolve_reference_path(
            "/home/example/RGCER/asset.pt",
            tmp_path,
            "asset",
        ) == Path("/home/example/RGCER/asset.pt")


def test_multiple_init_sources_must_resolve_to_one_identity(tmp_path):
    first = (tmp_path / "first.pt").resolve()
    second = (tmp_path / "second.pt").resolve()
    with pytest.raises(collector.RunCollectionError, match="references disagree"):
        collector.require_single_reference(
            [
                {"source": "args.init_state_path", "declared": str(first), "resolved": str(first)},
                {
                    "source": "run_metadata.init_overlay.output",
                    "declared": str(second),
                    "resolved": str(second),
                },
            ],
            "init",
        )


def test_training_scope_files_are_copied_and_reconstruction_is_labeled(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    runtime = repository / "runtime train samples.json"
    reconstructed = repository / "reconstructed train subset.tsv"
    label_free = repository / "train sample ids.csv"
    sensitive = repository / "secret train manifest.json"
    write_json(runtime, {"sample_ids": ["a", "b"], "labels": [7.25, 8.5]})
    reconstructed.write_text("sample_id\ttarget\na\t12345\n", encoding="utf-8")
    label_free.write_text("sample_id\na\n", encoding="utf-8")
    write_json(sensitive, {"access_token": "must-not-copy"})
    evidence = collector.EvidenceWriter(tmp_path / "output" / "evidence")
    rows = collector.collect_training_references(
        [
            (
                "run_metadata",
                {
                    "train_sample_list_path": str(runtime),
                    "reconstructed_train_subset_path": str(reconstructed),
                    "train_sample_ids_path": str(label_free),
                    "train_manifest_path": str(sensitive),
                },
            )
        ],
        repository,
        evidence,
        "synthetic_run",
    )
    by_path = {row["declared_path"]: row for row in rows}
    assert by_path[str(runtime)]["status"] == "COPIED_PROJECTED_WITHOUT_LABEL_VALUES"
    assert by_path[str(runtime)]["evidence_kind"] == "runtime_record"
    assert by_path[str(reconstructed)]["status"] == "COPIED_PROJECTED_WITHOUT_LABEL_VALUES"
    assert by_path[str(reconstructed)]["evidence_kind"] == "reconstructed"
    assert by_path[str(label_free)]["status"] == "COPIED"
    assert by_path[str(sensitive)]["status"] == "BLOCKED_SENSITIVE"
    runtime_projection = tmp_path / "output" / by_path[str(runtime)]["evidence_path"]
    runtime_payload = json.loads(runtime_projection.read_text(encoding="utf-8"))
    assert runtime_payload == {"sample_ids": ["a", "b"]}
    assert by_path[str(runtime)]["source_sha256"] == collector.sha256_file(runtime)
    reconstructed_projection = (
        tmp_path / "output" / by_path[str(reconstructed)]["evidence_path"]
    )
    assert reconstructed_projection.read_text(encoding="utf-8") == "sample_id\na\n"
    assert "12345" not in reconstructed_projection.read_text(encoding="utf-8")
    blocked_path = tmp_path / "output" / by_path[str(sensitive)]["evidence_path"]
    assert not blocked_path.exists()


def test_training_jsonl_projection_removes_nested_label_values(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "train sample manifest.jsonl"
    source.write_text(
        '{"sample_id":"a","y_true":31337}\n'
        '{"sample_id":"b","audit":{"predictions":[424242]}}\n',
        encoding="utf-8",
    )
    evidence = collector.EvidenceWriter(tmp_path / "output" / "evidence")
    rows = collector.collect_training_references(
        [("run_metadata", {"train_sample_manifest_path": str(source)})],
        repository,
        evidence,
        "synthetic_run",
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "COPIED_PROJECTED_WITHOUT_LABEL_VALUES"
    projection = tmp_path / "output" / rows[0]["evidence_path"]
    projection_text = projection.read_text(encoding="utf-8")
    assert "31337" not in projection_text
    assert "424242" not in projection_text
    assert [json.loads(line) for line in projection_text.splitlines()] == [
        {"sample_id": "a"},
        {"sample_id": "b", "audit": {}},
    ]


def test_checksum_manifest_excludes_itself_and_detects_tampering(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    payload = output / "payload.json"
    write_json(payload, {"value": 1})
    collector._write_checksum_manifest(output)  # noqa: SLF001
    collector.verify_checksum_manifest(output)
    assert "checksums.sha256" not in (output / "checksums.sha256").read_text(
        encoding="utf-8"
    )
    payload.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(collector.S4ECollectionError, match="checksum mismatch"):
        collector.verify_checksum_manifest(output)


def test_training_scope_conflict_and_missing_fields_fail_closed():
    complete = teacher_payload(42, 29)
    conflicting = {"data_config": {"datastore_fingerprint": "0" * 64}}
    scope = collector.project_scope_evidence(
        [("teacher_checkpoint", complete), ("run_metadata", conflicting)]
    )
    assert scope["status"] == "CONFLICT"
    assert scope["conflict_fields"] == ["datastore_fingerprint"]
    with pytest.raises(collector.RunCollectionError, match="training scope conflicts"):
        collector.require_complete_scope(scope, "teacher")

    missing = collector.project_scope_evidence([("empty", {})])
    assert missing["status"] == "INCOMPLETE"
    assert "task_names" in missing["missing_fields"]
    with pytest.raises(collector.RunCollectionError, match="not persisted"):
        collector.require_complete_scope(missing, "teacher")

    invalid_bool = teacher_payload(42, 29)
    invalid_bool["configuration"]["fit_conformal"] = "false"
    invalid_scope = collector.project_scope_evidence(
        [("teacher_checkpoint", invalid_bool)]
    )
    assert invalid_scope["fields"]["fit_conformal"]["status"] == "INVALID"
    with pytest.raises(collector.RunCollectionError, match="not persisted"):
        collector.require_complete_scope(invalid_scope, "teacher")


def test_teacher_and_best_metadata_reference_conflicts_are_rejected(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    sidecar = {
        "teacher_real_checkpoint_sha256": "a" * 64,
        "teacher_real_checkpoint": str(repository / "teacher.pt"),
        "teacher_real_run_dir": str(repository),
    }
    metadata = {
        "d7_artifact_contract": {
            "teacher_real_checkpoint_sha256": "b" * 64,
        }
    }
    with pytest.raises(collector.RunCollectionError, match="SHA-256 sources disagree"):
        collector.validate_teacher_reference_sources(metadata, sidecar, repository)

    spec = collector.RunSpec(
        run_id="run",
        method="B1",
        seed=42,
        fraction=10,
        run_dir=str(repository / "run"),
        checkpoint_path=str(repository / "locked.pt"),
        checkpoint_sha256="c" * 64,
        checkpoint_size_bytes=1,
        best_epoch=0,
        lock_init_state_path=str(repository / "init.pt"),
        expected_adaptation=adaptation_config("B1"),
        lock_source="synthetic",
    )
    with pytest.raises(collector.RunCollectionError, match="locked best checkpoint path"):
        collector.validate_best_references(
            spec,
            {"best_checkpoint_path": str(repository / "other.pt")},
            repository,
        )
