"""D8 functional source forgetting evaluator (plan §21-§31, §65-§66; fifth D8 review §27-§43, §56-§59).

Given a fine-tuned Human3 run (its final *_last.pt backbone) and the MATCHED
Animal56 teacher run, evaluate:

    teacher backbone + ORIGINAL animal heads    ->  R_animal^teacher
    fine-tuned backbone + ORIGINAL animal heads ->  R_animal^ft

on the Animal56 VALIDATION split (56 tasks, macro RMSE in raw label units).

Diagnostic only (plan §22): the outputs must never feed checkpoint selection,
LR schedules, early stopping or hyperparameter tuning.  For an S1 run the
backbone is bitwise identical to the teacher's, so
functional_forgetting_abs must be exactly 0 (< 1e-7, plan §25).

Provenance contract (fifth D8 review P0-6, §29-§34/§56/§58-§59): the teacher
is not trusted from the CLI.  The evaluator derives it from the candidate's
init-artifact provenance sidecar and verifies:

- run_metadata.d7_artifact_contract present, mode == b1, seed consistent
- teacher_dir equals provenance.teacher_real_run_dir (explicit overrides must
  exact-match)
- teacher checkpoint SHA256 equals provenance.teacher_real_checkpoint_sha256
- teacher reproducibility.base_seed == init human_seed == run seed
- run checkpoint data identity (manifest / datastore / feature schema) equals
  the contract recorded in run metadata

Writes `<run_dir>/functional_forgetting.json` with ``provenance_verified:
true`` — the D8 aggregator refuses unverified files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from d6_prep_inits import (  # noqa: E402
    _build_model,
    _load_teacher_checkpoint,
    _sha256_file,
)
from d8_retention import (  # noqa: E402
    build_hybrid_model,
    evaluate_animal_rmse,
    functional_forgetting_stats,
)

# Result-correction review P0-4: bump whenever the evaluator or its
# dependencies change; downstream merges refuse files with a different
# version (see d8_label_scaling_mechanism.ff_provenance_ok).
EVALUATOR_VERSION = "d8_functional_forgetting/1.1-provenance"


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except Exception:
        return ""


def _resolve_b1_artifact_contract(metadata: dict) -> tuple[dict, str]:
    """Seventh-D8 review P0-1 (§13-§16): two legitimate consumers — the
    standard D6 B1 reference (d6_artifact_contract) and D7/D8 candidates
    (d7_artifact_contract).  Both MUST carry mode == b1."""

    d7_contract = metadata.get("d7_artifact_contract")
    if isinstance(d7_contract, dict) and d7_contract.get("mode") == "b1":
        return d7_contract, "d7_artifact_contract"

    d6_contract = metadata.get("d6_artifact_contract")
    if (
        metadata.get("d6_candidate") == "b1"
        and isinstance(d6_contract, dict)
        and d6_contract.get("mode") == "b1"
    ):
        return d6_contract, "d6_artifact_contract"

    raise ValueError(
        "functional forgetting requires a provenance-verified B1 initialization "
        "contract from either d7_artifact_contract or the B1 d6_artifact_contract"
    )


def verify_run_provenance(run_dir: Path, teacher_dir: str | None = None) -> dict:
    """Bind the evaluator to the candidate's recorded matched teacher.

    Raises (ValueError / FileNotFoundError / RuntimeError) BEFORE any model
    evaluation when the run, its init provenance and the requested teacher do
    not agree."""

    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    args_path = run_dir / "args.json"
    for path in (metadata_path, args_path):
        if not path.is_file():
            raise FileNotFoundError(f"run dir is missing {path.name}: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args_payload = json.loads(args_path.read_text(encoding="utf-8"))

    contract, contract_source = _resolve_b1_artifact_contract(metadata)
    run_seed = metadata.get("seed")
    if int(contract.get("human_seed", -1)) != int(run_seed):
        raise ValueError("artifact contract human_seed does not match the run seed")

    init_state_path = args_payload.get("init_state_path")
    if not init_state_path:
        raise ValueError("args.json lacks init_state_path — cannot bind the teacher")
    init_state_path = Path(init_state_path)
    sidecar_path = Path(str(init_state_path) + ".provenance.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"init provenance sidecar missing: {sidecar_path}")
    provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for key in (
        "teacher_real_run_dir",
        "teacher_real_checkpoint_sha256",
        "expected_teacher_epoch",
        "teacher_initial_model_sha256",
        "human_seed",
        "output_sha256",
    ):
        if not provenance.get(key):
            raise ValueError(f"init provenance lacks required field {key!r}")

    recorded_teacher_dir = Path(provenance["teacher_real_run_dir"]).resolve()
    if teacher_dir is not None:
        requested_teacher_dir = Path(teacher_dir).resolve()
        if requested_teacher_dir != recorded_teacher_dir:
            raise ValueError(
                f"--teacher_dir {requested_teacher_dir} does not match the init "
                f"provenance teacher {recorded_teacher_dir} (sixth review §31)"
            )
    init_sha = _sha256_file(init_state_path)
    if init_sha != provenance["output_sha256"]:
        raise ValueError(
            "init artifact was modified after training: current sha256 "
            f"{init_sha} != provenance {provenance['output_sha256']}"
        )

    return {
        "run_dir": run_dir,
        "run_seed": int(run_seed),
        "contract": contract,
        "contract_source": contract_source,
        "provenance": provenance,
        "teacher_dir": recorded_teacher_dir,
        "expected_teacher_epoch": int(provenance["expected_teacher_epoch"]),
        "init_state_path": str(init_state_path),
        "artifact_sha256": provenance["output_sha256"],
    }


def verify_teacher_bundle(bundle: dict, teacher_path: Path, teacher_payload: dict) -> None:
    """Checkpoint-level provenance checks (§32-§33): sha256 binding and
    teacher base-seed consistency — must run before any model evaluation."""

    teacher_sha = _sha256_file(teacher_path)
    if teacher_sha != bundle["provenance"]["teacher_real_checkpoint_sha256"]:
        raise ValueError(
            "teacher checkpoint sha256 does not match the candidate's init "
            f"provenance ({teacher_sha} != "
            f"{bundle['provenance']['teacher_real_checkpoint_sha256']}) "
            "— refusing to evaluate against the wrong teacher"
        )
    teacher_base_seed = (teacher_payload.get("reproducibility") or {}).get("base_seed")
    if teacher_base_seed is None or int(teacher_base_seed) != bundle["run_seed"]:
        raise ValueError(
            f"teacher base_seed {teacher_base_seed!r} does not match the candidate "
            f"run seed {bundle['run_seed']}"
        )


def _load_run_backbone(run_dir: Path, contract: dict):
    matches = sorted(Path(run_dir).glob("*_last.pt"))
    if len(matches) != 1:
        # Fifth-D8 review P1-5 (§56): never silently pick the first file.
        raise RuntimeError(
            f"expected exactly one *_last.pt in {run_dir}, found {len(matches)}"
        )
    run_payload = torch.load(matches[0], map_location="cpu", weights_only=False)
    # Fifth-D8 review P1-6 (§58): the run checkpoint must carry the SAME data
    # identity as the contract recorded in run metadata.
    identity_checks = {
        "split_manifest_hash": (
            run_payload.get("split_manifest_hash"),
            contract.get("split_manifest_hash"),
        ),
        "datastore_fingerprint": (
            (run_payload.get("data_config") or {}).get("datastore_fingerprint"),
            contract.get("datastore_fingerprint"),
        ),
        "feature_schema_version": (
            run_payload.get("feature_schema_version"),
            contract.get("feature_schema_version"),
        ),
    }
    for field, (observed, expected) in identity_checks.items():
        if observed != expected:
            raise ValueError(
                f"run checkpoint {field} {observed!r} does not match the "
                f"contract {expected!r} — the backbone cannot be evaluated "
                "under a different data identity"
            )
    backbone = {
        key: value
        for key, value in run_payload["model_state"].items()
        if key.startswith("encoder.backbone.")
    }
    if not backbone:
        raise RuntimeError("run checkpoint contains no encoder.backbone.* tensors")
    return backbone, matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="fine-tuned Human3 run dir (with *_last.pt)")
    parser.add_argument(
        "--teacher_dir",
        default=None,
        help="optional; defaults to the teacher recorded in the candidate's "
        "init provenance — an explicit value must exact-match it",
    )
    parser.add_argument(
        "--expected_teacher_epoch",
        type=int,
        default=None,
        help="optional; defaults to the epoch recorded in the init provenance",
    )
    parser.add_argument("--gpu_id", default="cpu")
    parser.add_argument(
        "--output_json",
        default=None,
        help="defaults to <run_dir>/functional_forgetting.json",
    )
    args = parser.parse_args()

    device = (
        torch.device("cpu")
        if str(args.gpu_id) == "cpu"
        else torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    )

    bundle = verify_run_provenance(Path(args.run_dir), args.teacher_dir)
    run_dir = bundle["run_dir"]
    provenance = bundle["provenance"]
    contract = bundle["contract"]
    teacher_dir = bundle["teacher_dir"]
    expected_epoch = bundle["expected_teacher_epoch"]
    if args.expected_teacher_epoch is not None and int(args.expected_teacher_epoch) != expected_epoch:
        raise ValueError(
            f"--expected_teacher_epoch {args.expected_teacher_epoch} contradicts the "
            f"init provenance epoch {expected_epoch}"
        )

    teacher_path, teacher_payload = _load_teacher_checkpoint(teacher_dir, expected_epoch)
    verify_teacher_bundle(bundle, teacher_path, teacher_payload)

    template_config = dict(teacher_payload.get("configuration") or {})
    task_scalers = dict(teacher_payload.get("task_scalers") or {})
    animal_tasks = list(teacher_payload.get("task_names") or [])

    teacher_model = _build_model(bundle["run_seed"], "animal56", device, template_config)[0]
    teacher_model.load_state_dict(teacher_payload["model_state"], strict=True)
    teacher_model.to(device)
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    backbone, run_checkpoint = _load_run_backbone(run_dir, contract)
    hybrid = build_hybrid_model(teacher_model, backbone)
    hybrid.to(device)

    from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
    from main import build_parser, validate_params

    params = build_parser().parse_args([])
    for key in (
        "data_store_dir", "split_seed", "splitting", "vs", "calibration_size", "ts",
        "hidden_dim", "mid_dim", "a_layers", "a_heads", "edge_bias_mode",
        "spatial_pos_clip", "max_nodes_filter", "prediction_mode",
        "head_hidden_dim", "head_dropout",
    ):
        if key in template_config:
            setattr(params, key, template_config[key])
    setattr(params, "arch", "Graphormer")
    setattr(params, "toxacute_task_scope", "animal56")
    setattr(params, "seed", bundle["run_seed"])
    setattr(params, "fit_conformal", False)
    setattr(params, "train_eval_scope", "validation_only")
    setattr(params, "card_lambda_delta", 0.0)
    validate_params(params)

    from main import _loaders
    from dataset import DataCollator

    collator = DataCollator(
        spatial_pos_max_clip=params.spatial_pos_clip,
        max_node_filter=None,
    )
    loaders = _loaders(params, list(ANIMAL_SOURCE_TASKS), collator)
    batches = []
    for task in ANIMAL_SOURCE_TASKS:
        loader = loaders.get("val", loaders.get("validation", {})).get(task)
        if loader is None:
            raise RuntimeError(f"animal56 validation loader missing for {task!r}")
        for batch in loader:
            if getattr(batch, "get", lambda *_: None)("is_empty", False):
                continue
            # P1-3 (§58-§60): keep batches on CPU here — evaluate_animal_rmse
            # moves each batch to the device and back, so the full Animal56
            # validation set never resides on the GPU at once.
            batches.append((task, batch))
    if not batches:
        raise RuntimeError("animal56 validation produced no batches")

    teacher_macro, teacher_per_task = evaluate_animal_rmse(
        teacher_model, batches, device, task_scalers, ANIMAL_SOURCE_TASKS
    )
    current_macro, current_per_task = evaluate_animal_rmse(
        hybrid, batches, device, task_scalers, ANIMAL_SOURCE_TASKS
    )
    stats = functional_forgetting_stats(
        teacher_per_task,
        current_per_task,
        expected_task_set=set(ANIMAL_SOURCE_TASKS),
    )
    stats.update(
        {
            "run_dir": str(run_dir),
            "run_checkpoint": str(run_checkpoint),
            "teacher_checkpoint": str(teacher_path),
            # Fifth-D8 review §34: full provenance block.
            "provenance_verified": True,
            "artifact_contract_source": bundle["contract_source"],
            "teacher_checkpoint_sha256": _sha256_file(teacher_path),
            "teacher_initial_model_sha256": provenance["teacher_initial_model_sha256"],
            "candidate_run_seed": bundle["run_seed"],
            "split_manifest_hash": contract.get("split_manifest_hash"),
            "datastore_fingerprint": contract.get("datastore_fingerprint"),
            "feature_schema_version": contract.get("feature_schema_version"),
            "artifact_sha256": bundle["artifact_sha256"],
            "n_animal_tasks": len(ANIMAL_SOURCE_TASKS),
            "per_task_rmse_teacher": teacher_per_task,
            "per_task_rmse_current": current_per_task,
            # Result-correction review P0-4: artifact provenance so stale
            # results from a different commit / checkpoint / evaluator can
            # never be silently merged downstream.
            "checkpoint_sha256": _sha256_file(run_checkpoint),
            "git_commit": _git_commit(),
            "evaluator_version": EVALUATOR_VERSION,
            "evaluator_script_sha256": _sha256_file(Path(__file__)),
            "animal_manifest_hash": contract.get("split_manifest_hash"),
        }
    )

    output_json = Path(args.output_json) if args.output_json else run_dir / "functional_forgetting.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_json}")
    print(
        f"functional forgetting: teacher={teacher_macro:.4f} current={current_macro:.4f} "
        f"abs={stats['functional_forgetting_abs']:.4f} "
        f"relative={stats['functional_forgetting_relative']:.4f} "
        f"exact_zero={stats['functional_forgetting_exact_zero']} "
        f"provenance_verified=True"
    )


if __name__ == "__main__":
    main()
