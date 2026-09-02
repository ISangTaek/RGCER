"""D8 functional source forgetting evaluator (plan §21-§31, §65-§66).

Given a fine-tuned Human3 run (its final *_last.pt backbone) and the matched
Animal56 teacher run, evaluate:

    teacher backbone + ORIGINAL animal heads   ->  R_animal^teacher
    fine-tuned backbone + ORIGINAL animal heads ->  R_animal^ft

on the Animal56 VALIDATION split (56 tasks, macro RMSE in raw label units).

Diagnostic only (plan §22): the outputs must never feed checkpoint selection,
LR schedules, early stopping or hyperparameter tuning.  For an S1 run the
backbone is bitwise identical to the teacher's, so
functional_forgetting_abs must be exactly 0 (< 1e-7, plan §25).

Writes `<run_dir>/functional_forgetting.json` (consumed by the D8 aggregator).
"""

from __future__ import annotations

import argparse
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
)
from d8_retention import (  # noqa: E402
    build_hybrid_model,
    evaluate_animal_rmse,
    functional_forgetting_stats,
)


def _animal_validation_batches(params, animal_tasks, device):
    from main import _loaders
    from dataset import DataCollator

    collator = DataCollator(
        spatial_pos_max_clip=params.spatial_pos_clip,
        max_node_filter=None,
    )
    loaders = _loaders(params, list(animal_tasks), collator)
    batches = []
    for task in animal_tasks:
        loader = loaders.get("validation", {}).get(task)
        if loader is None:
            raise RuntimeError(f"animal56 validation loader missing for {task!r}")
        for batch in loader:
            if getattr(batch, "get", lambda *_: None)("is_empty", False):
                continue
            batches.append((task, batch.to(device)))
    if not batches:
        raise RuntimeError("animal56 validation produced no batches")
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="fine-tuned Human3 run dir (with *_last.pt)")
    parser.add_argument("--teacher_dir", required=True, help="matched Animal56 teacher run dir")
    parser.add_argument("--expected_teacher_epoch", type=int, required=True)
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

    run_dir = Path(args.run_dir)
    teacher_dir = Path(args.teacher_dir)
    teacher_path, teacher_payload = _load_teacher_checkpoint(
        teacher_dir, args.expected_teacher_epoch
    )
    template_config = dict(teacher_payload.get("configuration") or {})
    task_scalers = dict(teacher_payload.get("task_scalers") or {})
    animal_tasks = list(teacher_payload.get("task_names") or [])

    teacher_model = _build_model(
        int(template_config.get("seed", 42)), "animal56", device, template_config
    )[0]
    teacher_model.load_state_dict(teacher_payload["model_state"], strict=True)
    teacher_model.to(device)
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    run_checkpoint = sorted(run_dir.glob("*_last.pt"))
    if not run_checkpoint:
        raise FileNotFoundError(f"no *_last.pt under {run_dir}")
    run_payload = torch.load(run_checkpoint[0], map_location="cpu", weights_only=False)
    run_backbone = {
        key: value
        for key, value in run_payload["model_state"].items()
        if key.startswith("encoder.backbone.")
    }

    hybrid = build_hybrid_model(teacher_model, run_backbone)
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
    setattr(params, "seed", int(template_config.get("seed", 42)))
    setattr(params, "fit_conformal", False)
    setattr(params, "train_eval_scope", "validation_only")
    setattr(params, "card_lambda_delta", 0.0)
    validate_params(params)
    batches = _animal_validation_batches(params, ANIMAL_SOURCE_TASKS, device)

    teacher_macro, teacher_per_task = evaluate_animal_rmse(
        teacher_model, batches, device, task_scalers, ANIMAL_SOURCE_TASKS
    )
    current_macro, current_per_task = evaluate_animal_rmse(
        hybrid, batches, device, task_scalers, ANIMAL_SOURCE_TASKS
    )
    stats = functional_forgetting_stats(teacher_per_task, current_per_task)
    stats.update(
        {
            "run_dir": str(run_dir),
            "run_checkpoint": str(run_checkpoint[0]),
            "run_checkpoint_sha256_note": "see checkpoint_manifest.json for the hash",
            "teacher_checkpoint": str(teacher_path),
            "run_git_note": "git commit is recorded in run_metadata.json",
            "split_seed": template_config.get("split_seed"),
            "n_animal_tasks": len(ANIMAL_SOURCE_TASKS),
            "per_task_rmse_teacher": teacher_per_task,
            "per_task_rmse_current": current_per_task,
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
        f"exact_zero={stats['functional_forgetting_exact_zero']}"
    )


if __name__ == "__main__":
    main()
