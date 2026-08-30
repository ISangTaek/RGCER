"""D4-0 inference-time causal interventions on Task-only best checkpoints.

Plan §10-§22: with the learned routing weights of each finished Task-only
Router run, replay the **validation** forward pass under interventions that
isolate where the gains live:

- normal            : untouched forward (reference)
- base_only         : routing disabled, final = base (no hook needed)
- route_only        : NULL forced 0
- null_one          : NULL forced 1 (must equal base_only exactly)
- uniform_null_kept : active conditional weights -> 1/k, learned NULL kept
- uniform_null_zero : ... and NULL forced 0
- perm_<s>          : learned weight values shuffled across active sources
- del_top1          : learned top-1 source removed, rest renormalised
- delrand_<s>       : random active source removed, rest renormalised

Interventions operate through the router's inference-only ``intervention``
hook (architecture/response_guided_router.py); no training math changes.
Validation loaders only — calibration/test are never touched (§67).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS
from config import prepare_args
from dataset import DataCollator
from main import (
    _build_model_components,
    _device_from_params,
    _loaders,
    _resolve_data_store,
    build_task_dict,
    build_parser,
    task_names_for_params,
    validate_params,
)
from metric import compute_regression_metrics
from reproducibility import seed_everything
from trainer import Trainer
import weighting as weighting_method

INTERVENTION_SEEDS = (1001, 1002, 1003)

PER_SAMPLE_FIELDS = (
    "seed",
    "sample_id",
    "row_index",
    "task",
    "label",
    "base_prediction",
    "normal_prediction",
)  # extended with one column per intervention

SUMMARY_FIELDS = (
    "seed",
    "task",
    "intervention",
    "n",
    "rmse",
    "mae",
    "r2",
    "delta_rmse_vs_normal",
    "delta_rmse_vs_base",
    "pred_abs_diff_vs_normal_mean",
    "pred_abs_diff_vs_normal_max",
    "route_regret_mean",
    "final_regret_mean",
)


def _read_args(run_dir: Path):
    return json.loads((run_dir / "args.json").read_text(encoding="utf-8"))


def read_best_epoch(run_dir: Path) -> int:
    best_rows = [
        row
        for row in csv.DictReader((run_dir / "diagnostics" / "epoch_summary.csv").open(newline="", encoding="utf-8"))
        if row["is_best"] == "True"
    ]
    if not best_rows:
        raise SystemExit(f"No best epoch recorded under {run_dir}")
    return max(int(row["epoch"]) for row in best_rows)


# ----------------------------------------------------------------------
# Intervention closures.  Each receives the router's freshly computed
# tensors and returns replacements.  Conditional weights always keep the
# contract sum(active) == 1 (plan §67).
# ----------------------------------------------------------------------
def make_route_only(learned_null):
    def hook(*, conditional_weights, joint_source_weights, null_weight, entropy, allowed):
        null = torch.zeros_like(null_weight)
        joint = torch.cat((null, conditional_weights * (1.0 - null)), dim=-1)
        ent = _joint_entropy(joint)
        return conditional_weights, joint, null, ent

    return hook


def make_null_one():
    def hook(*, conditional_weights, joint_source_weights, null_weight, entropy, allowed):
        null = torch.ones_like(null_weight)
        joint = torch.cat((null, torch.zeros_like(conditional_weights)), dim=-1)
        return conditional_weights, joint, null, torch.zeros_like(entropy)

    return hook


def make_uniform(learned_null, keep_null: bool):
    def hook(*, conditional_weights, joint_source_weights, null_weight, entropy, allowed):
        mask = allowed.to(conditional_weights.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform = mask / count
        null = learned_null.expand_as(null_weight) if keep_null else torch.zeros_like(null_weight)
        joint = torch.cat((null, uniform * (1.0 - null)), dim=-1)
        return uniform, joint, null, _joint_entropy(joint)

    return hook


def make_permutation(learned_conditional: torch.Tensor, learned_allowed: torch.Tensor, learned_null, intervention_seed: int):
    """Shuffle the learned weight VALUES across this seed's own active set."""

    active = learned_allowed.reshape(-1).nonzero().reshape(-1)
    generator = torch.Generator().manual_seed(int(intervention_seed))
    permutation = active[torch.randperm(active.numel(), generator=generator)]

    def hook(*, conditional_weights, joint_source_weights, null_weight, entropy, allowed):
        weights = learned_conditional.reshape(-1).clone()
        permuted = weights.clone()
        permuted[active] = weights[permutation]
        permuted = permuted.unsqueeze(0).expand_as(conditional_weights).contiguous()
        null = learned_null.expand_as(null_weight)
        joint = torch.cat((null, permuted * (1.0 - null)), dim=-1)
        return permuted, joint, null, _joint_entropy(joint)

    return hook


def make_deletion(learned_conditional: torch.Tensor, learned_allowed: torch.Tensor, learned_null, delete_index: int | None, intervention_seed: int | None):
    weights_vector = learned_conditional.reshape(-1)

    def hook(*, conditional_weights, joint_source_weights, null_weight, entropy, allowed):
        active = allowed.reshape(-1).nonzero().reshape(-1)
        if delete_index is not None:
            victim = int(delete_index)
        else:
            generator = torch.Generator().manual_seed(int(intervention_seed))
            victim = int(active[torch.randint(active.numel(), (1,), generator=generator)])
        deleted = weights_vector.clone()
        deleted[victim] = 0.0
        active_mask = allowed[0].to(deleted.dtype).clone()
        active_mask[victim] = 0.0
        total = (deleted * active_mask).sum().clamp_min(1e-12)
        renormalised = (deleted * active_mask) / total
        renormalised = renormalised.unsqueeze(0).expand_as(conditional_weights).contiguous()
        null = learned_null.expand_as(null_weight)
        joint = torch.cat((null, renormalised * (1.0 - null)), dim=-1)
        return renormalised, joint, null, _joint_entropy(joint)

    return hook


def _joint_entropy(joint: torch.Tensor) -> torch.Tensor:
    probabilities = joint.clamp_min(1e-12)
    return -(probabilities * probabilities.log()).sum(dim=-1)


# ----------------------------------------------------------------------
# Passes
# ----------------------------------------------------------------------
def run_pass(trainer, loader, task: str, epoch: int, hook=None, routing_enabled=True):
    """One validation pass; returns per-sample arrays (no grad)."""

    router = getattr(getattr(getattr(trainer, "model", None), "encoder", None), "task_conditioner", None)
    router = getattr(router, "router", None) if router is not None else None
    was_hook = router.intervention if router is not None else None
    if router is not None:
        router.intervention = hook
    try:
        targets, base_preds, route_preds, final_preds, sample_ids = [], [], [], [], []
        profile = {}
        with torch.no_grad():
            for batch in loader:
                if getattr(batch, "get", lambda *_: None)("is_empty", False):
                    continue
                batch = batch.to(trainer.device)
                output, diagnostics = trainer._forward_task(
                    batch, task, epoch, return_aux=True, routing_enabled_override=routing_enabled
                )
                targets.append(batch.y.reshape(-1).float().cpu())
                base_preds.append(
                    trainer.decode_task_output(task, diagnostics["base_raw"], apply_conformal=False)["median"].cpu().reshape(-1)
                )
                route_preds.append(
                    trainer.decode_task_output(task, diagnostics["route_raw"], apply_conformal=False)["median"].cpu().reshape(-1)
                )
                final_preds.append(
                    trainer.decode_task_output(task, output[task], apply_conformal=False)["median"].cpu().reshape(-1)
                )
                sample_ids.extend(str(value) for value in (getattr(batch, "sample_id", None) or []))
                if not profile:
                    profile = {
                        "conditional": diagnostics["source_weights"][0].float().cpu().clone(),
                        "null": diagnostics["null_weight"][0].float().cpu().reshape(-1)[:1].clone(),
                        "allowed": (diagnostics["source_weights"][0] > 0).cpu(),
                    }
        return {
            "target": torch.cat(targets) if targets else torch.empty(0),
            "base": torch.cat(base_preds) if base_preds else torch.empty(0),
            "route": torch.cat(route_preds) if route_preds else torch.empty(0),
            "final": torch.cat(final_preds) if final_preds else torch.empty(0),
            "sample_ids": sample_ids,
            "profile": profile,
        }
    finally:
        if router is not None:
            router.intervention = was_hook


def metrics_vs(pass_result, normal_result, base_rmse: float, normal_preds: torch.Tensor):
    final = pass_result["final"]
    target = pass_result["target"]
    regression = compute_regression_metrics(final.numpy(), target.numpy())
    rmse = regression["RMSE"]
    diff = (final - normal_preds).abs()
    route_regret = (pass_result["route"] - target).abs() - (pass_result["base"] - target).abs()
    final_regret = (final - target).abs() - (pass_result["base"] - target).abs()
    return {
        "n": int(target.numel()),
        "rmse": rmse,
        "mae": regression["MAE"],
        "r2": regression["R2"],
        "delta_rmse_vs_normal": rmse - float("nan"),
        "delta_rmse_vs_base": rmse - base_rmse,
        "pred_abs_diff_vs_normal_mean": float(diff.mean()) if diff.numel() else float("nan"),
        "pred_abs_diff_vs_normal_max": float(diff.max()) if diff.numel() else float("nan"),
        "route_regret_mean": float(route_regret.mean()),
        "final_regret_mean": float(final_regret.mean()),
    }


def audit_seed(run_dir: Path, gpu_id, seed: int):
    params = build_parser().parse_args([])
    for key, value in _read_args(run_dir).items():
        setattr(params, key, value)
    if gpu_id is not None:
        params.gpu_id = gpu_id
    validate_params(params)
    best_epoch = read_best_epoch(run_dir)

    seed_everything(params.seed)
    task_names = task_names_for_params(params)
    kwargs, optim_param = prepare_args(params)
    encoder_class, architecture_class, decoders = _build_model_components(
        params, task_names, _device_from_params(params)
    )
    collator = DataCollator(spatial_pos_max_clip=params.spatial_pos_clip, max_node_filter=None)
    trainer = Trainer(
        task_dict=build_task_dict(params, task_names),
        weighting=weighting_method.__dict__[params.weighting],
        architecture=architecture_class,
        encoder_class=encoder_class,
        decoders=decoders,
        optim_param=optim_param,
        args=params,
        save_path=str(run_dir),
        load_path=None,
        **kwargs,
    )
    trainer.load_checkpoint(run_dir / f"{params.ckpt_name}_best.pt")
    store = _resolve_data_store(params, task_names, required=True)
    trainer.sample_row_index = {
        str(sample_id): int(row_index)
        for sample_id, row_index in zip(store.sample_ids.tolist(), store.row_indices.tolist())
    }
    loaders = _loaders(params, task_names, collator)

    trainer.model.eval()
    summary_rows: list[dict] = []
    per_sample_rows: list[dict] = []
    deletion_rows: list[dict] = []
    prior_rows: list[dict] = []

    for task in [name for name in task_names if name in set(HUMAN_TARGET_TASKS)]:
        normal = run_pass(trainer, loaders["val"][task], task, best_epoch)
        base_only = run_pass(trainer, loaders["val"][task], task, best_epoch, routing_enabled=False)
        learned_conditional = normal["profile"]["conditional"]
        learned_allowed = normal["profile"]["allowed"]
        learned_null = normal["profile"]["null"]

        # Learned prior vector dump for the D4-1 stability analysis (§34).
        for index, source_task in enumerate(task_names):
            prior_rows.append(
                {
                    "seed": seed,
                    "target_task": task,
                    "source_task": source_task,
                    "is_active": int(bool(learned_allowed[index])) if index < learned_allowed.numel() else 0,
                    "conditional_weight": float(learned_conditional[index]),
                    "null_weight": float(learned_null[0]),
                }
            )

        base_rmse = float(compute_regression_metrics(base_only["final"].numpy(), normal["target"].numpy())["RMSE"])
        normal_rmse = float(compute_regression_metrics(normal["final"].numpy(), normal["target"].numpy())["RMSE"])
        intervention_list = [
            ("normal", None, True),
            ("base_only", None, False),
            ("route_only", make_route_only(learned_null), True),
            ("null_one", make_null_one(), True),
            ("uniform_null_kept", make_uniform(learned_null, keep_null=True), True),
            ("uniform_null_zero", make_uniform(learned_null, keep_null=False), True),
        ]
        for intervention_seed in INTERVENTION_SEEDS:
            intervention_list.append(
                (f"perm_{intervention_seed}", make_permutation(learned_conditional, learned_allowed, learned_null, intervention_seed), True)
            )
        intervention_list.append(
            ("del_top1", make_deletion(learned_conditional, learned_allowed, learned_null, int(learned_conditional.argmax()), None), True)
        )
        for intervention_seed in INTERVENTION_SEEDS:
            intervention_list.append(
                (f"delrand_{intervention_seed}", make_deletion(learned_conditional, learned_allowed, learned_null, None, intervention_seed), True)
            )

        per_sample_interventions: dict[str, list[float]] = {}
        for name, hook, routing_enabled in intervention_list:
            if name == "normal":
                result = normal
            elif name == "base_only":
                result = base_only
            else:
                result = run_pass(trainer, loaders["val"][task], task, best_epoch, hook=hook, routing_enabled=routing_enabled)
            row = {"seed": seed, "task": task, "intervention": name}
            row.update(metrics_vs(result, normal, base_rmse, normal["final"]))
            row["delta_rmse_vs_normal"] = row["rmse"] - normal_rmse
            if name == "null_one":
                # §15: NULL=1 must reproduce Base-only exactly (prediction space).
                max_diff = float((result["final"] - base_only["final"]).abs().max())
                assert max_diff < 1e-6, f"null_one != base_only for {task}: {max_diff}"
            summary_rows.append(row)
            per_sample_interventions[name] = result["final"].tolist()
            if name.startswith("delrand_"):
                deletion_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "intervention": name,
                        "intervention_seed": int(name.split("_")[1]),
                        "random_deletion_impact": row["rmse"] - normal_rmse,
                    }
                )
            if name == "del_top1":
                deletion_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "intervention": name,
                        "intervention_seed": "",
                        "random_deletion_impact": row["rmse"] - normal_rmse,
                        "top1_deletion_impact": row["rmse"] - normal_rmse,
                    }
                )

        for index, sample_id in enumerate(normal["sample_ids"]):
            row = {
                "seed": seed,
                "sample_id": sample_id,
                "row_index": trainer.sample_row_index.get(sample_id, ""),
                "task": task,
                "label": float(normal["target"][index]),
                "base_prediction": float(normal["base"][index]),
                "normal_prediction": float(normal["final"][index]),
            }
            for name, _, _ in intervention_list:
                row[f"pred_{name}"] = per_sample_interventions[name][index]
            per_sample_rows.append(row)

    return summary_rows, per_sample_rows, deletion_rows, prior_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=[
            "artifacts/runs/d3_mechanisms/task_only_router/d3_task_only_router_e40/seed_42",
            "artifacts/runs/d3_mechanisms/task_only_router/d3_task_only_router_e40/seed_43",
            "artifacts/runs/d3_mechanisms/task_only_router/d3_task_only_router_e40/seed_44",
        ],
    )
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--gpu_id", default="1")
    args = parser.parse_args()

    all_summary, all_per_sample, all_deletion, all_prior = [], [], [], []
    for run_dir in args.run_dirs:
        run_path = Path(run_dir)
        seed = int(run_path.name.replace("seed_", ""))
        summary, per_sample, deletion, prior = audit_seed(run_path, args.gpu_id, seed)
        all_summary.extend(summary)
        all_per_sample.extend(per_sample)
        all_deletion.extend(deletion)
        all_prior.extend(prior)
        print(f"audited {run_path}: {len(summary)} intervention rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"wrote {path}")

    write(output_dir / "D4_0_INTERVENTION_SUMMARY.csv", SUMMARY_FIELDS, all_summary)
    sample_fields = list(PER_SAMPLE_FIELDS) + [
        f"pred_{name}"
        for name in (
            ["base_only", "route_only", "null_one", "uniform_null_kept", "uniform_null_zero"]
            + [f"perm_{seed}" for seed in INTERVENTION_SEEDS]
            + ["del_top1"]
            + [f"delrand_{seed}" for seed in INTERVENTION_SEEDS]
        )
    ]
    write(output_dir / "D4_0_INTERVENTION_PER_SAMPLE.csv", sample_fields, all_per_sample)
    write(
        output_dir / "D4_0_SOURCE_DELETION.csv",
        ("seed", "task", "intervention", "intervention_seed", "top1_deletion_impact", "random_deletion_impact"),
        all_deletion,
    )
    write(
        output_dir / "d4_0_prior_vectors.csv",
        ("seed", "target_task", "source_task", "is_active", "conditional_weight", "null_weight"),
        all_prior,
    )


if __name__ == "__main__":
    main()
