"""Aggregate D3 mechanism-localization runs (plan §30-§32, §42).

Reads the nine D3 run directories under
``artifacts/runs/d3_mechanisms/{target_only,response_stacking,task_only_router}/d3_<mech>_e40/seed_*``
and writes:

- DIAG_D3_MECHANISM_SUMMARY.csv  (plan §31 columns, one row per mechanism x seed)
- DIAG_D3_PAIRED_COMPARISON.csv  (plan §32 paired Human3 validation table)
- per-run experiment_record.md + checkpoint_manifest.json (plan §42-§43)

The NoNull column of the paired table comes from DIAG_D1_SUMMARY.csv when
available (fallback NaN).  "conditioned" refers to the routed/conditioned
prediction path (diagnostics ``route``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]

HPS_REFERENCE_HUMAN3_RMSE = {
    42: 1.163448,
    43: 1.112644,
    44: 0.984083,
}

MECHANISMS = {
    "target_only": "artifacts/runs/d3_mechanisms/target_only/d3_target_only_e40",
    "response_stacking": "artifacts/runs/d3_mechanisms/response_stacking/d3_response_stacking_e40",
    "task_only_router": "artifacts/runs/d3_mechanisms/task_only_router/d3_task_only_router_e40",
}

SUMMARY_FIELDS = [
    "mechanism",
    "seed",
    "best_epoch",
    "hps_reference_human3_rmse",
    "best_human3_rmse",
    "paired_delta_vs_hps",
    "man_rmse",
    "women_rmse",
    "human_rmse",
    "all59_macro_rmse",
    "base_human3_rmse",
    "conditioned_human3_rmse",
    "final_human3_rmse",
    "conditioned_minus_base_abs_mean",
    "conditioned_regret_mean",
    "conditioned_negative_transfer_rate",
    "routing_entropy_mean",
    "routing_variance_mean",
    "unique_top1_sources",
    "context_norm_ratio",
    "film_delta_ratio",
    "adapter_delta_ratio",
    "final_rep_delta_ratio",
    "router_grad_mean",
    "film_grad_mean",
    "adapter_grad_mean",
    "status",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _best_row(epoch_summary: list[dict]) -> dict | None:
    best_rows = [row for row in epoch_summary if row.get("is_best", "").strip().lower() == "true"]
    if not best_rows:
        return None
    return max(best_rows, key=lambda row: int(row["epoch"]))


def _human3_macro_at(path_metrics: list[dict], epoch: str, path: str) -> float:
    values = [
        float(row["rmse"])
        for row in path_metrics
        if row["epoch"] == epoch and row["path"] == path and row["task"] in HUMAN_TASKS
    ]
    return _finite_mean(values) if values else float("nan")


def _endpoint_rmse(path_metrics: list[dict], epoch: str) -> dict[str, float]:
    return {
        task: next(
            (
                float(row["rmse"])
                for row in path_metrics
                if row["epoch"] == epoch and row["path"] == "final" and row["task"] == task
            ),
            float("nan"),
        )
        for task in HUMAN_TASKS
    }


def _grad_mean_at(gradient_norms: list[dict], epoch: str, groups: tuple[str, ...]) -> float:
    rows = [row for row in gradient_norms if row["epoch"] == epoch and row["group"] in groups]
    return _finite_mean([row["mean"] for row in rows]) if rows else float("nan")


def _representation_at(representation_epoch: list[dict], epoch: str, field: str) -> float:
    rows = [row for row in representation_epoch if row["epoch"] == epoch and row["task"] in HUMAN_TASKS]
    return _finite_mean([row.get(field, float("nan")) for row in rows]) if rows else float("nan")


def _routing_at(routing_epoch: list[dict], epoch: str, field: str) -> float:
    rows = [row for row in routing_epoch if row["epoch"] == epoch and row["task"] in HUMAN_TASKS]
    return _finite_mean([row.get(field, float("nan")) for row in rows]) if rows else float("nan")


def _unique_top1_sources(run_dir: Path) -> int:
    path = run_dir / "diagnostics" / "routing_source_frequency.json"
    if not path.exists():
        return 0
    tasks = json.loads(path.read_text(encoding="utf-8")).get("tasks", {})
    union: set[str] = set()
    for task in HUMAN_TASKS:
        union.update(tasks.get(task, {}).get("top1_source_counts", {}).keys())
    return len(union)


def _checkpoint_manifest(run_dir: Path, ckpt_prefix: str, best_epoch) -> dict:
    manifest: dict = {"best_epoch": best_epoch}
    for label, filename in (("best", f"{ckpt_prefix}_best.pt"), ("last", f"{ckpt_prefix}_last.pt")):
        path = run_dir / filename
        entry = {"path": str(path)}
        if path.exists():
            entry["sha256"] = _sha256_file(path)
            entry["bytes"] = path.stat().st_size
        else:
            entry["sha256"] = None
            entry["bytes"] = None
        manifest[f"{label}_checkpoint"] = entry
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest["checkpoint_version"] = metadata.get("checkpoint_version")
        manifest["git_commit"] = metadata.get("git_commit")
        manifest["evaluation_scope"] = metadata.get("evaluation_scope")
    return manifest


def _experiment_record(run_dir: Path, mechanism: str, seed: int, summary_row: dict) -> None:
    record = [
        f"# Experiment record: D3 {mechanism} seed {seed}",
        "",
        f"- phase: D3_MECHANISM_LOCALIZATION",
        f"- mechanism: {mechanism}",
        f"- seed: {seed}",
        f"- best_epoch: {summary_row.get('best_epoch')}",
        f"- best human3 validation RMSE: {summary_row.get('best_human3_rmse')}",
        f"- paired delta vs HPS: {summary_row.get('paired_delta_vs_hps')}",
        f"- status: {summary_row.get('status')}",
        "",
        "Validation-only diagnostic run (plan D3 §16/§19/§22): no calibration/test access,",
        "no CQR, no test evaluation.",
    ]
    (run_dir / "experiment_record.md").write_text("\n".join(record) + "\n", encoding="utf-8")


def summarize_run(mechanism: str, seed: int, run_dir: Path, status: str) -> dict:
    diagnostics = run_dir / "diagnostics"
    epoch_summary = _read_csv(diagnostics / "epoch_summary.csv")
    path_metrics = _read_csv(diagnostics / "human3_path_metrics.csv")
    routing_epoch = _read_csv(diagnostics / "routing_epoch.csv")
    gradient_norms = _read_csv(diagnostics / "gradient_norms.csv")
    representation_epoch = _read_csv(diagnostics / "representation_epoch.csv")

    best = _best_row(epoch_summary)
    if best is None:
        return {
            "mechanism": mechanism,
            "seed": seed,
            "status": status or "INCOMPLETE_NO_BEST_EPOCH",
        }
    best_epoch = best["epoch"]
    endpoint = _endpoint_rmse(path_metrics, best_epoch)

    best_val = float(best["val_human3_macro_rmse"])
    row_status = status or ("OK" if math.isfinite(best_val) else "FAILED_NON_FINITE")

    conditioned_minus_base = float("nan")
    conditioned_regret = float("nan")
    conditioned_ntr = float("nan")
    predictions_path = diagnostics / "best_validation_human3_predictions.csv"
    if predictions_path.exists():
        pred_rows = [
            row
            for row in _read_csv(predictions_path)
            if row["task"] in HUMAN_TASKS
        ]
        deltas = [
            abs(float(row["route_prediction"]) - float(row["base_prediction"]))
            for row in pred_rows
        ]
        regrets = [float(row["route_regret"]) for row in pred_rows]
        conditioned_minus_base = _finite_mean(deltas)
        conditioned_regret = _finite_mean(regrets)
        conditioned_ntr = _finite_mean([1.0 if regret > 0 else 0.0 for regret in regrets])

    hps_reference = HPS_REFERENCE_HUMAN3_RMSE[seed]
    return {
        "mechanism": mechanism,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "hps_reference_human3_rmse": hps_reference,
        "best_human3_rmse": best_val,
        "paired_delta_vs_hps": hps_reference - best_val,
        "man_rmse": endpoint.get(HUMAN_TASKS[0], float("nan")),
        "women_rmse": endpoint.get(HUMAN_TASKS[1], float("nan")),
        "human_rmse": endpoint.get(HUMAN_TASKS[2], float("nan")),
        "all59_macro_rmse": float(best["val_all59_macro_rmse"]),
        "base_human3_rmse": _human3_macro_at(path_metrics, best_epoch, "base"),
        "conditioned_human3_rmse": _human3_macro_at(path_metrics, best_epoch, "route"),
        "final_human3_rmse": _human3_macro_at(path_metrics, best_epoch, "final"),
        "conditioned_minus_base_abs_mean": conditioned_minus_base,
        "conditioned_regret_mean": conditioned_regret,
        "conditioned_negative_transfer_rate": conditioned_ntr,
        "routing_entropy_mean": _routing_at(routing_epoch, best_epoch, "mean_routing_entropy"),
        "routing_variance_mean": _routing_at(routing_epoch, best_epoch, "routing_variance"),
        "unique_top1_sources": _unique_top1_sources(run_dir),
        "context_norm_ratio": _representation_at(representation_epoch, best_epoch, "conditioning_context_ratio_mean"),
        "film_delta_ratio": _representation_at(representation_epoch, best_epoch, "film_delta_ratio_mean"),
        "adapter_delta_ratio": _representation_at(representation_epoch, best_epoch, "adapter_delta_ratio_mean"),
        "final_rep_delta_ratio": _representation_at(representation_epoch, best_epoch, "route_rep_delta_ratio_mean"),
        "router_grad_mean": _grad_mean_at(
            gradient_norms, best_epoch, ("router_query_projection", "router_key_projection", "router_null")
        ),
        "film_grad_mean": _grad_mean_at(gradient_norms, best_epoch, ("film_generator",)),
        "adapter_grad_mean": _grad_mean_at(gradient_norms, best_epoch, ("adapter",)),
        "status": row_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", default="artifacts/runs/d3_mechanisms")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--d1_summary", default="DIAG_D1_SUMMARY.csv")
    args = parser.parse_args()

    no_null_by_seed: dict[int, float] = {}
    d1_path = Path(args.d1_summary)
    if d1_path.exists():
        for row in _read_csv(d1_path):
            try:
                no_null_by_seed[int(row["seed"])] = float(row["no_null_best_human3_rmse"])
            except (KeyError, TypeError, ValueError):
                continue

    rows = []
    for mechanism, rel_root in MECHANISMS.items():
        mech_root = Path(args.runs_root) / mechanism
        for seed in args.seeds:
            run_dir = mech_root / f"seed_{seed}"
            if not run_dir.exists():
                rows.append({"mechanism": mechanism, "seed": seed, "status": "MISSING_RUN_DIR"})
                continue
            row = summarize_run(mechanism, seed, run_dir, status=None)
            rows.append(row)
            ckpt_prefix = f"rgcer_d3_{mechanism}_e40_seed{seed}"
            manifest = _checkpoint_manifest(run_dir, ckpt_prefix, row.get("best_epoch"))
            manifest["seed"] = seed
            manifest["mechanism"] = mechanism
            (run_dir / "checkpoint_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            _experiment_record(run_dir, mechanism, seed, row)

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "DIAG_D3_MECHANISM_SUMMARY.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {summary_path}")

    # Plan §32: paired comparison across all mechanisms on Human3 validation.
    by_mechanism: dict[str, dict[int, float]] = {}
    for row in rows:
        value = row.get("best_human3_rmse")
        if isinstance(value, (int, float)) and math.isfinite(value):
            by_mechanism.setdefault(row["mechanism"], {})[int(row["seed"])] = float(value)

    paired_rows = []
    for seed in args.seeds:
        hps = HPS_REFERENCE_HUMAN3_RMSE[seed]
        target_only = by_mechanism.get("target_only", {}).get(seed, float("nan"))
        stacking = by_mechanism.get("response_stacking", {}).get(seed, float("nan"))
        task_only = by_mechanism.get("task_only_router", {}).get(seed, float("nan"))
        paired_rows.append(
            {
                "seed": seed,
                "HPS": hps,
                "NoNull": no_null_by_seed.get(seed, float("nan")),
                "TargetOnly": target_only,
                "ResponseStacking": stacking,
                "TaskOnlyRouter": task_only,
                "HPS_minus_TargetOnly": hps - target_only,
                "HPS_minus_ResponseStacking": hps - stacking,
                "HPS_minus_TaskOnlyRouter": hps - task_only,
                "TargetOnly_minus_ResponseStacking": target_only - stacking,
                "TargetOnly_minus_TaskOnlyRouter": target_only - task_only,
            }
        )
    paired_path = output_dir / "DIAG_D3_PAIRED_COMPARISON.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        for row in paired_rows:
            writer.writerow(row)
    print(f"wrote {paired_path}")

    # Mechanism means for the §33-§37 case evaluation.
    means = {
        mechanism: _finite_mean(list(values.values()))
        for mechanism, values in by_mechanism.items()
    }
    print(json.dumps({"mechanism_mean_human3_rmse": means}, indent=2))


if __name__ == "__main__":
    main()
