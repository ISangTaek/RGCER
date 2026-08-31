"""Aggregate D4 Stage-1 results (plan §29-§35, §39-§40, §52-§53, §62-§63).

Produces:
- D4_1_5SEED_SUMMARY.csv       per (model, seed) best-epoch summary + shaping/routing gains
- D4_1_ENDPOINT_SUMMARY.csv    per (model, seed, human endpoint) RMSE/MAE/R2
- D4_1_PAIRED_COMPARISON.csv   paired HPS - TaskOnly deltas + mean/std/median/W-T-L
- D4_1_PRIOR_STABILITY.csv     cross-seed prior agreement per human target
- D4_1_PRIOR_TOP_SOURCES.csv   top-10 sources by cross-seed mean weight (+ metadata)

Task-only seeds 42/43/44 resume runs cover epochs 40-99 in the D4 root; their
epochs 0-39 diagnostics stay in the D3 run directories, so both are merged
before selecting the global best epoch.
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

from architecture.toxacute_tasks import parse_toxacute_task_name

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]

HPS_REFERENCE_HUMAN3_RMSE = {
    42: 1.163448,
    43: 1.112644,
    44: 0.984083,
}

FIVE_SEED_FIELDS = [
    "model",
    "seed",
    "source",
    "best_epoch",
    "human3_rmse",
    "man_rmse",
    "women_rmse",
    "human_rmse",
    "all59_macro_rmse",
    "hps_minus_taskonly",
    "base_human3_rmse",
    "route_human3_rmse",
    "final_human3_rmse",
    "mean_null",
    "mean_transfer_mass",
    "training_shaping_gain",
    "inference_routing_gain",
    "first_improvement_epoch_vs_hps",
    "status",
]

ENDPOINT_FIELDS = ["model", "seed", "task", "rmse", "mae", "r2", "n"]

STABILITY_FIELDS = [
    "target_task",
    "seed_a",
    "seed_b",
    "top1_agree",
    "top8_jaccard",
    "weight_spearman",
    "weight_cosine",
    "null_weight_abs_diff",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _merged_rows(diagnostics_dirs: list[Path], filename: str) -> list[dict]:
    rows: list[dict] = []
    for directory in diagnostics_dirs:
        rows.extend(_read_csv(directory / filename))
    return rows


def _best_row(epoch_summary: list[dict]) -> dict | None:
    candidates = [
        row
        for row in epoch_summary
        if row.get("val_human3_macro_rmse") not in (None, "", "nan")
        and math.isfinite(float(row["val_human3_macro_rmse"]))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["val_human3_macro_rmse"]))


def _human3_macro_at(path_metrics: list[dict], epoch: str, path: str) -> float:
    values = [
        float(row["rmse"])
        for row in path_metrics
        if row["epoch"] == epoch and row["path"] == path and row["task"] in HUMAN_TASKS
    ]
    return _finite_mean(values) if values else float("nan")


def _endpoint_metrics(path_metrics: list[dict], epoch: str) -> dict[str, dict]:
    metrics = {}
    for task in HUMAN_TASKS:
        row = next(
            (
                row
                for row in path_metrics
                if row["epoch"] == epoch and row["path"] == "final" and row["task"] == task
            ),
            None,
        )
        metrics[task] = (
            {
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
                "r2": float(row["r2"]),
                "n": int(row["n"]),
            }
            if row
            else {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
        )
    return metrics


def summarize_model_run(
    model: str,
    seed: int,
    source: str,
    diagnostics_dirs: list[Path],
    hps_reference: float,
) -> tuple[dict, list[dict]]:
    epoch_summary = _merged_rows(diagnostics_dirs, "epoch_summary.csv")
    path_metrics = _merged_rows(diagnostics_dirs, "human3_path_metrics.csv")
    routing_epoch = _merged_rows(diagnostics_dirs, "routing_epoch.csv")

    best = _best_row(epoch_summary)
    if best is None:
        return (
            {
                "model": model,
                "seed": seed,
                "source": source,
                "status": "INCOMPLETE_NO_BEST_EPOCH",
            },
            [],
        )
    best_epoch = best["epoch"]
    best_val = float(best["val_human3_macro_rmse"])
    endpoints = _endpoint_metrics(path_metrics, best_epoch)

    routing_rows = [row for row in routing_epoch if row["epoch"] == best_epoch and row["task"] in HUMAN_TASKS]
    mean_null = _finite_mean([row.get("mean_null_weight", float("nan")) for row in routing_rows]) if routing_rows else float("nan")
    mean_transfer = _finite_mean([row.get("mean_transfer_mass", float("nan")) for row in routing_rows]) if routing_rows else float("nan")

    first_improvement = ""
    if model == "task_only":
        for row in sorted(epoch_summary, key=lambda row: int(row["epoch"])):
            value = row.get("val_human3_macro_rmse")
            if value and math.isfinite(float(value)) and float(value) < hps_reference:
                first_improvement = row["epoch"]
                break

    row = {
        "model": model,
        "seed": seed,
        "source": source,
        "best_epoch": int(best_epoch),
        "human3_rmse": best_val,
        "man_rmse": endpoints[HUMAN_TASKS[0]]["rmse"],
        "women_rmse": endpoints[HUMAN_TASKS[1]]["rmse"],
        "human_rmse": endpoints[HUMAN_TASKS[2]]["rmse"],
        "all59_macro_rmse": float(best["val_all59_macro_rmse"]),
        "hps_minus_taskonly": hps_reference - best_val if model == "task_only" else float("nan"),
        "base_human3_rmse": _human3_macro_at(path_metrics, best_epoch, "base"),
        "route_human3_rmse": _human3_macro_at(path_metrics, best_epoch, "route"),
        "final_human3_rmse": _human3_macro_at(path_metrics, best_epoch, "final"),
        "mean_null": mean_null,
        "mean_transfer_mass": mean_transfer,
        "training_shaping_gain": hps_reference - _human3_macro_at(path_metrics, best_epoch, "base")
        if model == "task_only"
        else float("nan"),
        "inference_routing_gain": _human3_macro_at(path_metrics, best_epoch, "base")
        - _human3_macro_at(path_metrics, best_epoch, "final")
        if model == "task_only"
        else float("nan"),
        "first_improvement_epoch_vs_hps": first_improvement,
        "status": "OK" if math.isfinite(best_val) else "FAILED_NON_FINITE",
    }

    endpoint_rows = [
        {"model": model, "seed": seed, "task": task, **endpoints[task]} for task in HUMAN_TASKS
    ]
    return row, endpoint_rows


def _rankdata(values):
    values = list(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = average
        position = end + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return float("nan")
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_centered = [value - sum(left_rank) / len(left_rank) for value in left_rank]
    right_centered = [value - sum(right_rank) / len(right_rank) for value in right_rank]
    denominator = math.sqrt(sum(value**2 for value in left_centered) * sum(value**2 for value in right_centered))
    if denominator == 0:
        return float("nan")
    return sum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value**2 for value in left) * sum(value**2 for value in right))
    if denominator == 0:
        return float("nan")
    return sum(a * b for a, b in zip(left, right)) / denominator


def _prior_stability(prior_rows: list[dict], seeds: list[int]) -> list[dict]:
    by_target_seed: dict[str, dict[int, dict]] = {}
    for row in prior_rows:
        target = row["target_task"]
        seed = int(row["seed"])
        by_target_seed.setdefault(target, {}).setdefault(seed, {})[row["source_task"]] = row

    stability_rows = []
    for target, per_seed in by_target_seed.items():
        for index_a, seed_a in enumerate(seeds):
            for seed_b in seeds[index_a + 1 :]:
                vector_a = per_seed.get(seed_a, {})
                vector_b = per_seed.get(seed_b, {})
                if not vector_a or not vector_b:
                    continue
                sources = sorted(set(vector_a) | set(vector_b))
                weights_a = [float(vector_a.get(source, {}).get("conditional_weight", 0.0)) for source in sources]
                weights_b = [float(vector_b.get(source, {}).get("conditional_weight", 0.0)) for source in sources]
                top1_a = max(sources, key=lambda source: float(vector_a.get(source, {}).get("conditional_weight", 0.0)))
                top1_b = max(sources, key=lambda source: float(vector_b.get(source, {}).get("conditional_weight", 0.0)))
                top8_a = set(
                    sorted(sources, key=lambda source: float(vector_a.get(source, {}).get("conditional_weight", 0.0)), reverse=True)[:8]
                )
                top8_b = set(
                    sorted(sources, key=lambda source: float(vector_b.get(source, {}).get("conditional_weight", 0.0)), reverse=True)[:8]
                )
                null_a = {float(value.get("null_weight", float("nan"))) for value in vector_a.values()}
                null_b = {float(value.get("null_weight", float("nan"))) for value in vector_b.values()}
                stability_rows.append(
                    {
                        "target_task": target,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "top1_agree": int(top1_a == top1_b),
                        "top8_jaccard": len(top8_a & top8_b) / max(len(top8_a | top8_b), 1),
                        "weight_spearman": _spearman(weights_a, weights_b),
                        "weight_cosine": _cosine(weights_a, weights_b),
                        "null_weight_abs_diff": abs(_finite_mean(list(null_a)) - _finite_mean(list(null_b))),
                    }
                )
    return stability_rows


def _prior_top_sources(prior_rows: list[dict], seeds: list[int]) -> list[dict]:
    by_target: dict[str, dict[str, list[float]]] = {}
    for row in prior_rows:
        target = row["target_task"]
        by_target.setdefault(target, {}).setdefault(row["source_task"], []).append(
            float(row["conditional_weight"])
        )

    top_rows = []
    for target, per_source in by_target.items():
        means = {
            source: _finite_mean(values)
            for source, values in per_source.items()
            if len(values) == len(seeds)
        }
        ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)[:10]
        for rank, (source, mean_weight) in enumerate(ranked, start=1):
            values = per_source[source]
            std = (
                math.sqrt(sum((value - mean_weight) ** 2 for value in values) / len(values))
                if len(values) > 1
                else float("nan")
            )
            ranks = [
                sorted(per_source[source], reverse=True).index(value) + 1 if value in sorted(per_source[source], reverse=True) else float("nan")
                for value in values
            ]
            rank_std = (
                math.sqrt(sum((rank - _finite_mean(ranks)) ** 2 for rank in ranks) / len(ranks))
                if len(ranks) > 1
                else float("nan")
            )
            try:
                metadata = parse_toxacute_task_name(source)
                species, route, toxicity = metadata.organism, metadata.route, metadata.measurement
            except ValueError:
                species = route = toxicity = ""
            top_rows.append(
                {
                    "target_task": target,
                    "rank": rank,
                    "source_task": source,
                    "mean_weight": mean_weight,
                    "std_across_seeds": std,
                    "rank_std": rank_std,
                    "species": species,
                    "route": route,
                    "toxicity_type": toxicity,
                }
            )
    return top_rows


def _summarize_hps_reference(seed: int, run_dir: Path) -> tuple[dict, list[dict]]:
    """Recover per-endpoint HPS reference metrics from a formal run's metrics.json.

    The frozen HPS 42/43/44 runs predate the per-epoch diagnostics CSVs but
    their metrics.json history carries the identical validation numbers.
    """

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return (
            {"model": "hps", "seed": seed, "source": "frozen_reference", "status": "MISSING_METRICS_JSON"},
            [],
        )
    history = json.loads(metrics_path.read_text(encoding="utf-8")).get("history", [])
    validation_epochs = [entry for entry in history if "validation" in entry]
    if not validation_epochs:
        return (
            {"model": "hps", "seed": seed, "source": "frozen_reference", "status": "MISSING_HISTORY"},
            [],
        )
    best = min(
        validation_epochs,
        key=lambda entry: entry["validation"].get("human3_macro_rmse", float("inf")),
    )
    tasks = best["validation"]["tasks"]
    best_val = float(best["validation"]["human3_macro_rmse"])
    row = {
        "model": "hps",
        "seed": seed,
        "source": "formal_hps_100e",
        "best_epoch": int(best["epoch"]),
        "human3_rmse": best_val,
        "man_rmse": float(tasks.get(HUMAN_TASKS[0], {}).get("RMSE", float("nan"))),
        "women_rmse": float(tasks.get(HUMAN_TASKS[1], {}).get("RMSE", float("nan"))),
        "human_rmse": float(tasks.get(HUMAN_TASKS[2], {}).get("RMSE", float("nan"))),
        "all59_macro_rmse": float(best["validation"].get("all_task_macro_rmse", float("nan"))),
        "status": "OK",
    }
    endpoint_rows = [
        {
            "model": "hps",
            "seed": seed,
            "task": task,
            "rmse": float(tasks.get(task, {}).get("RMSE", float("nan"))),
            "mae": float("nan"),
            "r2": float(tasks.get(task, {}).get("R2", float("nan"))),
            "n": 0,
        }
        for task in HUMAN_TASKS
    ]
    return row, endpoint_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d4_task_root", default="artifacts/runs/d4_task_only_100e/d4_task_only_e100")
    parser.add_argument("--d3_task_root", default="artifacts/runs/d3_mechanisms/task_only_router/d3_task_only_router_e40")
    parser.add_argument("--hps_root", default="artifacts/runs/d4_hps_100e/d4_hps_e100")
    parser.add_argument("--formal_hps_root", default="artifacts/runs/formal_hps/formal")
    parser.add_argument("--prior_csv", default="d4_0_prior_vectors.csv")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_rows: list[dict] = []
    endpoint_rows: list[dict] = []

    # HPS first: 45/46 fresh runs, 42/43/44 formal references.  Task-only
    # paired deltas below need every seed's HPS reference value.
    hps_reference_by_seed: dict[int, float] = {}
    for seed in (45, 46):
        diagnostics_dirs = [Path(args.hps_root) / f"seed_{seed}" / "diagnostics"]
        if not diagnostics_dirs[0].exists():
            summary_rows.append(
                {"model": "hps", "seed": seed, "source": "missing", "status": "MISSING_RUN_DIR"}
            )
            continue
        row, endpoint_section = summarize_model_run("hps", seed, "fresh_100", diagnostics_dirs, float("nan"))
        summary_rows.append(row)
        endpoint_rows.extend(endpoint_section)
        if row.get("human3_rmse") is not None:
            hps_reference_by_seed[seed] = float(row["human3_rmse"])
    for seed in (42, 43, 44):
        reference_row, reference_endpoints = _summarize_hps_reference(
            seed, Path(args.formal_hps_root) / f"seed_{seed}"
        )
        if "human3_rmse" not in reference_row:
            reference_row["human3_rmse"] = HPS_REFERENCE_HUMAN3_RMSE[seed]
        summary_rows.append(reference_row)
        endpoint_rows.extend(reference_endpoints)
        hps_reference_by_seed[seed] = float(
            reference_row.get("human3_rmse", HPS_REFERENCE_HUMAN3_RMSE[seed])
        )

    # Task-only runs: 42/43/44 merge the D3-era diagnostics (epochs 0-39).
    for seed in args.seeds:
        diagnostics_dirs = [Path(args.d4_task_root) / f"seed_{seed}" / "diagnostics"]
        if seed in (42, 43, 44):
            diagnostics_dirs.insert(0, Path(args.d3_task_root) / f"seed_{seed}" / "diagnostics")
        if not any(directory.exists() for directory in diagnostics_dirs):
            summary_rows.append(
                {"model": "task_only", "seed": seed, "source": "missing", "status": "MISSING_RUN_DIR"}
            )
            continue
        source = "resume_40_to_100" if seed in (42, 43, 44) else "fresh_100"
        row, endpoint_section = summarize_model_run(
            "task_only", seed, source, diagnostics_dirs, hps_reference_by_seed[seed]
        )
        summary_rows.append(row)
        endpoint_rows.extend(endpoint_section)

    # Paired comparison over the shared seeds (§62-§63).
    task_only_by_seed = {
        int(row["seed"]): float(row["human3_rmse"])
        for row in summary_rows
        if row["model"] == "task_only" and row.get("human3_rmse") not in (None, "")
    }
    hps_by_seed = {
        int(row["seed"]): float(row["human3_rmse"])
        for row in summary_rows
        if row["model"] == "hps" and row.get("human3_rmse") not in (None, "")
    }
    deltas = []
    paired_rows = []
    for seed in args.seeds:
        if seed not in task_only_by_seed or seed not in hps_by_seed:
            continue
        delta = hps_by_seed[seed] - task_only_by_seed[seed]
        deltas.append(delta)
        paired_rows.append(
            {
                "seed": seed,
                "hps_human3_rmse": hps_by_seed[seed],
                "taskonly_human3_rmse": task_only_by_seed[seed],
                "paired_delta_rmse": delta,
            }
        )
    effect = {
        "paired_delta_rmse_mean": _finite_mean(deltas),
        "paired_delta_rmse_std": (
            math.sqrt(sum((delta - _finite_mean(deltas)) ** 2 for delta in deltas) / len(deltas))
            if len(deltas) > 1
            else float("nan")
        ),
        "paired_delta_rmse_median": sorted(deltas)[len(deltas) // 2] if deltas else float("nan"),
        "win": sum(1 for delta in deltas if delta > 0),
        "tie": sum(1 for delta in deltas if delta == 0),
        "loss": sum(1 for delta in deltas if delta < 0),
        "seeds_compared": len(deltas),
    }

    prior_rows = _read_csv(Path(args.prior_csv))
    stability_rows = _prior_stability(prior_rows, args.seeds)
    top_source_rows = _prior_top_sources(prior_rows, args.seeds)

    def write(name: str, fields, rows):
        path = output_dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"wrote {path}")

    write("D4_1_5SEED_SUMMARY.csv", FIVE_SEED_FIELDS, summary_rows)
    write("D4_1_ENDPOINT_SUMMARY.csv", ENDPOINT_FIELDS, endpoint_rows)
    paired_fields = ["seed", "hps_human3_rmse", "taskonly_human3_rmse", "paired_delta_rmse"]
    write("D4_1_PAIRED_COMPARISON.csv", paired_fields, paired_rows)
    write("D4_1_PRIOR_STABILITY.csv", STABILITY_FIELDS, stability_rows)
    write(
        "D4_1_PRIOR_TOP_SOURCES.csv",
        ["target_task", "rank", "source_task", "mean_weight", "std_across_seeds", "rank_std", "species", "route", "toxicity_type"],
        top_source_rows,
    )
    import json

    print(json.dumps(effect, indent=2))


if __name__ == "__main__":
    main()
