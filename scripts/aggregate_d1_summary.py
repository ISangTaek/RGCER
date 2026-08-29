"""Aggregate D1 No-NULL diagnostic runs into DIAG_D1_SUMMARY.csv (plan §24).

Reads the three ``artifacts/runs/diag_no_null/d1_no_null_e40/seed_*`` run
directories, recomputes the human3 path macros from the per-task
``human3_path_metrics.csv`` (the in-run epoch_summary columns may predate the
human3-macro fix), pairs each seed against the HPS validation reference, and
emits the summary plus a ``checkpoint_manifest.json`` per seed (plan §26 —
no checkpoint payloads are copied).

Usage: python scripts/aggregate_d1_summary.py [--runs_root PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]

# Plan §20: paired HPS validation references (frozen diagnostic baseline).
HPS_REFERENCE_HUMAN3_RMSE = {
    42: 1.163448,
    43: 1.112644,
    44: 0.984083,
}

SUMMARY_FIELDS = [
    "seed",
    "best_epoch",
    "hps_reference_human3_rmse",
    "no_null_best_human3_rmse",
    "paired_delta_rmse",
    "man_rmse",
    "women_rmse",
    "human_rmse",
    "base_human3_rmse",
    "route_human3_rmse",
    "final_human3_rmse",
    "all59_macro_rmse",
    "mean_routing_entropy_human3",
    "mean_routing_variance_human3",
    "mean_routing_variance_joint_human3",
    "mean_transfer_mass_human3",
    "mean_joint_source_mass_human3",
    "unique_top1_sources_human3",
    "mean_router_grad_norm",
    "mean_adapter_grad_norm",
    "mean_film_generator_grad_norm",
    "mean_backbone_grad_norm",
    "status",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_mean(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human3_macro_at(path_metrics: list[dict], epoch: str, path: str, metric: str = "rmse") -> float:
    values = [
        float(row[metric])
        for row in path_metrics
        if row["epoch"] == epoch and row["path"] == path and row["task"] in HUMAN_TASKS
    ]
    return _finite_mean(values) if values else float("nan")


def _final_rmse_per_endpoint(path_metrics: list[dict], epoch: str) -> dict[str, float]:
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


def _routing_human3_at(routing_epoch: list[dict], epoch: str) -> dict[str, float]:
    rows = [row for row in routing_epoch if row["epoch"] == epoch and row["task"] in HUMAN_TASKS]
    fields = (
        "mean_routing_entropy",
        "routing_variance",
        "routing_variance_joint",
        "mean_transfer_mass",
        "mean_joint_source_mass",
        "mean_null_weight",
    )
    return {field: _finite_mean([row[field] for row in rows]) for field in fields} if rows else {}


def _grad_mean_at(gradient_norms: list[dict], epoch: str, group: str) -> float:
    rows = [row for row in gradient_norms if row["epoch"] == epoch and row["group"] == group]
    return _finite_mean([row["mean"] for row in rows]) if rows else float("nan")


def _checkpoint_manifest(seed_dir: Path, ckpt_prefix: str) -> dict:
    manifest = {}
    for label, filename in (
        ("best", f"{ckpt_prefix}_best.pt"),
        ("last", f"{ckpt_prefix}_last.pt"),
    ):
        path = seed_dir / filename
        entry = {"path": str(path)}
        if path.exists():
            entry["sha256"] = _sha256_file(path)
            entry["bytes"] = path.stat().st_size
        else:
            entry["sha256"] = None
            entry["bytes"] = None
        manifest[f"{label}_checkpoint"] = entry
    return manifest


def summarize_seed(seed: int, seed_dir: Path) -> dict:
    diagnostics = seed_dir / "diagnostics"
    epoch_summary = _read_csv(diagnostics / "epoch_summary.csv")
    path_metrics = _read_csv(diagnostics / "human3_path_metrics.csv")
    routing_epoch = _read_csv(diagnostics / "routing_epoch.csv")
    gradient_norms = _read_csv(diagnostics / "gradient_norms.csv")
    source_frequency = {}
    frequency_path = diagnostics / "routing_source_frequency.json"
    if frequency_path.exists():
        source_frequency = json.loads(frequency_path.read_text(encoding="utf-8")).get("tasks", {})

    best_rows = [row for row in epoch_summary if row.get("is_best", "").strip().lower() == "true"]
    if not best_rows:
        return {"seed": seed, "status": "INCOMPLETE_NO_BEST_EPOCH"}
    best_row = max(best_rows, key=lambda row: int(row["epoch"]))
    best_epoch = best_row["epoch"]

    status = "OK"
    def _check(value: float) -> float:
        nonlocal status
        if not math.isfinite(value):
            status = "FAILED_NON_FINITE"
        return value

    endpoint_rmse = _final_rmse_per_endpoint(path_metrics, best_epoch)
    routing = _routing_human3_at(routing_epoch, best_epoch)
    frequencies = source_frequency.get(HUMAN_TASKS[2], {}) if source_frequency else {}

    hps_reference = HPS_REFERENCE_HUMAN3_RMSE[seed]
    best_val = _check(float(best_row["val_human3_macro_rmse"]))
    ckpt_prefix = f"rgcer_d1_no_null_e40_seed{seed}"

    return {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "hps_reference_human3_rmse": hps_reference,
        "no_null_best_human3_rmse": best_val,
        "paired_delta_rmse": _check(hps_reference - best_val),
        "man_rmse": _check(endpoint_rmse.get(HUMAN_TASKS[0], float("nan"))),
        "women_rmse": _check(endpoint_rmse.get(HUMAN_TASKS[1], float("nan"))),
        "human_rmse": _check(endpoint_rmse.get(HUMAN_TASKS[2], float("nan"))),
        "base_human3_rmse": _check(_human3_macro_at(path_metrics, best_epoch, "base")),
        "route_human3_rmse": _check(_human3_macro_at(path_metrics, best_epoch, "route")),
        "final_human3_rmse": _check(_human3_macro_at(path_metrics, best_epoch, "final")),
        "all59_macro_rmse": _check(float(best_row["val_all59_macro_rmse"])),
        "mean_routing_entropy_human3": _check(routing.get("mean_routing_entropy", float("nan"))),
        "mean_routing_variance_human3": _check(routing.get("routing_variance", float("nan"))),
        "mean_routing_variance_joint_human3": _check(routing.get("routing_variance_joint", float("nan"))),
        "mean_transfer_mass_human3": _check(routing.get("mean_transfer_mass", float("nan"))),
        "mean_joint_source_mass_human3": _check(routing.get("mean_joint_source_mass", float("nan"))),
        "unique_top1_sources_human3": frequencies.get("unique_top1_sources", 0),
        "mean_router_grad_norm": _check(_grad_mean_at(gradient_norms, best_epoch, "router_query_projection")),
        "mean_adapter_grad_norm": _check(_grad_mean_at(gradient_norms, best_epoch, "adapter")),
        "mean_film_generator_grad_norm": _check(_grad_mean_at(gradient_norms, best_epoch, "film_generator")),
        "mean_backbone_grad_norm": _check(_grad_mean_at(gradient_norms, best_epoch, "backbone")),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_root",
        default="artifacts/runs/diag_no_null/d1_no_null_e40",
        help="Directory containing one seed_<N> run directory per D1 seed.",
    )
    parser.add_argument("--output", default="DIAG_D1_SUMMARY.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows = []
    for seed in args.seeds:
        seed_dir = runs_root / f"seed_{seed}"
        if not seed_dir.exists():
            rows.append({"seed": seed, "status": "MISSING_RUN_DIR"})
            continue
        rows.append(summarize_seed(seed, seed_dir))

        manifest = _checkpoint_manifest(seed_dir, f"rgcer_d1_no_null_e40_seed{seed}")
        manifest["seed"] = seed
        manifest["best_epoch"] = rows[-1].get("best_epoch")
        (seed_dir / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    paired = [row.get("paired_delta_rmse") for row in rows if isinstance(row.get("paired_delta_rmse"), float)]
    aggregate = {
        "seeds": args.seeds,
        "paired_delta_rmse_mean": _finite_mean([value for value in paired if math.isfinite(value)]),
        "seeds_beating_hps": sum(
            1 for value in paired if math.isfinite(value) and value > 0
        ),
        "gate_d1_a": None,
    }
    routing_alive = all(
        isinstance(row.get("mean_transfer_mass_human3"), float)
        and math.isfinite(row.get("mean_transfer_mass_human3", float("nan")))
        and row["mean_transfer_mass_human3"] > 0
        for row in rows
        if row.get("status") == "OK"
    )
    if aggregate["seeds_beating_hps"] >= 2 and aggregate["paired_delta_rmse_mean"] > 0 and routing_alive:
        aggregate["gate_d1_a"] = "CROSS-ENDPOINT ROUTE IS VIABLE"
    else:
        aggregate["gate_d1_a"] = "SEE PLAN §22 (D1-B / D1-C) — RETURN TO CHATGPT"

    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {output}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
