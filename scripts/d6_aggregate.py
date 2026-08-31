"""D6 Progressive Micro-Screening aggregation and auditable selection (§34, §73-§76).

Modes:
- stage-a : rank B1/O1/O2/O3 by StableRMSE_20 (mean of the last five epochs)
  against the B0 Human3-only reference; apply the §31 elimination and §33
  originality minimum gates; select Top-2 per §32.
- stage-b : three-seed (42/44/46) confirmation of the Stage-A Top-2 with the
  B0 paired reference; StableGain gates §43-§44; Top-1 per §45-§46.
- stage-c : five-seed full-budget confirmation of the Stage-B Top-1 with the
  §51-§52 gates.

Every decision (thresholds, rankings, ties, reasons) is written to
D6_SELECTION_TRACE.json so the selection is auditable (§76).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
CANDIDATES = {
    "b0": {"originality_eligible": False},
    "b1": {"originality_eligible": False},
    "o1": {"originality_eligible": True},
    "o2": {"originality_eligible": True},
    "o3": {"originality_eligible": True},
}
# §46: minimal-change preference order for near ties.
PREFERENCE_ORDER = ["o1", "o2", "o3"]

STABLE_WINDOW = 5
ELIMINATE_MACRO_MARGIN = 0.05
ELIMINATE_ENDPOINT_MARGIN = 0.05
ORIGINAL_MIN_GAIN = 0.01
STRONG_GAIN = 0.02
TIE_MARGIN = 0.01


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path}")


def _finite_mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def candidate_run_dir(runs_root: Path, candidate: str, tag: str) -> Path | None:
    base = runs_root / candidate / tag
    if base.exists():
        return base
    matches = sorted((runs_root / candidate).glob(f"{tag}/seed_*"))
    return matches[0].parent if matches else None


def trajectory(run_dir: Path) -> list[dict]:
    rows = _read_csv(run_dir / "diagnostics" / "epoch_summary.csv")
    parsed = [
        {"epoch": int(row["epoch"]), "val": float(row["val_human3_macro_rmse"])}
        for row in rows
        if row.get("val_human3_macro_rmse") not in (None, "", "nan")
    ]
    return sorted(parsed, key=lambda row: row["epoch"])


def endpoint_stable(run_dir: Path, epochs: set[int]) -> dict[str, float]:
    rows = [
        row
        for row in _read_csv(run_dir / "diagnostics" / "human3_path_metrics.csv")
        if row["path"] == "final" and int(row["epoch"]) in epochs
    ]
    return {
        task: _finite_mean([row["rmse"] for row in rows if row["task"] == task])
        for task in HUMAN_TASKS
    }


def summarize_run(run_dir: Path | None, candidate: str, seed: int) -> dict:
    row: dict = {
        "candidate": candidate,
        "originality_eligible": CANDIDATES.get(candidate, {}).get("originality_eligible", False),
        "seed": seed,
    }
    if run_dir is None or not run_dir.exists():
        row["status"] = "MISSING_RUN"
        return row
    trajectory_rows = trajectory(run_dir)
    if len(trajectory_rows) < STABLE_WINDOW:
        row["status"] = "INCOMPLETE_TRAJECTORY"
        return row
    values = [entry["val"] for entry in trajectory_rows]
    window = values[-STABLE_WINDOW:]
    stable = float(np_mean(window))
    best = min(values)
    best_index = values.index(best)
    local = values[max(0, best_index - 2) : best_index + 3]
    epochs = {entry["epoch"] for entry in trajectory_rows[-STABLE_WINDOW:]}
    endpoints = endpoint_stable(run_dir, epochs)
    row.update(
        {
            "status": "PASS",
            "stable_rmse": stable,
            "best_rmse": best,
            "best_epoch": trajectory_rows[best_index]["epoch"],
            "best_sharpness": float(np_mean(local)) - best,
            "trajectory_std": float(np_std(values)),
            "man_rmse": endpoints[HUMAN_TASKS[0]],
            "women_rmse": endpoints[HUMAN_TASKS[1]],
            "human_rmse": endpoints[HUMAN_TASKS[2]],
        }
    )
    if any(not math.isfinite(row[key]) for key in ("stable_rmse", "man_rmse", "women_rmse", "human_rmse")):
        row["status"] = "FAILED_NON_FINITE"
    return row


def np_mean(values):
    return sum(values) / len(values)


def np_std(values):
    mean = np_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1))


def _pcts_of_best(values: list[float], best: float) -> int:
    return sum(1 for value in values if value <= best * 1.02)


def stage_a(runs_root: Path, seed: int, output_dir: Path, trace: dict) -> list[str]:
    rows = []
    b0_row = None
    for candidate in CANDIDATES:
        tag = f"d6_{candidate}_e20"
        run_dir = candidate_run_dir(runs_root, candidate, tag)
        row = summarize_run(run_dir, candidate, seed)
        if row.get("status") != "PASS":
            rows.append(row)
            continue
        if candidate == "b0":
            b0_row = row
        rows.append(row)

    if b0_row is None or b0_row.get("status") != "PASS":
        raise SystemExit("Stage A requires a PASS B0 Human3-only reference run")
    b0_stable = float(b0_row["stable_rmse"])
    b0_endpoints = {task: float(b0_row[f"{task.split('_')[0]}_rmse"]) for task in HUMAN_TASKS}

    for row in rows:
        if row.get("status") != "PASS":
            continue
        endpoints = {task: float(row[f"{task.split('_')[0]}_rmse"]) for task in HUMAN_TASKS}
        worse_endpoints = sum(
            1 for task in HUMAN_TASKS if endpoints[task] >= b0_endpoints[task] + ELIMINATE_ENDPOINT_MARGIN
        )
        row["stable_gain_vs_human3_only"] = b0_stable - float(row["stable_rmse"])
        reasons = []
        if float(row["stable_rmse"]) >= b0_stable + ELIMINATE_MACRO_MARGIN:
            reasons.append("stable_rmse worse than B0 by >=0.05")
        if worse_endpoints >= 2:
            reasons.append(">=2 endpoints worse by >=0.05")
        if reasons:
            row["status"] = "ELIMINATED"
            row["elimination_reason"] = "; ".join(reasons)
        elif CANDIDATES[row["candidate"]]["originality_eligible"]:
            endpoint_improved = sum(
                1 for task in HUMAN_TASKS if endpoints[task] < b0_endpoints[task]
            )
            gain = b0_stable - float(row["stable_rmse"])
            row["meets_original_minimum"] = bool(gain >= ORIGINAL_MIN_GAIN or endpoint_improved >= 2)
        row["b0_endpoints"] = b0_endpoints

    ranked = sorted(
        (
            row
            for row in rows
            if row.get("status") == "PASS" and row["candidate"] != "b0"
        ),
        key=lambda row: row["stable_rmse"],
    )
    eligible = [
        row
        for row in ranked
        if not CANDIDATES[row["candidate"]]["originality_eligible"]
        or row.get("meets_original_minimum")
    ]
    top2 = [row["candidate"] for row in eligible[:2]]
    while len(top2) < 2:
        top2.append("NONE")
    trace["stage_a"] = {
        "seed": seed,
        "stable_window": "last 5 epochs of 20 (epochs 15-19)",
        "b0_stable_rmse": b0_stable,
        "ranking": [
            {"candidate": row["candidate"], "stable_rmse": row["stable_rmse"],
             "status": row.get("status"), "meets_original_minimum": row.get("meets_original_minimum")}
            for row in ranked
        ],
        "thresholds": {
            "eliminate_macro_margin": ELIMINATE_MACRO_MARGIN,
            "eliminate_endpoint_margin": ELIMINATE_ENDPOINT_MARGIN,
            "original_min_gain": ORIGINAL_MIN_GAIN,
        },
        "top2": top2,
        "top2_reason": "§32: top-2 by StableRMSE among B1/O1/O2/O3 that pass §33 original minimum",
    }

    fields = ["candidate", "originality_eligible", "seed", "status", "elimination_reason",
              "stable_rmse", "best_rmse", "best_epoch", "best_sharpness", "trajectory_std",
              "man_rmse", "women_rmse", "human_rmse", "stable_gain_vs_human3_only",
              "meets_original_minimum"]
    _write_csv(output_dir / "D6A_MICROSCREEN_SUMMARY.csv", fields, rows)

    endpoint_rows = []
    for row in rows:
        for task in HUMAN_TASKS:
            endpoint_rows.append(
                {
                    "candidate": row["candidate"],
                    "seed": row["seed"],
                    "task": task,
                    "stable_rmse": row.get(f"{task.split('_')[0]}_rmse"),
                    "b0_stable_rmse": b0_endpoints[task],
                    "delta_vs_b0": (b0_endpoints[task] - row[f"{task.split('_')[0]}_rmse"])
                    if row.get("status") == "PASS"
                    else float("nan"),
                }
            )
    _write_csv(
        output_dir / "D6A_ENDPOINT_RESULTS.csv",
        ["candidate", "seed", "task", "stable_rmse", "b0_stable_rmse", "delta_vs_b0"],
        endpoint_rows,
    )
    return top2


def stage_b(runs_root: Path, candidates: list[str], seeds: list[int], output_dir: Path, trace: dict) -> str | None:
    summary_rows = []
    paired_rows = []
    stable_by_candidate: dict[str, dict[int, float]] = {}
    b0_by_seed: dict[int, float] = {}
    for candidate in candidates + ["b0"]:
        tag = f"d6_{candidate}_e40"
        for seed in seeds:
            run_dir = candidate_run_dir(runs_root, candidate, tag)
            row = summarize_run(candidate_run_dir(runs_root, candidate, tag), candidate, seed)
            summary_rows.append(row)
            if row.get("status") == "PASS":
                stable_by_candidate.setdefault(candidate, {})[seed] = float(row["stable_rmse"])
                if candidate == "b0":
                    b0_by_seed[seed] = float(row["stable_rmse"])

    gate_rows = []
    verdicts = []
    for candidate in candidates:
        per_seed = stable_by_candidate.get(candidate, {})
        paired = [
            {
                "seed": seed,
                "b0_stable_rmse": b0_by_seed[seed],
                "candidate_stable_rmse": per_seed[seed],
                "stable_gain": b0_by_seed[seed] - per_seed[seed],
            }
            for seed in seeds
            if seed in per_seed and seed in b0_by_seed
        ]
        paired_rows.extend(paired)
        gains = [entry["stable_gain"] for entry in paired]
        mean_gain = _finite_mean(gains)
        positive = sum(1 for gain in gains if gain > 0)
        endpoint_means = {
            task: _finite_mean(
                [
                    float(row[f"{task.split('_')[0]}_rmse"])
                    for row in summary_rows
                    if row["candidate"] == candidate and row.get("status") == "PASS"
                ]
            )
            for task in HUMAN_TASKS
        }
        b0_endpoint_means = {
            task: _finite_mean(
                [
                    float(row[f"{task.split('_')[0]}_rmse"])
                    for row in summary_rows
                    if row["candidate"] == "b0" and row.get("status") == "PASS"
                ]
            )
            for task in HUMAN_TASKS
        }
        endpoints_non_worse = sum(
            1 for task in HUMAN_TASKS if endpoint_means[task] <= b0_endpoint_means[task] + 1e-9
        )
        gate_pass = bool(
            mean_gain > 0
            and positive >= 2
            and endpoints_non_worse >= 2
            and len(gains) == len(seeds)
        )
        verdict = "STRONG" if gate_pass and mean_gain >= STRONG_GAIN else (
            "BORDERLINE" if gate_pass else "FAIL"
        )
        verdicts.append((candidate, mean_gain, positive, endpoints_non_worse, gate_pass, verdict))
        gate_rows.append(
            {
                "candidate": candidate,
                "mean_stable_gain": mean_gain,
                "positive_seeds": positive,
                "seeds_compared": len(gains),
                "endpoints_non_worse": endpoints_non_worse,
                "gate": verdict,
            }
        )

    original_gate_pass = [entry for entry in verdicts if CANDIDATES[entry[0]]["originality_eligible"] and entry[4]]
    top1 = None
    if original_gate_pass:
        strong = [entry for entry in original_gate_pass if entry[5] == "STRONG"]
        pool = strong or [
            entry for entry in original_gate_pass if entry[1] is not None and entry[1] > 0
        ]
        pool_sorted = sorted(pool, key=lambda entry: stable_by_candidate.get(entry[0], {}).get("mean", 9e9)) if pool else []
        means = {
            candidate: _finite_mean(list(stable_by_candidate.get(candidate, {}).values()))
            for candidate, *_ in pool
        }
        pool_sorted = sorted(pool, key=lambda entry: means[entry[0]])
        top1 = pool_sorted[0][0]
        if len(pool_sorted) >= 2:
            best_mean, second_mean = means[pool_sorted[0][0]], means[pool_sorted[1][0]]
            if abs(best_mean - second_mean) < TIE_MARGIN:
                preference = [name for name in PREFERENCE_ORDER if name in {entry[0] for entry in pool_sorted[:2]}]
                top1 = preference[0] if preference else top1
                trace.setdefault("stage_b", {})["tie_break"] = {
                    "means": means, "applied_preference": PREFERENCE_ORDER, "top1": top1,
                }

    trace["stage_b"] = {
        "seeds": seeds,
        "ranking": [
            {"candidate": candidate, "mean_stable_gain": mean_gain, "positive_seeds": positive,
             "endpoints_non_worse": endpoints_non_worse, "gate": verdict}
            for candidate, mean_gain, positive, endpoints_non_worse, gate_pass, verdict in verdicts
        ],
        "thresholds": {"strong_gain": STRONG_GAIN, "tie_margin": TIE_MARGIN,
                        "preference_order": PREFERENCE_ORDER},
        "top1": top1,
        "top1_reason": "§43-§46" if top1 else "no original candidate passed the Stage-B gate (§81 STOP)",
        "novelty_candidates_failed": top1 is None,
    }

    _write_csv(
        output_dir / "D6B_TOP2_3SEED_SUMMARY.csv",
        ["candidate", "seed", "status", "stable_rmse", "best_rmse", "best_epoch",
         "best_sharpness", "trajectory_std", "man_rmse", "women_rmse", "human_rmse"],
        summary_rows,
    )
    _write_csv(
        output_dir / "D6B_PAIRED_COMPARISON.csv",
        ["candidate", "seed", "b0_stable_rmse", "candidate_stable_rmse", "stable_gain"],
        paired_rows,
    )
    _write_csv(
        output_dir / "D6B_GATE.csv",
        ["candidate", "mean_stable_gain", "positive_seeds", "seeds_compared",
         "endpoints_non_worse", "gate"],
        gate_rows,
    )
    return top1


def stage_c(runs_root: Path, top1: str, seeds: list[int], output_dir: Path, trace: dict) -> None:
    summary_rows = []
    endpoint_rows = []
    task_stable: dict[int, float] = {}
    b0_stable: dict[int, float] = {}
    for candidate in (top1, "b0"):
        tag = f"d6_{candidate}_e100"
        for seed in seeds:
            run_dir = candidate_run_dir(runs_root, candidate, tag)
            row = summarize_run(run_dir, candidate, seed)
            summary_rows.append(row)
            if row.get("status") != "PASS":
                continue
            stable = float(row["stable_rmse"])
            if candidate == top1:
                task_stable[seed] = stable
            else:
                b0_stable[seed] = stable
            for task in HUMAN_TASKS:
                endpoint_rows.append(
                    {
                        "model": candidate,
                        "seed": seed,
                        "task": task,
                        "stable_rmse": float(row[f"{task.split('_')[0]}_rmse"]),
                    }
                )
    gains = [task_stable[seed] - b0_stable[seed] for seed in seeds if seed in task_stable and seed in b0_stable]
    gate = {
        "paired_mean_stable_gain": _finite_mean(gains),
        "positive_seeds": sum(1 for gain in gains if gain > 0),
        "seeds_compared": len(gains),
        "gate_pass": bool(
            _finite_mean(gains) > 0
            and (
                sum(1 for gain in gains if gain > 0) >= 4
                or (sum(1 for gain in gains if gain > 0) >= 3 and _finite_mean(gains) >= STRONG_GAIN)
            )
        ),
    }
    trace["stage_c"] = {"top1": top1, "seeds": seeds, "gate": gate}
    _write_csv(
        output_dir / "D6C_FINAL_5SEED_SUMMARY.csv",
        ["candidate", "seed", "status", "stable_rmse", "best_rmse", "best_epoch",
         "best_sharpness", "trajectory_std", "man_rmse", "women_rmse", "human_rmse"],
        summary_rows,
    )
    _write_csv(
        output_dir / "D6C_ENDPOINT_SUMMARY.csv",
        ["model", "seed", "task", "stable_rmse"],
        endpoint_rows,
    )
    _write_csv(
        output_dir / "D6C_STABILITY_SUMMARY.csv",
        ["model", "seed", "stable_rmse", "best_rmse", "best_epoch", "best_sharpness", "trajectory_std"],
        [
            {key: row.get(key) for key in ("candidate", "seed", "stable_rmse", "best_rmse", "best_epoch", "best_sharpness", "trajectory_std")}
            for row in summary_rows
        ],
    )
    print(json.dumps({"stage_c_gate": gate}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage-a", "stage-b", "stage-c"])
    parser.add_argument("--runs_root", default="artifacts/runs")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--top2", nargs="+", default=None, help="Stage-B candidates (from Stage A)")
    parser.add_argument("--top1", default=None, help="Stage-C candidate (from Stage B)")
    parser.add_argument("--csdt_csv", default=None)
    parser.add_argument("--clst_csv", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "D6_SELECTION_TRACE.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}

    if args.mode == "stage-a":
        top2 = stage_a(Path(args.runs_root), args.seed, output_dir, trace)
        print(f"Stage A Top-2: {top2}")
    elif args.mode == "stage-b":
        candidates = args.top2 or []
        if not candidates:
            raise SystemExit("stage-b requires --top2 from Stage A")
        top1 = stage_b(Path(args.runs_root), candidates, [42, 44, 46], output_dir, trace)
        print(f"Stage B Top-1: {top1}")
        if top1 is None:
            print("STOP: no original candidate passed the Stage-B gate (plan §81).")
    else:
        if not args.top1:
            raise SystemExit("stage-c requires --top1 from Stage B")
        stage_c(Path(args.runs_root), args.top1, [42, 43, 44, 45, 46], output_dir, trace)
        print("Stage C complete; STOP before any test/CQR (plan §82).")

    if args.csdt_csv:
        trace.setdefault("artifacts", {})["csdt_parameter_delta_csv"] = args.csdt_csv
    if args.clst_csv:
        trace.setdefault("artifacts", {})["clst_layer_score_csv"] = args.clst_csv
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    print(f"updated {trace_path}")


if __name__ == "__main__":
    main()
