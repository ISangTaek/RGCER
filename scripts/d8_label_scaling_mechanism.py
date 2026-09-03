"""D8 result-correction Task D: label-scaling mechanism (drift + forgetting)
and the Figure 4 master tables.

The original "low-resource advantage" expectation was NOT supported: at 10%
labels B1 is slightly better, at 25% roughly equal, and from 50% upward S1
is stably better.  Figure 4 is therefore re-framed as a target-data-dependent
stability-plasticity transition, which requires the mechanism columns
(B1 drift + Animal56 functional forgetting) for every fraction x seed.

Steps
-----
1. For every B1 scaling run (fraction x seed) without a
   ``functional_forgetting.json``, run the existing reviewed evaluator
   ``scripts/d8_functional_forgetting.py`` (matched-teacher provenance,
   Animal56 validation only).  The 100% points reuse the formal runs'
   existing results.  No training is repeated (plan §26/§62).
2. Aggregate the evaluator JSONs into
   forgetting/D8_LABEL_SCALING_FUNCTIONAL_FORGETTING.csv and
   forgetting/D8_LABEL_SCALING_ANIMAL_ENDPOINT_FORGETTING.csv.
3. Join performance (stable RMSE) + drift (Task D drift CSV) + forgetting
   into figure4/D8_FIGURE4_MASTER_TABLE.csv and
   figure4/D8_FIGURE4_FRACTION_SUMMARY.csv.
4. Exploratory correlations (gain vs drift/forgetting across fraction
   means) — explicitly labelled descriptive/exploratory (§36).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
STABLE_WINDOW = 5


def stable_endpoint_rmse(run_dir: Path) -> dict | None:
    path = run_dir / "diagnostics" / "human3_path_metrics.csv"
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    epochs = sorted({int(r["epoch"]) for r in rows})
    if not epochs or epochs[-1] != 39:
        return None
    keep = set(epochs[-STABLE_WINDOW:])
    per_task = {}
    for task in HUMAN_TASKS:
        values = [
            float(r["rmse"]) for r in rows
            if r["task"] == task and int(r["epoch"]) in keep
        ]
        if len(values) < STABLE_WINDOW:
            return None
        per_task[task] = sum(values) / len(values)
    return per_task


def scaling_run_dir(scaling_root: Path, fraction: int, seed: int) -> Path:
    return scaling_root / "b1" / f"d8_b1_f{fraction}_e40" / f"seed_{seed}"


def formal_b1_run_dir(d6_root: Path, seed: int) -> Path:
    return Path(d6_root) / "d6_stage_b" / "b1" / "d6_b1_e40" / f"seed_{seed}"


def run_functional_forgetting(run_dir: Path, gpu_id: str, log_path: Path) -> bool:
    """Invoke the existing evaluator; True on success."""
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, "scripts/d8_functional_forgetting.py",
             "--run_dir", str(run_dir), "--gpu_id", gpu_id],
            stdout=log, stderr=subprocess.STDOUT,
        )
    return completed.returncode == 0


def load_ff_summary(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "teacher_animal56_macro_rmse": data.get("teacher_animal56_macro_rmse"),
        "current_animal56_macro_rmse": data.get("current_animal56_macro_rmse"),
        "functional_forgetting_abs": data.get("functional_forgetting_abs"),
        "functional_forgetting_relative": data.get("functional_forgetting_relative"),
        "animal_tasks_worsened": data.get("animal_tasks_worsened"),
        "animal_tasks_improved": data.get("animal_tasks_improved"),
        "animal_tasks_unchanged": data.get("animal_tasks_unchanged"),
        "delta_max_abs": data.get("delta_max_abs"),
        "provenance_verified": data.get("provenance_verified"),
    }


def mean_std_median(values):
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, std, median


def exploratory_correlations(fraction_summary: list[dict]) -> dict:
    """gain vs mechanism across the five fraction MEANS — n=5, descriptive
    only (plan §36: no causal claim)."""
    from scipy import stats

    def series(field):
        return [row[field] for row in fraction_summary]

    gains = series("gain_mean")
    result = {}
    for field in ("feature_drift_mean", "backbone_param_drift_mean",
                  "functional_forgetting_abs_mean"):
        values = series(field)
        if any(not math.isfinite(v) for v in values + gains):
            continue
        pearson = stats.pearsonr(gains, values)
        spearman = stats.spearmanr(gains, values)
        result[f"gain_vs_{field}"] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "n_points": len(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6_root", default="artifacts/runs/d6")
    parser.add_argument("--d7_root", default="artifacts/runs/d7")
    parser.add_argument("--scaling_root", default="artifacts/runs/d8_label_scaling")
    parser.add_argument(
        "--drift_csv", default="artifacts/results/d8_result_correction/drift/D8_LABEL_SCALING_B1_DRIFT.csv"
    )
    parser.add_argument(
        "--output_dir", default="artifacts/results/d8_result_correction"
    )
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--fractions", nargs="+", type=int, default=[10, 25, 50, 75, 100])
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="skip evaluator runs whose functional_forgetting.json already exists",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "forgetting").mkdir(parents=True, exist_ok=True)
    (output_dir / "figure4").mkdir(parents=True, exist_ok=True)

    # ---- step 1: functional forgetting for every fraction x seed ----------
    for fraction in args.fractions:
        for seed in args.seeds:
            if fraction == 100:
                run_dir = formal_b1_run_dir(args.d6_root, seed)
            else:
                run_dir = scaling_run_dir(Path(args.scaling_root), fraction, seed)
            ff_path = run_dir / "functional_forgetting.json"
            if args.skip_existing and ff_path.is_file():
                continue
            print(f"evaluating functional forgetting: f{fraction} s{seed} ...")
            log_path = output_dir / "forgetting" / f"ff_f{fraction}_s{seed}.log"
            ok = run_functional_forgetting(run_dir, args.gpu_id, log_path)
            if not ok:
                raise SystemExit(
                    f"STOP: functional forgetting failed for f{fraction} s{seed} "
                    f"(see {log_path})"
                )

    # ---- step 2: aggregate evaluator JSONs --------------------------------
    ff_rows = []
    endpoint_rows = []
    for fraction in args.fractions:
        for seed in args.seeds:
            if fraction == 100:
                run_dir = formal_b1_run_dir(args.d6_root, seed)
            else:
                run_dir = scaling_run_dir(Path(args.scaling_root), fraction, seed)
            ff_path = run_dir / "functional_forgetting.json"
            if not ff_path.is_file():
                raise SystemExit(f"STOP: missing {ff_path} (plan §62: report, do not retrain)")
            summary = load_ff_summary(ff_path)
            ff_rows.append({"fraction": fraction / 100.0, "seed": seed, **summary})
            raw = json.loads(ff_path.read_text(encoding="utf-8"))
            teacher = raw.get("per_task_rmse_teacher") or {}
            current = raw.get("per_task_rmse_current") or {}
            for task in sorted(set(teacher) & set(current)):
                endpoint_rows.append(
                    {
                        "fraction": fraction / 100.0,
                        "seed": seed,
                        "animal_endpoint": task,
                        "rmse_teacher": f"{float(teacher[task]):.8f}",
                        "rmse_current": f"{float(current[task]):.8f}",
                        "delta": f"{float(current[task]) - float(teacher[task]):.8f}",
                    }
                )

    def _write(path: Path, fieldnames, rows_list):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_list)
        print(f"wrote {path} ({len(rows_list)} rows)")

    _write(
        output_dir / "forgetting" / "D8_LABEL_SCALING_FUNCTIONAL_FORGETTING.csv",
        ["fraction", "seed", "teacher_animal56_macro_rmse", "current_animal56_macro_rmse",
         "functional_forgetting_abs", "functional_forgetting_relative",
         "animal_tasks_worsened", "animal_tasks_improved", "animal_tasks_unchanged",
         "delta_max_abs", "provenance_verified"],
        ff_rows,
    )
    _write(
        output_dir / "forgetting" / "D8_LABEL_SCALING_ANIMAL_ENDPOINT_FORGETTING.csv",
        ["fraction", "seed", "animal_endpoint", "rmse_teacher", "rmse_current", "delta"],
        endpoint_rows,
    )

    # ---- step 3: Figure 4 master table (performance + drift + forgetting) -
    performance = {}
    for fraction in args.fractions:
        for seed in args.seeds:
            if fraction == 100:
                b1_dir = formal_b1_run_dir(args.d6_root, seed)
            else:
                b1_dir = scaling_run_dir(Path(args.scaling_root), fraction, seed)
            b1_metrics = stable_endpoint_rmse(b1_dir)
            if b1_metrics is None:
                raise SystemExit(f"STOP: missing/incomplete metrics in {b1_dir}")
            performance[(fraction, seed)] = b1_metrics

    s1_by_seed = {}
    for seed in args.seeds:
        s1_dir = Path(args.d7_root) / "d7_stage_b" / "s1" / "d7_s1_e40" / f"seed_{seed}"
        s1_metrics = stable_endpoint_rmse(s1_dir)
        if s1_metrics is None:
            raise SystemExit(f"STOP: missing S1 formal metrics in {s1_dir}")
        s1_by_seed[seed] = s1_metrics

    drift_by_key = {}
    drift_path = Path(args.drift_csv)
    if drift_path.is_file():
        for row in csv.DictReader(drift_path.open(encoding="utf-8")):
            drift_by_key[(round(float(row["fraction"]), 2), int(row["seed"]))] = row

    master_rows = []
    for fraction in args.fractions:
        for seed in args.seeds:
            b1_macro = sum(performance[(fraction, seed)].values()) / len(HUMAN_TASKS)
            s1_macro = sum(s1_by_seed[seed].values()) / len(HUMAN_TASKS)
            drift = drift_by_key.get((round(fraction / 100.0, 2), seed), {})
            ff = next(
                (r for r in ff_rows if r["fraction"] == fraction / 100.0 and r["seed"] == seed),
                {},
            )
            master_rows.append(
                {
                    "fraction": fraction / 100.0,
                    "seed": seed,
                    "b1_stable_rmse": f"{b1_macro:.6f}",
                    "s1_stable_rmse": f"{s1_macro:.6f}",
                    "gain_b1_minus_s1": f"{b1_macro - s1_macro:.6f}",
                    "b1_backbone_param_drift": drift.get("backbone_param_drift", ""),
                    "b1_feature_drift": drift.get("feature_drift", ""),
                    "b1_functional_forgetting_abs": ff.get("functional_forgetting_abs", ""),
                    "b1_functional_forgetting_relative": ff.get("functional_forgetting_relative", ""),
                    "animal_tasks_worsened": ff.get("animal_tasks_worsened", ""),
                }
            )
    _write(
        output_dir / "figure4" / "D8_FIGURE4_MASTER_TABLE.csv",
        ["fraction", "seed", "b1_stable_rmse", "s1_stable_rmse", "gain_b1_minus_s1",
         "b1_backbone_param_drift", "b1_feature_drift",
         "b1_functional_forgetting_abs", "b1_functional_forgetting_relative",
         "animal_tasks_worsened"],
        master_rows,
    )

    # ---- step 4: fraction summary -----------------------------------------
    summary_rows = []
    for fraction in args.fractions:
        frac = fraction / 100.0
        seeds = args.seeds
        gain_mean, gain_std, gain_median = mean_std_median(
            [float(r["gain_b1_minus_s1"]) for r in master_rows if r["fraction"] == frac]
        )
        fd_mean, fd_std, fd_median = mean_std_median(
            [float(r["b1_feature_drift"]) for r in master_rows
             if r["fraction"] == frac and r["b1_feature_drift"] != ""]
        )
        pd_mean, pd_std, pd_median = mean_std_median(
            [float(r["b1_backbone_param_drift"]) for r in master_rows
             if r["fraction"] == frac and r["b1_backbone_param_drift"] != ""]
        )
        ff_mean, ff_std, ff_median = mean_std_median(
            [float(r["b1_functional_forgetting_abs"]) for r in master_rows
             if r["fraction"] == frac and r["b1_functional_forgetting_abs"] != ""]
        )
        summary_rows.append(
            {
                "fraction": frac,
                "n_seeds": len(seeds),
                "gain_mean": f"{gain_mean:.6f}",
                "gain_std": f"{gain_std:.6f}",
                "gain_median": f"{gain_median:.6f}",
                "feature_drift_mean": f"{fd_mean:.6f}",
                "feature_drift_std": f"{fd_std:.6f}",
                "feature_drift_median": f"{fd_median:.6f}",
                "backbone_param_drift_mean": f"{pd_mean:.6f}",
                "backbone_param_drift_std": f"{pd_std:.6f}",
                "backbone_param_drift_median": f"{pd_median:.6f}",
                "functional_forgetting_abs_mean": f"{ff_mean:.6f}",
                "functional_forgetting_abs_std": f"{ff_std:.6f}",
                "functional_forgetting_abs_median": f"{ff_median:.6f}",
            }
        )
    _write(
        output_dir / "figure4" / "D8_FIGURE4_FRACTION_SUMMARY.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )

    correlations = exploratory_correlations(summary_rows)
    correlations["note"] = (
        "descriptive/exploratory only: Pearson/Spearman across the five "
        "fraction means (n=5); no causal claim (plan §36)"
    )
    (output_dir / "figure4" / "D8_FIGURE4_CORRELATIONS.json").write_text(
        json.dumps(correlations, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {output_dir / 'figure4' / 'D8_FIGURE4_CORRELATIONS.json'}")
    for key, value in correlations.items():
        if isinstance(value, dict):
            print(
                f"{key}: pearson_r={value['pearson_r']:.4f} "
                f"spearman_rho={value['spearman_rho']:.4f}"
            )


if __name__ == "__main__":
    main()
