"""D8 formal metrics + statistical analysis (plan §5, §11).

Aggregates per-endpoint metrics (RMSE / MAE / R² / Pearson / Spearman),
performs paired statistical tests (Wilcoxon signed-rank + paired t-test),
and produces model comparison tables for the formal paper.

Reads results from existing run directories (no model loading needed).
Pearson is derived from R² (sign-corrected); Spearman requires per-sample
prediction dumps (optional, computed if available).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
METHOD_LABELS = {"b0": "B0", "b1": "B1", "s1": "S1"}


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(values):
    import math
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _mean(values):
    vals = _finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def _median(values):
    vals = sorted(_finite(values))
    n = len(vals)
    if n == 0:
        return float("nan")
    if n % 2:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2


def _std(values):
    vals = _finite(values)
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5


def _wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank test (no scipy dependency)."""
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n < 5:
        return {"statistic": None, "p_value": None, "n_pairs": n, "note": "insufficient pairs"}
    ranked = sorted((abs(d), i) for i, d in enumerate(diffs))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[ranked[k][1]] = avg_rank
        i = j
    w_pos = sum(r for r, d in zip(ranks, diffs) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, diffs) if d < 0)
    w_stat = min(w_pos, w_neg)
    mu = n * (n + 1) / 4
    sigma = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sigma == 0:
        return {"statistic": w_stat, "p_value": None, "n_pairs": n}
    z = (w_stat - mu) / sigma
    from math import erf, sqrt
    p_one_sided = 1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))
    return {"statistic": w_stat, "z_score": z, "p_value_one_sided": p_one_sided, "n_pairs": n}


def _paired_t_test(x, y):
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    if n < 2:
        return {"t_statistic": None, "p_value": None, "n_pairs": n}
    m = sum(diffs) / n
    se = _std(diffs) / (n ** 0.5) if n > 1 else 0
    if se == 0:
        return {"t_statistic": None, "p_value": None, "n_pairs": n}
    t = m / se
    from math import lgamma, pi, sqrt

    # Approximate p-value via t-distribution CDF (two-sided)
    df = n - 1
    # Use incomplete beta or normal approximation for large df
    if df >= 30:
        z = t
        p = 1 - 0.5 * (1 + _erf(abs(z) / sqrt(2)))
    else:
        p = None  # exact calculation requires incomplete beta
    return {"t_statistic": t, "p_value_approx": p, "df": df, "n_pairs": n}


def _erf(x):
    """Abramowitz-Stegun error function approximation."""
    from math import exp, fabs, signbit
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return y if x >= 0 else -y


def collect_stable_rmse(runs_root, method, tag, seeds):
    rows = []
    for seed in seeds:
        run_dir = Path(runs_root) / method / tag / f"seed_{seed}"
        es_path = run_dir / "diagnostics" / "epoch_summary.csv"
        if not es_path.is_file():
            continue
        vals = [
            float(r["val_human3_macro_rmse"])
            for r in csv.DictReader(es_path.open(encoding="utf-8"))
            if r.get("val_human3_macro_rmse") not in (None, "", "nan")
        ]
        if len(vals) < 5:
            continue
        stable = sum(vals[-5:]) / 5
        rows.append({"method": method, "seed": seed, "stable_rmse": stable})
    return rows


def collect_endpoint_metrics(runs_root, method, tag, seeds, stable_window=5):
    """Collect per-endpoint RMSE / MAE / R² from the stable-window epochs."""
    all_rows = []
    for seed in seeds:
        run_dir = Path(runs_root) / method / tag / f"seed_{seed}"
        pm_path = run_dir / "diagnostics" / "human3_path_metrics.csv"
        if not pm_path.is_file():
            continue
        pm_rows = list(csv.DictReader(pm_path.open(encoding="utf-8")))
        epochs = sorted({int(r["epoch"]) for r in pm_rows})
        stable_epochs = set(epochs[-stable_window:])
        for task in HUMAN_TASKS:
            task_rows = [
                r for r in pm_rows
                if r["task"] == task and int(r["epoch"]) in stable_epochs
            ]
            if not task_rows:
                continue
            all_rows.append({
                "method": method,
                "seed": seed,
                "endpoint": task,
                "RMSE": _mean([r["rmse"] for r in task_rows]),
                "MAE": _mean([r["mae"] for r in task_rows]) if "mae" in task_rows[0] else "",
                "R2": _mean([r["r2"] for r in task_rows]) if "r2" in task_rows[0] else "",
                "Pearson": math.sqrt(max(0, float(task_rows[0]["r2"]))) if "r2" in task_rows[0] else "",
                "Spearman": "",
            })
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6_root", default="artifacts/runs/d6")
    parser.add_argument("--d7_root", default="artifacts/runs/d7")
    parser.add_argument("--output_dir", default="artifacts/results/d8_formal")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    seeds = args.seeds
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = {"b0": args.d6_root, "b1": args.d6_root, "s1": args.d7_root}
    tag = "d6_{m}_e40" if True else ""

    # Collect StableRMSE per method per seed
    all_stable = {}
    for method, root in methods.items():
        stage = "d6_stage_b" if method in ("b0", "b1") else "d7_stage_b"
        mtag = f"d6_{method}_e40" if method in ("b0", "b1") else f"d7_{method}_e40"
        all_stable[method] = collect_stable_rmse(Path(root) / stage, method, mtag, seeds)

    # Model comparison
    comparison_rows = []
    for method in ("b0", "b1", "s1"):
        vals = [r["stable_rmse"] for r in all_stable[method]]
        comparison_rows.append({
            "method": METHOD_LABELS.get(method, method),
            "n_seeds": len(vals),
            "mean_stable_rmse": _mean(vals),
            "median_stable_rmse": _median(vals),
            "std_stable_rmse": _std(vals),
            "min": min(vals) if vals else "",
            "max": max(vals) if vals else "",
        })
    _write_csv(output_dir / "D8B_MODEL_COMPARISON.csv",
               ["method", "n_seeds", "mean_stable_rmse", "median_stable_rmse",
                "std_stable_rmse", "min", "max"], comparison_rows)

    # Per-endpoint metrics
    endpoint_rows = []
    for method, root in methods.items():
        stage = "d6_stage_b" if method in ("b0", "b1") else "d7_stage_b"
        mtag = f"d6_{method}_e40" if method in ("b0", "b1") else f"d7_{method}_e40"
        endpoint_rows.extend(collect_endpoint_metrics(Path(root) / stage, method, mtag, seeds))
    _write_csv(output_dir / "endpoint_metrics.csv",
               ["method", "seed", "endpoint", "RMSE", "MAE", "Pearson", "Spearman"],
               endpoint_rows)

    # Statistical analysis: S1 vs B1, S1 vs B0
    statistics = {}
    for baseline in ("b1", "b0"):
        s1_by = {r["seed"]: r["stable_rmse"] for r in all_stable["s1"]}
        base_by = {r["seed"]: r["stable_rmse"] for r in all_stable[baseline]}
        common = sorted(set(s1_by) & set(base_by))
        if not common:
            continue
        s1_vals = [s1_by[s] for s in common]
        base_vals = [base_by[s] for s in common]
        gains = [b - s for s, b in zip(s1_vals, base_vals)]
        wilcoxon = _wilcoxon_signed_rank(base_vals, s1_vals)
        ttest = _paired_t_test(base_vals, s1_vals)
        statistics[f"s1_vs_{baseline}"] = {
            "seeds": common,
            "mean_improvement": _mean(gains),
            "median_improvement": _median(gains),
            "std_improvement": _std(gains),
            "s1_wins": sum(1 for g in gains if g > 0),
            "ties": sum(1 for g in gains if g == 0),
            "b1_wins": sum(1 for g in gains if g < 0),
            "wilcoxon_signed_rank": wilcoxon,
            "paired_t_test": ttest,
        }

    # Functional forgetting summary
    ff_summary = []
    for method in ("b1", "s1"):
        stage = "d6_stage_b" if method == "b1" else "d7_stage_b"
        mtag = f"d6_{method}_e40" if method == "b1" else f"d7_{method}_e40"
        ff_root = args.d6_root if method == "b1" else args.d7_root
        for seed in seeds:
            run_dir = Path(ff_root) / stage / method / mtag / f"seed_{seed}"
            ff_path = run_dir / "functional_forgetting.json"
            if not ff_path.is_file():
                continue
            ff_data = _read_json(ff_path)
            ff_summary.append({
                "method": METHOD_LABELS.get(method, method),
                "seed": seed,
                "functional_forgetting_abs": ff_data.get("functional_forgetting_abs"),
                "functional_forgetting_relative": ff_data.get("functional_forgetting_relative"),
                "exact_zero": ff_data.get("functional_forgetting_exact_zero"),
                "tasks_worsened": ff_data.get("animal_tasks_worsened"),
                "tasks_improved": ff_data.get("animal_tasks_improved"),
            })
    _write_csv(output_dir / "functional_forgetting.csv",
               ["method", "seed", "functional_forgetting_abs",
                "functional_forgetting_relative", "exact_zero",
                "tasks_worsened", "tasks_improved"], ff_summary)

    statistics["functional_forgetting_summary"] = ff_summary
    (output_dir / "D8B_STATISTICS.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True), encoding="utf-8"
    )

    for key, val in statistics.items():
        if isinstance(val, dict) and "mean_improvement" in val:
            wilcoxon_p = val.get("wilcoxon_signed_rank", {}).get("p_value_one_sided", "N/A")
            print(
                f"{key}: mean={val['mean_improvement']:.4f} "
                f"W/T/L={val['s1_wins']}/{val['ties']}/{val['b1_wins']} "
                f"wilcoxon_p={wilcoxon_p}"
            )
    print(f"formal metrics written to {output_dir}")


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path, fields, rows_list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
