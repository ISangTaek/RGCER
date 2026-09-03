"""D8 result-correction Task B: exact one-sided paired statistics, S1 vs B1.

The formal-phase report used a normal-approximation Wilcoxon p (~0.0216).
With n = 5 paired seeds the plan requires the EXACT one-sided signed-rank
test (§11-§16): when all five gains are positive the exact p is 1/2^5 =
0.03125.  scipy.stats.wilcoxon(..., alternative="greater", method="exact")
is used when available; otherwise an exact sign-flip enumeration with
average ranks (identical to scipy's exact method for this n) is used.

Also reports the exact one-sided sign test as a supplementary cross-check
and the full gain distribution (§15).

Output: metrics/D8_S1_VS_B1_PAIRED_STATS.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv


def _average_ranks(values):
    """Average ranks (1-based) with tie handling, ascending."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = average
        i = j
    return ranks


def exact_wilcoxon_p_one_sided(gains) -> float:
    """Exact P(W+ >= observed) under the null sign-flip distribution.

    Zeros are dropped (scipy convention), |gains| get average ranks, and all
    2^n sign patterns are enumerated — feasible because n = 5 here.
    """
    nonzero = [g for g in gains if g != 0]
    n = len(nonzero)
    if n == 0:
        return float("nan")
    ranks = _average_ranks([abs(g) for g in nonzero])
    w_observed = sum(rank for rank, g in zip(ranks, nonzero) if g > 0)
    count = 0
    total = 0
    for mask in range(1 << n):
        w = sum(rank for bit, rank in enumerate(ranks) if (mask >> bit) & 1)
        total += 1
        if w >= w_observed - 1e-12:
            count += 1
    return count / total


def exact_sign_test_p_one_sided(gains) -> float:
    """Exact one-sided sign test: P(X >= k) for X ~ Bin(n_nonzero, 0.5)."""
    nonzero = [g for g in gains if g != 0]
    n = len(nonzero)
    k = sum(1 for g in nonzero if g > 0)
    if n == 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def compute_paired_stats(gains, seeds, engine_note) -> dict:
    arr = [float(g) for g in gains]
    wilcoxon_p = None
    try:
        from scipy import stats

        wilcoxon = stats.wilcoxon(arr, alternative="greater", method="exact")
        wilcoxon_p = float(wilcoxon.pvalue)
        engine = f"scipy ({engine_note})"
    except (ImportError, TypeError, ValueError):
        wilcoxon_p = exact_wilcoxon_p_one_sided(arr)
        engine = "exact sign-flip enumeration (scipy API unavailable)"
    return {
        "comparison": "S1_vs_B1",
        "metric": "StableRMSE",
        "seeds": list(seeds),
        "gains": arr,
        "mean_gain": sum(arr) / len(arr),
        "median_gain": sorted(arr)[len(arr) // 2] if len(arr) % 2 else
        (sorted(arr)[len(arr) // 2 - 1] + sorted(arr)[len(arr) // 2]) / 2,
        "std_gain": (
            (sum((g - sum(arr) / len(arr)) ** 2 for g in arr) / (len(arr) - 1)) ** 0.5
            if len(arr) > 1 else 0.0
        ),
        "min_gain": min(arr),
        "max_gain": max(arr),
        "wins": sum(1 for g in arr if g > 0),
        "ties": sum(1 for g in arr if g == 0),
        "losses": sum(1 for g in arr if g < 0),
        "wilcoxon_exact_one_sided_p": wilcoxon_p,
        "wilcoxon_enumeration_check_p": exact_wilcoxon_p_one_sided(arr),
        "sign_test_exact_one_sided_p": exact_sign_test_p_one_sided(arr),
        "test": "Wilcoxon signed-rank",
        "alternative": "greater",
        "method": "exact",
        "n": len(arr),
        "engine": engine,
    }


def stable_rmse_by_method(method: str, seeds, d6_root: Path, d7_root: Path) -> dict:
    from scripts.d8_formal_metrics import collect_stable_rmse

    if method == "s1":
        root, stage, tag = d7_root, "d7_stage_b", "d7_s1_e40"
    else:
        root, stage, tag = d6_root, "d6_stage_b", f"d6_{method}_e40"
    rows = collect_stable_rmse(Path(root) / stage, method, tag, seeds)
    return {int(row["seed"]): float(row["stable_rmse"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6_root", default="artifacts/runs/d6")
    parser.add_argument("--d7_root", default="artifacts/runs/d7")
    parser.add_argument("--output_dir", default="artifacts/results/d8_result_correction/metrics")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--scipy_version", default=None, help="override recorded engine note")
    args = parser.parse_args()

    b1 = stable_rmse_by_method("b1", args.seeds, Path(args.d6_root), Path(args.d7_root))
    s1 = stable_rmse_by_method("s1", args.seeds, Path(args.d6_root), Path(args.d7_root))
    common = sorted(set(b1) & set(s1))
    if len(common) != len(args.seeds):
        raise SystemExit(
            f"stable RMSE incomplete: b1={sorted(b1)} s1={sorted(s1)} seeds={args.seeds}"
        )
    gains = [b1[seed] - s1[seed] for seed in common]

    try:
        import scipy

        engine_note = f"scipy {scipy.__version__}, method='exact', alternative='greater'"
    except ImportError:
        engine_note = "scipy unavailable"
    if args.scipy_version:
        engine_note = args.scipy_version

    stats_payload = compute_paired_stats(gains, common, engine_note)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "D8_S1_VS_B1_PAIRED_STATS.json"
    output_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_path}")
    print(
        f"S1_vs_B1: mean={stats_payload['mean_gain']:.4f} "
        f"W/T/L={stats_payload['wins']}/{stats_payload['ties']}/{stats_payload['losses']} "
        f"wilcoxon_exact_p={stats_payload['wilcoxon_exact_one_sided_p']:.5f} "
        f"sign_test_p={stats_payload['sign_test_exact_one_sided_p']:.5f} "
        f"engine={stats_payload['engine']}"
    )


if __name__ == "__main__":
    main()
