"""D8 result-correction Task A: recompute endpoint metrics from predictions.

Reads each formal run's saved validation predictions
(diagnostics/best_validation_human3_predictions.csv), maps schema aliases
(label/target -> y_true, final_prediction/prediction -> y_pred) and computes
RMSE / MAE / Pearson / Spearman / R2 DIRECTLY from y_true and y_pred via
scipy.stats / sklearn.  The former sqrt(R2) shortcut and the R2<0 -> 0 clamp
are retired: degenerate correlations are reported as NaN with an explicit
status instead of a fabricated 0.

Cross-check (plan §9): every recomputed RMSE must match the existing
diagnostics/human3_path_metrics.csv row of the SAME best epoch
(path == final) within 1e-6, otherwise the script exits STOP_TASK_A and the
paper metrics must not be replaced.

Outputs (under --output_dir):
    D8_RECOMPUTED_ENDPOINT_METRICS.csv   method,seed,endpoint,n,rmse,mae,
                                         pearson_r,pearson_p,spearman_rho,
                                         spearman_p,r2,status,prediction_source
    D8_RECOMPUTED_MACRO_METRICS.csv      per method,seed macro means
                                         (NaN-ignoring, with valid counts)
    D8_METRIC_RECOMPUTE_AUDIT.json       num_runs/rows, degenerate counts,
                                         max RMSE difference, all_rmse_match
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

import numpy as np

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
METHOD_LABELS = {"b0": "B0", "b1": "B1", "s1": "S1"}
PREDICTION_FILENAME = "best_validation_human3_predictions.csv"
RMSE_TOLERANCE = 1e-6

Y_TRUE_ALIASES = ("y_true", "label", "target")
Y_PRED_ALIASES = ("y_pred", "final_prediction", "prediction")
ENDPOINT_ALIASES = ("endpoint", "task")
ID_ALIASES = ("sample_id", "id")


def _first_column(row: dict, aliases) -> str | None:
    for name in aliases:
        if name in row and row[name] != "":
            return name
    return None


def normalize_prediction_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Map raw prediction rows onto {endpoint, sample_id, y_true, y_pred}.

    Returns the normalized rows plus the actual column names used, so the
    audit can record the prediction source schema (plan §3-§4).
    """
    if not rows:
        raise ValueError("prediction file is empty")
    mapping = {
        "y_true": _first_column(rows[0], Y_TRUE_ALIASES),
        "y_pred": _first_column(rows[0], Y_PRED_ALIASES),
        "endpoint": _first_column(rows[0], ENDPOINT_ALIASES),
        "sample_id": _first_column(rows[0], ID_ALIASES),
    }
    missing = [key for key, value in mapping.items() if value is None]
    if missing:
        raise ValueError(f"prediction schema missing columns for {missing}")
    normalized = []
    for row in rows:
        normalized.append(
            {
                "endpoint": str(row[mapping["endpoint"]]),
                "sample_id": str(row[mapping["sample_id"]]),
                "y_true": float(row[mapping["y_true"]]),
                "y_pred": float(row[mapping["y_pred"]]),
            }
        )
    return normalized, mapping


def endpoint_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    """Direct RMSE/MAE/Pearson/Spearman/R2 with plan §6 edge cases."""
    y = np.asarray(y_true, dtype=np.float64)
    yhat = np.asarray(y_pred, dtype=np.float64)
    n = int(y.size)
    result = {
        "n": n,
        "rmse": float("nan"),
        "mae": float("nan"),
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "r2": float("nan"),
        "status": "OK",
    }
    if n == 0:
        result["status"] = "INSUFFICIENT_N"
        return result
    result["rmse"] = float(np.sqrt(np.mean((yhat - y) ** 2)))
    result["mae"] = float(np.mean(np.abs(yhat - y)))
    result["r2"] = float(1.0 - np.sum((yhat - y) ** 2) / np.sum((y - y.mean()) ** 2))
    if n < 3:
        result["status"] = "INSUFFICIENT_N"
        return result
    if np.all(y == y[0]) or np.all(yhat == yhat[0]):
        result["status"] = "CONSTANT_INPUT"
        return result
    from scipy import stats

    pearson = stats.pearsonr(y, yhat)
    spearman = stats.spearmanr(y, yhat)
    result["pearson_r"] = float(pearson.statistic)
    result["pearson_p"] = float(pearson.pvalue)
    result["spearman_rho"] = float(spearman.statistic)
    result["spearman_p"] = float(spearman.pvalue)
    return result


def macro_metrics(endpoint_rows: list[dict]) -> dict:
    """Mean over endpoints, ignoring NaN, with valid-count columns (§8)."""

    def _mean(field: str) -> tuple[float, int]:
        values = [row[field] for row in endpoint_rows if math.isfinite(row[field])]
        return (sum(values) / len(values) if values else float("nan"), len(values))

    macro_rmse, _ = _mean("rmse")
    macro_mae, _ = _mean("mae")
    macro_pearson, pearson_valid = _mean("pearson_r")
    macro_spearman, spearman_valid = _mean("spearman_rho")
    macro_r2, _ = _mean("r2")
    return {
        "macro_rmse": macro_rmse,
        "macro_mae": macro_mae,
        "macro_pearson": macro_pearson,
        "macro_spearman": macro_spearman,
        "macro_r2": macro_r2,
        "pearson_valid_tasks": pearson_valid,
        "spearman_valid_tasks": spearman_valid,
    }


def best_epoch_from_run(run_dir: Path) -> int | None:
    """Best epoch recorded in the run's *_best.pt checkpoint payload."""
    import torch

    best_pts = sorted(run_dir.glob("*_best.pt"))
    if len(best_pts) != 1:
        return None
    payload = torch.load(best_pts[0], map_location="cpu", weights_only=False)
    epoch = payload.get("epoch")
    return int(epoch) if epoch is not None else None


def fallback_best_epoch(run_dir: Path) -> int | None:
    """Argmin of validation human3 macro RMSE (selection_scope human3)."""
    path = run_dir / "diagnostics" / "epoch_summary.csv"
    if not path.is_file():
        return None
    rows = [
        (int(r["epoch"]), float(r["val_human3_macro_rmse"]))
        for r in csv.DictReader(path.open(encoding="utf-8"))
        if r.get("val_human3_macro_rmse") not in (None, "", "nan")
    ]
    if not rows:
        return None
    return min(rows, key=lambda pair: pair[1])[0]


def existing_rmse_by_endpoint(run_dir: Path, epoch: int) -> dict[str, float]:
    """path == final RMSE per endpoint at the given epoch."""
    path = run_dir / "diagnostics" / "human3_path_metrics.csv"
    result: dict[str, float] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if int(row["epoch"]) == int(epoch) and row.get("path") == "final":
            result[row["task"]] = float(row["rmse"])
    return result


def run_dir_for(method: str, seed: int, d6_root: Path, d7_root: Path) -> Path:
    if method == "s1":
        return d7_root / "d7_stage_b" / "s1" / "d7_s1_e40" / f"seed_{seed}"
    return d6_root / "d6_stage_b" / method / f"d6_{method}_e40" / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d6_root", default="artifacts/runs/d6")
    parser.add_argument("--d7_root", default="artifacts/runs/d7")
    parser.add_argument("--output_dir", default="artifacts/results/d8_result_correction/metrics")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--rmse_tolerance", type=float, default=RMSE_TOLERANCE,
        help="max |recomputed - existing| RMSE before STOP_TASK_A",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_rows: list[dict] = []
    macro_rows: list[dict] = []
    num_runs = 0
    num_constant = 0
    num_insufficient = 0
    max_rmse_diff = 0.0
    all_rmse_match = True

    for method in ("b0", "b1", "s1"):
        for seed in args.seeds:
            run_dir = run_dir_for(method, seed, Path(args.d6_root), Path(args.d7_root))
            pred_path = run_dir / "diagnostics" / PREDICTION_FILENAME
            if not pred_path.is_file():
                raise FileNotFoundError(
                    f"missing predictions for {method} seed {seed}: {pred_path} "
                    "(plan §62: STOP, do not retrain)"
                )
            with pred_path.open(newline="", encoding="utf-8") as handle:
                raw = list(csv.DictReader(handle))
            predictions, mapping = normalize_prediction_rows(raw)
            best_epoch = best_epoch_from_run(run_dir)
            epoch_source = "best_checkpoint"
            if best_epoch is None:
                best_epoch = fallback_best_epoch(run_dir)
                epoch_source = "argmin_epoch_summary"
            if best_epoch is None:
                raise RuntimeError(f"cannot determine best epoch for {run_dir}")
            existing = existing_rmse_by_endpoint(run_dir, best_epoch)

            per_endpoint = []
            for task in HUMAN_TASKS:
                pairs = [p for p in predictions if p["endpoint"] == task]
                metrics = endpoint_metrics([p["y_true"] for p in pairs], [p["y_pred"] for p in pairs])
                if metrics["status"] == "CONSTANT_INPUT":
                    num_constant += 1
                if metrics["status"] == "INSUFFICIENT_N":
                    num_insufficient += 1
                recomputed_rmse = metrics["rmse"]
                if task in existing and math.isfinite(recomputed_rmse):
                    diff = abs(recomputed_rmse - existing[task])
                    max_rmse_diff = max(max_rmse_diff, diff)
                    if diff > args.rmse_tolerance:
                        all_rmse_match = False
                        print(
                            f"RMSE MISMATCH {method} s{seed} {task}: "
                            f"recomputed={recomputed_rmse:.10f} "
                            f"existing={existing[task]:.10f} diff={diff:.3e}"
                        )
                endpoint_rows.append(
                    {
                        "method": METHOD_LABELS[method],
                        "seed": seed,
                        "endpoint": task,
                        "n": metrics["n"],
                        "rmse": f"{metrics['rmse']:.10g}",
                        "mae": f"{metrics['mae']:.10g}",
                        "pearson_r": f"{metrics['pearson_r']:.10g}",
                        "pearson_p": f"{metrics['pearson_p']:.10g}",
                        "spearman_rho": f"{metrics['spearman_rho']:.10g}",
                        "spearman_p": f"{metrics['spearman_p']:.10g}",
                        "r2": f"{metrics['r2']:.10g}",
                        "status": metrics["status"],
                        "prediction_source": f"{pred_path} (epoch={best_epoch}, via {epoch_source})",
                    }
                )
                per_endpoint.append(metrics)
            macro = macro_metrics(per_endpoint)
            macro_rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "seed": seed,
                    **{key: f"{value:.10g}" for key, value in macro.items()},
                }
            )
            num_runs += 1

    def _write(path: Path, fieldnames, rows_list):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_list)
        print(f"wrote {path}")

    _write(
        output_dir / "D8_RECOMPUTED_ENDPOINT_METRICS.csv",
        ["method", "seed", "endpoint", "n", "rmse", "mae", "pearson_r", "pearson_p",
         "spearman_rho", "spearman_p", "r2", "status", "prediction_source"],
        endpoint_rows,
    )
    _write(
        output_dir / "D8_RECOMPUTED_MACRO_METRICS.csv",
        ["method", "seed", "macro_rmse", "macro_mae", "macro_pearson",
         "macro_spearman", "macro_r2", "pearson_valid_tasks", "spearman_valid_tasks"],
        macro_rows,
    )
    audit = {
        "num_runs": num_runs,
        "num_endpoint_rows": len(endpoint_rows),
        "num_constant_input": num_constant,
        "num_insufficient_n": num_insufficient,
        "max_rmse_difference_vs_existing": max_rmse_diff,
        "rmse_tolerance": args.rmse_tolerance,
        "all_rmse_match": all_rmse_match,
        "y_true_column": "label",
        "y_pred_column": "final_prediction",
        "test_accessed": False,
    }
    (output_dir / "D8_METRIC_RECOMPUTE_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {output_dir / 'D8_METRIC_RECOMPUTE_AUDIT.json'}")
    if not all_rmse_match:
        print("STOP_TASK_A: recomputed RMSE does not match the existing metrics")
        raise SystemExit(2)
    print(f"TASK_A_OK runs={num_runs} max_rmse_diff={max_rmse_diff:.3e}")


if __name__ == "__main__":
    main()
