"""D5 Stage-1: transfer-signal feasibility probe + validation-noise audit.

Plan D5 sections 12-53.  Two phases:

- ``collect``: forward each HPS seed's best checkpoint over the human3
  **train + validation** loaders only, extracting the 56 animal-endpoint
  decoded median predictions (source features), the target base prediction,
  labels, and scaffold groups.  Calibration/test are never touched.
- ``analyze``: writes the D5_0 selection-noise CSVs (from existing epoch
  trajectories and best-validation predictions) and the D5_1 Ridge probe
  CSVs (P1 source-only, P2 base+residual; scaffold GroupKFold alpha
  selection inside train; 20 sample-row shuffle negative controls).
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
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS, parse_toxacute_task_name
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
from reproducibility import seed_everything
from trainer import Trainer
import weighting as weighting_method

HUMAN_TASKS = list(HUMAN_TARGET_TASKS)
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
SHUFFLE_SEEDS = list(range(7001, 7021))
BOOTSTRAP_ROUNDS = 10000

HPS_RUN_DIRS = {
    42: "artifacts/runs/formal_hps/formal/seed_42",
    43: "artifacts/runs/formal_hps/formal/seed_43",
    44: "artifacts/runs/formal_hps/formal/seed_44",
    45: "artifacts/runs/d4_hps_100e/d4_hps_e100/seed_45",
    46: "artifacts/runs/d4_hps_100e/d4_hps_e100/seed_46",
}
TASK_ONLY_DIRS = {
    # (epochs 0-39 era, epochs 40-99 era or fresh)
    42: (
        "artifacts/runs/d3_mechanisms/task_only_router/d3_task_only_router_e40/seed_42",
        "artifacts/runs/d4_task_only_100e/d4_task_only_e100/seed_42",
    ),
    43: (
        "artifacts/runs/d3_mechanisms/task_only_router/d3_task_only_router_e40/seed_43",
        "artifacts/runs/d4_task_only_100e/d4_task_only_e100/seed_43",
    ),
    44: (
        "artifacts/runs/d3_mechanisms/task_only_router/d3_task_only_router_e40/seed_44",
        "artifacts/runs/d4_task_only_100e/d4_task_only_e100/seed_44",
    ),
    45: (None, "artifacts/runs/d4_task_only_100e/d4_task_only_e100/seed_45"),
    46: (None, "artifacts/runs/d4_task_only_100e/d4_task_only_e100/seed_46"),
}

SEEDS = [42, 43, 44, 45, 46]


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


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - target)))


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size < 2 or np.unique(target).size < 2:
        return float("nan")
    ss_total = float(np.sum((target - target.mean()) ** 2))
    if ss_total <= 0:
        return float("nan")
    return float(1.0 - np.sum((pred - target) ** 2) / ss_total)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return float("nan")
    left_rank = np.argsort(np.argsort(left)).astype(float)
    right_rank = np.argsort(np.argsort(right)).astype(float)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = math.sqrt(float((left_centered**2).sum()) * float((right_centered**2).sum()))
    if denominator == 0:
        return float("nan")
    return float((left_centered * right_centered).sum() / denominator)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(float((left_centered**2).sum()) * float((right_centered**2).sum()))
    if denominator == 0:
        return float("nan")
    return float((left_centered * right_centered).sum() / denominator)


# ======================================================================
# collect phase
# ======================================================================
def _load_trainer(run_dir: Path, gpu_id):
    params = build_parser().parse_args([])
    for key, value in json.loads((run_dir / "args.json").read_text(encoding="utf-8")).items():
        setattr(params, key, value)
    if gpu_id is not None:
        params.gpu_id = gpu_id
    validate_params(params)

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
    checkpoint_path = run_dir / f"{params.ckpt_name}_best.pt"
    try:
        trainer.load_checkpoint(checkpoint_path)
    except Exception as exc:  # noqa: BLE001 - formal-era checkpoints may miss v6 blocks
        print(f"strict load failed ({exc}); falling back to raw state dict")
        payload = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
        state = payload.get("model_state", payload.get("model_state_dict"))
        trainer.model.load_state_dict(state, strict=True)
        scalers = payload.get("task_scalers")
        if scalers:
            trainer.task_scalers = dict(scalers)
    trainer.model.eval()
    return trainer, params, collator, task_names


def best_epoch_for(run_dir: Path, params) -> int:
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "diagnostics" / "epoch_summary.csv"
    if metrics_path.exists():
        history = json.loads(metrics_path.read_text(encoding="utf-8")).get("history", [])
        validation_epochs = [entry for entry in history if "validation" in entry]
        if validation_epochs:
            best = min(
                validation_epochs,
                key=lambda entry: entry["validation"].get("human3_macro_rmse", float("inf")),
            )
            return int(best["epoch"])
    if summary_path.exists():
        rows = [
            row
            for row in _read_csv(summary_path)
            if row.get("is_best") == "True"
        ]
        if rows:
            return max(int(row["epoch"]) for row in rows)
    raise SystemExit(f"cannot determine best epoch under {run_dir}")


def collect_seed(seed: int, run_dir: Path, gpu_id, output_dir: Path) -> Path:
    params = build_parser().parse_args([])
    for key, value in json.loads((run_dir / "args.json").read_text(encoding="utf-8")).items():
        setattr(params, key, value)
    if gpu_id is not None:
        params.gpu_id = gpu_id
    validate_params(params)
    best_epoch = best_epoch_for(run_dir, params)

    trainer, params, collator, task_names = _load_trainer(run_dir, gpu_id)
    store = _resolve_data_store(params, task_names, required=True)
    trainer.sample_row_index = {
        str(sample_id): int(row_index)
        for sample_id, row_index in zip(store.sample_ids.tolist(), store.row_indices.tolist())
    }
    scaffold_by_sample = {
        str(record.get("sample_id")): str(record.get("scaffold", ""))
        for record in json.loads((Path(store.root) / "split_manifest.json").read_text(encoding="utf-8")).get("records", [])
    }
    loaders = _loaders(params, task_names, collator)
    animal_tasks = [task for task in task_names if task not in set(HUMAN_TASKS)]

    trainer.model.eval()
    fields = ["split", "task", "sample_id", "row_index", "scaffold", "label", "base_pred"] + [
        f"src__{task}" for task in animal_tasks
    ]
    out_path = output_dir / f"hps_features_seed{seed}.csv"
    rows = []
    with torch.no_grad():
        for task in HUMAN_TASKS:
            for split in ("train", "val"):
                loader = loaders[split].get(task)
                if loader is None:
                    continue
                for batch in loader:
                    if getattr(batch, "get", lambda *_: None)("is_empty", False):
                        continue
                    batch = batch.to(trainer.device)
                    predictions = trainer.model(batch, return_all_tasks=True)
                    decoded = {
                        name: trainer.decode_task_output(name, predictions[name], apply_conformal=False)[
                            "median"
                        ]
                        .cpu()
                        .reshape(-1)
                        for name in task_names
                    }
                    labels = batch.y.reshape(-1).float().cpu()
                    sample_ids = [str(value) for value in (getattr(batch, "sample_id", None) or [])]
                    for index in range(labels.numel()):
                        sample_id = sample_ids[index] if index < len(sample_ids) else ""
                        row = {
                            "split": split,
                            "task": task,
                            "sample_id": sample_id,
                            "row_index": trainer.sample_row_index.get(sample_id, ""),
                            "scaffold": scaffold_by_sample.get(sample_id, ""),
                            "label": float(labels[index]),
                            "base_pred": float(decoded[task][index]),
                        }
                        for source in animal_tasks:
                            row[f"src__{source}"] = float(decoded[source][index])
                        rows.append(row)
    _write_csv(out_path, fields, rows)
    print(f"collected {len(rows)} rows for seed {seed} (best epoch {best_epoch})")
    return out_path


# ======================================================================
# trajectory loading for D5-0
# ======================================================================
def hps_trajectory(seed: int) -> list[dict]:
    run_dir = Path(HPS_RUN_DIRS[seed])
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        history = json.loads(metrics_path.read_text(encoding="utf-8")).get("history", [])
        rows = []
        for entry in history:
            validation = entry.get("validation")
            if not validation:
                continue
            train = entry.get("train", {})
            rows.append(
                {
                    "epoch": int(entry["epoch"]),
                    "val": float(validation.get("human3_macro_rmse", float("nan"))),
                    "train": float(train.get("human3_macro_rmse", float("nan"))),
                }
            )
        return sorted(rows, key=lambda row: row["epoch"])
    summary = _read_csv(run_dir / "diagnostics" / "epoch_summary.csv")
    return sorted(
        (
            {
                "epoch": int(row["epoch"]),
                "val": float(row["val_human3_macro_rmse"]),
                "train": float(row.get("train_human3_macro_rmse", float("nan"))),
            }
            for row in summary
            if row.get("val_human3_macro_rmse") not in (None, "", "nan")
        ),
        key=lambda row: row["epoch"],
    )


def task_only_trajectory(seed: int) -> list[dict]:
    early, late = TASK_ONLY_DIRS[seed]
    rows: list[dict] = []
    for era in (early, late):
        if not era:
            continue
        for row in _read_csv(Path(era) / "diagnostics" / "epoch_summary.csv"):
            value = row.get("val_human3_macro_rmse")
            if value in (None, "", "nan"):
                continue
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "val": float(value),
                    "train": float(row.get("train_human3_macro_rmse", float("nan"))),
                }
            )
    return sorted(
        sorted(rows, key=lambda row: row["epoch"]),
        key=lambda row: row["epoch"],
    )


def trajectory_metrics(rows: list[dict]) -> dict:
    values = np.array([row["val"] for row in rows], dtype=float)
    trains = np.array([row["train"] for row in rows], dtype=float)
    best_index = int(np.nanargmin(values))
    best = float(values[best_index])
    within = lambda tolerance: int(np.sum(values <= best * (1.0 + tolerance)))
    local = values[max(0, best_index - 2) : best_index + 3]
    deltas = np.diff(values)
    if deltas.size >= 2:
        centered = deltas - deltas.mean()
        denom = float((centered**2).sum())
        lag1 = (
            float((centered[:-1] * centered[1:]).sum() / denom) if denom else float("nan")
        )
    else:
        lag1 = float("nan")
    return {
        "best_epoch": rows[best_index]["epoch"],
        "best_rmse": best,
        "median_rmse": float(np.nanmedian(values)),
        "mean_rmse": float(np.nanmean(values)),
        "best_minus_median": float(np.nanmedian(values) - best),
        "local5_mean": float(np.nanmean(local)),
        "best_sharpness": float(np.nanmean(local) - best),
        "epochs_within_1pct": within(0.01),
        "epochs_within_2pct": within(0.02),
        "epochs_within_5pct": within(0.05),
        "volatility_std": float(np.nanstd(deltas)),
        "volatility_median_abs_delta": float(np.nanmedian(np.abs(deltas))),
        "volatility_p95_abs_delta": float(np.nanpercentile(np.abs(deltas), 95)),
        "autocorr_lag1": lag1,
        "train_at_best": float(trains[best_index]),
        "generalization_gap": float(values[best_index] - trains[best_index]),
        "n_epochs": int(values.size),
    }


def endpoint_trajectory(model: str, seed: int, task: str) -> list[dict]:
    rows: list[dict] = []
    if model == "hps":
        run_dir = Path(HPS_RUN_DIRS[seed])
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            history = json.loads(metrics_path.read_text(encoding="utf-8")).get("history", [])
            for entry in history:
                validation = entry.get("validation")
                if not validation:
                    continue
                value = validation.get("tasks", {}).get(task, {}).get("RMSE", float("nan"))
                rows.append({"epoch": int(entry["epoch"]), "val": float(value)})
        return sorted(rows, key=lambda row: row["epoch"])
    early, late = TASK_ONLY_DIRS[seed]
    for era in (early, late):
        if not era:
            continue
        for row in _read_csv(Path(era) / "diagnostics" / "human3_path_metrics.csv"):
            if row["task"] == task and row["path"] == "final":
                rows.append({"epoch": int(row["epoch"]), "val": float(row["rmse"])})
    return sorted(rows, key=lambda row: row["epoch"])


# ======================================================================
# Task-only per-sample validation predictions
# ======================================================================
def task_only_best_predictions(seed: int) -> list[dict]:
    early, late = TASK_ONLY_DIRS[seed]
    for era in (late, early):
        if not era:
            continue
        path = Path(era) / "diagnostics" / "best_validation_human3_predictions.csv"
        rows = _read_csv(path)
        if rows:
            return rows
    return []


def hps_val_predictions(features_path: Path) -> list[dict]:
    rows = []
    for row in _read_csv(features_path):
        if row["split"] != "val":
            continue
        rows.append(
            {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "row_index": row["row_index"],
                "label": float(row["label"]),
                "final_prediction": float(row["base_pred"]),
            }
        )
    return rows


# ======================================================================
# D5-0 analysis
# ======================================================================
def run_d5_0(features_by_seed: dict[int, Path], output_dir: Path):
    selection_rows = []
    for model, trajectory_fn in (("hps", hps_trajectory), ("task_only", task_only_trajectory)):
        for seed in SEEDS:
            rows = trajectory_fn(seed)
            if not rows:
                selection_rows.append({"model": model, "seed": seed, "status": "MISSING_TRAJECTORY"})
                continue
            row = {"model": model, "seed": seed, "status": "OK"}
            row.update(trajectory_metrics(rows))
            selection_rows.append(row)
    _write_csv(
        output_dir / "D5_0_SELECTION_NOISE_SUMMARY.csv",
        ["model", "seed", "status", "n_epochs", "best_epoch", "best_rmse", "median_rmse", "mean_rmse",
         "best_minus_median", "local5_mean", "best_sharpness", "epochs_within_1pct", "epochs_within_2pct",
         "epochs_within_5pct", "volatility_std", "volatility_median_abs_delta", "volatility_p95_abs_delta",
         "autocorr_lag1", "train_at_best", "generalization_gap"],
        selection_rows,
    )

    volatility_rows = []
    for model in ("hps", "task_only"):
        for seed in SEEDS:
            for task in HUMAN_TASKS:
                trajectory = endpoint_trajectory(model, seed, task)
                if not trajectory:
                    continue
                values = np.array([row["val"] for row in trajectory], dtype=float)
                best_index = int(np.nanargmin(values))
                volatility_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "task": task,
                        "best_epoch": trajectory[best_index]["epoch"],
                        "best_rmse": float(values[best_index]),
                        "median_rmse": float(np.nanmedian(values)),
                        "trajectory_std": float(np.nanstd(values)),
                    }
                )
    _write_csv(
        output_dir / "D5_0_ENDPOINT_VOLATILITY.csv",
        ["model", "seed", "task", "best_epoch", "best_rmse", "median_rmse", "trajectory_std"],
        volatility_rows,
    )

    # per-sample validation predictions for the remaining D5-0 CSVs
    predictions: dict[tuple[str, int], dict[tuple[str, str], dict]] = {}
    for seed in SEEDS:
        hps_rows = hps_val_predictions(features_by_seed[seed])
        predictions[("hps", seed)] = {
            (row["task"], row["sample_id"]): row for row in hps_rows
        }
        task_rows = task_only_best_predictions(seed)
        predictions[("task_only", seed)] = {
            (row["task"], row["sample_id"]): row for row in task_rows
        }

    influence_rows = []
    for (model, seed), table in predictions.items():
        for task in HUMAN_TASKS:
            entries = [row for key, row in table.items() if key[0] == task]
            if len(entries) < 3:
                continue
            preds = np.array([row["final_prediction"] for row in entries])
            labels = np.array([float(row["label"]) for row in entries])
            full = _rmse(preds, labels)
            for index, row in enumerate(entries):
                keep = np.ones(preds.size, dtype=bool)
                keep[index] = False
                loo = _rmse(preds[keep], labels[keep])
                influence_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "task": task,
                        "sample_id": row["sample_id"],
                        "row_index": row.get("row_index", ""),
                        "full_rmse": full,
                        "leave_one_out_rmse": loo,
                        "influence": full - loo,
                    }
                )
    _write_csv(
        output_dir / "D5_0_SAMPLE_INFLUENCE.csv",
        ["model", "seed", "task", "sample_id", "row_index", "full_rmse", "leave_one_out_rmse", "influence"],
        influence_rows,
    )

    bootstrap_rows = []
    generator = np.random.default_rng(20260831)
    for seed in SEEDS:
        hps_table = predictions[("hps", seed)]
        task_table = predictions[("task_only", seed)]
        shared = sorted(set(hps_table) & set(task_table))
        if not shared:
            continue
        labels = np.array([hps_table[key]["label"] for key in shared])
        hps_err = np.array(
            [hps_table[key]["final_prediction"] - hps_table[key]["label"] for key in shared]
        )
        task_err = np.array(
            [task_table[key]["final_prediction"] - task_table[key]["label"] for key in shared]
        )
        for scope, mask in (
            *[(task, np.array([key[0] == task for key in shared])) for task in HUMAN_TASKS],
            ("__human3_macro__", np.ones(len(shared), dtype=bool)),
        ):
            if mask.sum() < 3:
                continue
            mse_hps = float(np.mean(hps_err[mask] ** 2))
            mse_task = float(np.mean(task_err[mask] ** 2))
            observed = math.sqrt(mse_hps) - math.sqrt(mse_task)
            deltas = np.empty(BOOTSTRAP_ROUNDS)
            n = int(mask.sum())
            for round_index in range(BOOTSTRAP_ROUNDS):
                sample_index = generator.integers(0, n, size=n)
                delta = math.sqrt(np.mean(hps_err[mask][sample_index] ** 2)) - math.sqrt(
                    np.mean(task_err[mask][sample_index] ** 2)
                )
                deltas[round_index] = delta
            bootstrap_rows.append(
                {
                    "seed": seed,
                    "scope": scope,
                    "n": n,
                    "observed_delta_rmse": observed,
                    "bootstrap_mean": float(deltas.mean()),
                    "bootstrap_p2_5": float(np.percentile(deltas, 2.5)),
                    "bootstrap_p97_5": float(np.percentile(deltas, 97.5)),
                    "fraction_positive": float((deltas > 0).mean()),
                }
            )
    _write_csv(
        output_dir / "D5_0_BOOTSTRAP_DIAGNOSTIC.csv",
        ["seed", "scope", "n", "observed_delta_rmse", "bootstrap_mean", "bootstrap_p2_5",
         "bootstrap_p97_5", "fraction_positive"],
        bootstrap_rows,
    )

    disagreement_rows = []
    ensemble_rows = []
    for model in ("hps", "task_only"):
        per_seed_tables = [predictions[(model, seed)] for seed in SEEDS]
        for task in HUMAN_TASKS:
            keys = sorted(
                {key for table in per_seed_tables for key in table if key[0] == task}
            )
            if not keys:
                continue
            stacked = np.array(
                [[table.get(key, {}).get("final_prediction", float("nan")) for table in per_seed_tables] for key in keys]
            )
            labels = np.array([per_seed_tables[0][key]["label"] for key in keys])
            with np.errstate(invalid="ignore"):
                seed_std = np.nanstd(stacked, axis=0, ddof=1)
            disagreement_rows.append(
                {
                    "model": model,
                    "task": task,
                    "n_samples": len(keys),
                    "mean_seed_std": float(np.nanmean(seed_std)),
                    "median_seed_std": float(np.nanmedian(seed_std)),
                    "p95_seed_std": float(np.nanpercentile(seed_std, 95)),
                }
            )
            ensemble = np.nanmean(stacked, axis=0)
            ensemble_rows.append(
                {
                    "model": model,
                    "task": task,
                    "n_samples": len(keys),
                    "rmse": _rmse(ensemble, labels),
                    "mae": _mae(ensemble, labels),
                    "r2": _r2(ensemble, labels),
                }
            )
    _write_csv(
        output_dir / "D5_0_SEED_DISAGREEMENT.csv",
        ["model", "task", "n_samples", "mean_seed_std", "median_seed_std", "p95_seed_std"],
        disagreement_rows,
    )
    _write_csv(
        output_dir / "D5_0_ENSEMBLE_DIAGNOSTIC.csv",
        ["model", "task", "n_samples", "rmse", "mae", "r2"],
        ensemble_rows,
    )


# ======================================================================
# D5-1A analysis
# ======================================================================
def _group_cv_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, list[dict]]:
    n_groups = len(set(groups.tolist()))
    n_splits = 5 if n_groups >= 5 else (3 if n_groups >= 3 else 2)
    splitter = GroupKFold(n_splits=n_splits)
    cv_rows = []
    mean_rmse = {}
    for alpha in ALPHA_GRID:
        fold_rmse = []
        for train_index, valid_index in splitter.split(x, y, groups=groups):
            if valid_index.size == 0 or train_index.size == 0:
                continue
            model = Ridge(alpha=alpha)
            model.fit(x[train_index], y[train_index])
            prediction = model.predict(x[valid_index])
            fold_rmse.append(_rmse(prediction, y[valid_index]))
        if not fold_rmse:
            continue
        mean_rmse[alpha] = float(np.mean(fold_rmse))
        cv_rows.append({"alpha": alpha, "mean_cv_rmse": mean_rmse[alpha], "n_folds": len(fold_rmse)})
    if not mean_rmse:
        return 1.0, cv_rows
    selected = min(mean_rmse, key=mean_rmse.get)
    return selected, cv_rows


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return model


def run_d5_1(features_by_seed: dict[int, Path], output_dir: Path):
    summary_rows = []
    per_sample_rows = []
    cv_rows = []
    coefficient_rows = []
    correlation_rows = []
    shuffle_rows = []
    macro_by_seed = {}

    feature_tables = {seed: _read_csv(path) for seed, path in features_by_seed.items()}
    source_columns = [
        column
        for column in feature_tables[SEEDS[0]][0].keys()
        if column.startswith("src__")
    ]

    for seed in SEEDS:
        table = feature_tables[seed]
        macro = {"base": [], "p1": [], "p2": []}
        for target in HUMAN_TASKS:
            train_rows = [
                row for row in table if row["split"] == "train" and row["task"] == target
            ]
            val_rows = [row for row in table if row["split"] == "val" and row["task"] == target]
            if not train_rows or not val_rows:
                summary_rows.append(
                    {"seed": seed, "target_task": target, "status": "INSUFFICIENT_ROWS"}
                )
                continue

            y_train = np.array([float(row["label"]) for row in train_rows])
            base_train = np.array([float(row["base_pred"]) for row in train_rows])
            x_train_raw = np.array(
                [[float(row[column]) for column in source_columns] for row in train_rows]
            )
            groups = np.array([row["scaffold"] for row in train_rows])
            y_val = np.array([float(row["label"]) for row in val_rows])
            base_val = np.array([float(row["base_pred"]) for row in val_rows])
            x_val_raw = np.array(
                [[float(row[column]) for column in source_columns] for row in val_rows]
            )

            train_std = x_train_raw.std(axis=0)
            keep_columns = train_std >= 1e-8
            dropped = int((~keep_columns).sum())
            x_train = (x_train_raw[:, keep_columns] - x_train_raw[:, keep_columns].mean(axis=0)) / train_std[
                keep_columns
            ]
            x_val = (x_val_raw[:, keep_columns] - x_train_raw[:, keep_columns].mean(axis=0)) / train_std[
                keep_columns
            ]

            alpha_p1, cv_p1 = _group_cv_alpha(x_train, y_train, groups)
            model_p1 = _fit_ridge(x_train, y_train, alpha_p1)
            p1_val = model_p1.predict(x_val)

            residual_train = y_train - base_train
            alpha_p2, cv_p2 = _group_cv_alpha(x_train, residual_train, groups)
            model_p2 = _fit_ridge(x_train, residual_train, alpha_p2)
            p2_val = base_val + model_p2.predict(x_val)

            base_rmse = _rmse(base_val, y_val)
            p1_rmse = _rmse(p1_val, y_val)
            p2_rmse = _rmse(p2_val, y_val)

            # shuffle negative controls: break sample<->source alignment in train
            shuffle_gains = []
            for shuffle_seed in SHUFFLE_SEEDS:
                generator = np.random.default_rng(shuffle_seed)
                order = generator.permutation(y_train.size)
                shuffled_residual = residual_train[order]
                alpha_shuffle, _ = _group_cv_alpha(x_train, shuffled_residual, groups)
                shuffle_model = _fit_ridge(x_train, shuffled_residual, alpha_shuffle)
                shuffle_prediction = base_val + shuffle_model.predict(x_val)
                shuffle_gains.append(base_rmse - _rmse(shuffle_prediction, y_val))

            for row_index, val_row in enumerate(val_rows):
                per_sample_rows.append(
                    {
                        "seed": seed,
                        "sample_id": val_row["sample_id"],
                        "row_index": val_row["row_index"],
                        "task": target,
                        "label": float(val_row["label"]),
                        "base_prediction": float(base_val[row_index]),
                        "source_only_prediction": float(p1_val[row_index]),
                        "base_plus_source_prediction": float(p2_val[row_index]),
                        "base_error": abs(float(base_val[row_index]) - float(val_row["label"])),
                        "source_only_error": abs(float(p1_val[row_index]) - float(val_row["label"])),
                        "base_plus_source_error": abs(float(p2_val[row_index]) - float(val_row["label"])),
                    }
                )

            base_r2 = _r2(base_val, y_val)
            summary_rows.append(
                {
                    "seed": seed,
                    "target_task": target,
                    "n_train": len(train_rows),
                    "n_val": len(val_rows),
                    "base_rmse": base_rmse,
                    "source_only_rmse": p1_rmse,
                    "base_plus_source_rmse": p2_rmse,
                    "incremental_transfer_gain": base_rmse - p2_rmse,
                    "source_only_gain": base_rmse - p1_rmse,
                    "base_mae": _mae(base_val, y_val),
                    "source_only_mae": _mae(p1_val, y_val),
                    "base_plus_source_mae": _mae(p2_val, y_val),
                    "base_r2": base_r2,
                    "source_only_r2": _r2(p1_val, y_val),
                    "base_plus_source_r2": _r2(p2_val, y_val),
                    "selected_alpha_source_only": alpha_p1,
                    "selected_alpha_residual": alpha_p2,
                    "num_valid_source_features": int(keep_columns.sum()),
                    "dropped_constant_features": dropped,
                    "coefficient_l2_norm": float(np.linalg.norm(model_p2.coef_)),
                    "shuffle_gain_mean": _finite_mean(shuffle_gains),
                    "shuffle_gain_std": (
                        float(np.std(shuffle_gains)) if len(shuffle_gains) > 1 else float("nan")
                    ),
                    "shuffle_gain_p95": float(np.percentile(shuffle_gains, 95)),
                    "status": "OK",
                }
            )
            shuffle_rows.append(
                {
                    "seed": seed,
                    "target_task": target,
                    "shuffle_gain_mean": _finite_mean(shuffle_gains),
                    "shuffle_gain_std": float(np.std(shuffle_gains)) if len(shuffle_gains) > 1 else float("nan"),
                    "shuffle_gain_p95": float(np.percentile(shuffle_gains, 95)),
                    "real_incremental_gain": base_rmse - p2_rmse,
                }
            )
            for probe, alpha, model in (("source_only", alpha_p1, model_p1), ("base_plus_residual", alpha_p2, model_p2)):
                for source, coefficient in zip(
                    [column for column, keep in zip(source_columns, keep_columns) if keep],
                    model.coef_,
                ):
                    coefficient_rows.append(
                        {
                            "seed": seed,
                            "target_task": target,
                            "probe": probe,
                            "source_task": source[len("src__") :],
                            "alpha": alpha,
                            "coefficient": float(coefficient),
                        }
                    )
            cv_rows.extend(
                {"seed": seed, "target_task": target, "probe": probe, **row}
                for probe, rows in (("source_only", cv_p1), ("base_plus_residual", cv_p2))
                for row in rows
            )

            macro["base"].append(base_rmse)
            macro["p1"].append(p1_rmse)
            macro["p2"].append(p2_rmse)

            # per-source train correlations with label and residual (§46-§47)
            for source_index, source in enumerate([column for column, keep in zip(source_columns, keep_columns) if keep]):
                source_values = x_train_raw[:, source_columns.index(source)]
                correlation_rows.append(
                    {
                        "seed": seed,
                        "target_task": target,
                        "source_task": source[len("src__") :],
                        "pearson_label": _pearson(source_values, y_train),
                        "spearman_label": _spearman(source_values, y_train),
                        "pearson_residual": _pearson(source_values, residual_train),
                        "spearman_residual": _spearman(source_values, residual_train),
                    }
                )
        if macro["base"]:
            summary_rows.append(
                {
                    "seed": seed,
                    "target_task": "__human3_macro__",
                    "base_rmse": _finite_mean(macro["base"]),
                    "source_only_rmse": _finite_mean(macro["p1"]),
                    "base_plus_source_rmse": _finite_mean(macro["p2"]),
                    "incremental_transfer_gain": _finite_mean(macro["base"]) - _finite_mean(macro["p2"]),
                    "source_only_gain": _finite_mean(macro["base"]) - _finite_mean(macro["p1"]),
                    "status": "OK",
                }
            )
            macro_by_seed[seed] = macro

    _write_csv(
        output_dir / "D5_1_SOURCE_PROBE_SUMMARY.csv",
        ["seed", "target_task", "n_train", "n_val", "base_rmse", "source_only_rmse",
         "base_plus_source_rmse", "incremental_transfer_gain", "source_only_gain", "base_mae",
         "source_only_mae", "base_plus_source_mae", "base_r2", "source_only_r2",
         "base_plus_source_r2", "selected_alpha_source_only", "selected_alpha_residual",
         "num_valid_source_features", "dropped_constant_features", "coefficient_l2_norm",
         "shuffle_gain_mean", "shuffle_gain_std", "shuffle_gain_p95", "status"],
        summary_rows,
    )
    _write_csv(
        output_dir / "D5_1_SOURCE_PROBE_PER_SAMPLE.csv",
        ["seed", "sample_id", "row_index", "task", "label", "base_prediction",
         "source_only_prediction", "base_plus_source_prediction", "base_error",
         "source_only_error", "base_plus_source_error"],
        per_sample_rows,
    )
    _write_csv(
        output_dir / "D5_1_RIDGE_CV_RESULTS.csv",
        ["seed", "target_task", "probe", "alpha", "mean_cv_rmse", "n_folds"],
        cv_rows,
    )
    _write_csv(
        output_dir / "D5_1_RIDGE_COEFFICIENTS.csv",
        ["seed", "target_task", "probe", "source_task", "alpha", "coefficient"],
        coefficient_rows,
    )

    stability_rows = []
    for probe in ("source_only", "base_plus_residual"):
        for target in HUMAN_TASKS:
            vectors = {}
            for seed in SEEDS:
                coefficients = [
                    float(row["coefficient"])
                    for row in coefficient_rows
                    if row["seed"] == seed and row["target_task"] == target and row["probe"] == probe
                ]
                if coefficients:
                    vectors[seed] = np.array(coefficients)
            seed_list = sorted(vectors)
            for index_a, seed_a in enumerate(seed_list):
                for seed_b in seed_list[index_a + 1 :]:
                    left, right = vectors[seed_a], vectors[seed_b]
                    top5_a = set(np.argsort(-np.abs(left))[:5].tolist())
                    top5_b = set(np.argsort(-np.abs(right))[:5].tolist())
                    top10_a = set(np.argsort(-np.abs(left))[:10].tolist())
                    top10_b = set(np.argsort(-np.abs(right))[:10].tolist())
                    stability_rows.append(
                        {
                            "probe": probe,
                            "target_task": target,
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "coefficient_spearman": _spearman(left, right),
                            "coefficient_cosine": _cosine(left, right),
                            "top5_overlap": len(top5_a & top5_b) / 5.0,
                            "top10_overlap": len(top10_a & top10_b) / 10.0,
                        }
                    )
    _write_csv(
        output_dir / "D5_1_COEFFICIENT_STABILITY.csv",
        ["probe", "target_task", "seed_a", "seed_b", "coefficient_spearman", "coefficient_cosine",
         "top5_overlap", "top10_overlap"],
        stability_rows,
    )
    _write_csv(
        output_dir / "D5_1_SHUFFLE_CONTROL.csv",
        ["seed", "target_task", "shuffle_gain_mean", "shuffle_gain_std", "shuffle_gain_p95",
         "real_incremental_gain"],
        shuffle_rows,
    )
    _write_csv(
        output_dir / "D5_1_TRAIN_SOURCE_TARGET_CORRELATION.csv",
        ["seed", "target_task", "source_task", "pearson_label", "spearman_label"],
        [
            {key: row[key] for key in ("seed", "target_task", "source_task", "pearson_label", "spearman_label")}
            for row in correlation_rows
        ],
    )
    _write_csv(
        output_dir / "D5_1_TRAIN_SOURCE_RESIDUAL_CORRELATION.csv",
        ["seed", "target_task", "source_task", "pearson_residual", "spearman_residual"],
        [
            {key: row[key] for key in ("seed", "target_task", "source_task", "pearson_residual", "spearman_residual")}
            for row in correlation_rows
        ],
    )

    paired = [
        summary_rows_row
        for summary_rows_row in summary_rows
        if summary_rows_row.get("target_task") == "__human3_macro__"
    ]
    gains = [float(row["incremental_transfer_gain"]) for row in paired]
    positive = sum(1 for gain in gains if gain > 0)
    print(
        json.dumps(
            {
                "human3_p2_paired_mean_gain": _finite_mean(gains),
                "seeds_positive": positive,
                "seeds_compared": len(gains),
                "per_seed": {str(row["seed"]): float(row["incremental_transfer_gain"]) for row in paired},
            },
            indent=2,
        )
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = math.sqrt(float((left**2).sum()) * float((right**2).sum()))
    if denominator == 0:
        return float("nan")
    return float((left * right).sum() / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["collect", "analyze"])
    parser.add_argument("--gpu_id", default="1")
    parser.add_argument("--data_dir", default="d5_stage1_data")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    features_by_seed = {seed: data_dir / f"hps_features_seed{seed}.csv" for seed in SEEDS}

    if args.phase == "collect":
        for seed in args.seeds:
            if features_by_seed[seed].exists():
                print(f"seed {seed} features already collected; skipping")
                continue
            collect_seed(seed, Path(HPS_RUN_DIRS[seed]), args.gpu_id, data_dir)
        return

    missing = [seed for seed in SEEDS if not features_by_seed[seed].exists()]
    if missing:
        raise SystemExit(f"missing feature files for seeds {missing}; run collect first")
    run_d5_0(features_by_seed, Path(args.output_dir))
    run_d5_1(features_by_seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
