"""Analysis helpers for explicit HPS/route/final routing diagnostics.

The helpers accept both the legacy Prompt-only diagnostic names and the RGCER
names.  This keeps baseline comparisons possible without making the proposed
model depend on the old ``transfer_gate`` implementation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import pandas as pd
import torch

from architecture.prediction_heads import decode_prediction


def _unpack(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def _diag_value(diagnostics, *names, default=None):
    if diagnostics is None:
        return default
    if hasattr(diagnostics, "as_dict"):
        diagnostics = diagnostics.as_dict()
    if not isinstance(diagnostics, dict):
        diagnostics = vars(diagnostics)
    for name in names:
        if name in diagnostics:
            return diagnostics[name]
    return default


def _prediction_column(predictions, task_name):
    raw = predictions[task_name]
    if raw.ndim == 1:
        return raw.reshape(-1)
    return raw[..., 0]


def _decode_output(raw, task_name, prediction_mode, trainer=None, decode_fn: Optional[Callable] = None, apply_conformal=False):
    """Decode one raw output into the same units as the labels.

    ``Trainer.decode_task_output`` is preferred because it applies the task
    scaler and optional CQR state.  ``decode_fn`` keeps the analysis helpers
    usable by external evaluation pipelines without importing Trainer.
    """

    if trainer is not None:
        return trainer.decode_task_output(task_name, raw, apply_conformal=apply_conformal)
    if decode_fn is not None:
        try:
            return decode_fn(task_name, raw, apply_conformal=apply_conformal)
        except TypeError:
            return decode_fn(task_name, raw)
    return decode_prediction(raw, mode=prediction_mode).as_dict()


def _as_batch_weights(diagnostics, batch_size, task_count, device):
    weights = _diag_value(diagnostics, "source_weights", "routing_weights")
    if weights is None:
        weights = torch.zeros(batch_size, task_count, device=device)
    return weights


def _as_null_weight(diagnostics, batch_size, device, dtype=torch.float32):
    null = _diag_value(diagnostics, "null_weight")
    if null is not None:
        return null.reshape(batch_size, -1)[:, :1]
    gate = _diag_value(diagnostics, "transfer_gate")
    if gate is not None:
        return 1.0 - gate.reshape(batch_size, -1)[:, :1]
    return torch.ones(batch_size, 1, device=device, dtype=dtype)


@torch.no_grad()
def collect_routing_records(
    model,
    dataloader,
    target_task: str,
    task_names: List[str],
    device=None,
    trainer=None,
    decode_fn: Optional[Callable] = None,
    prediction_mode: str = "point",
    apply_conformal: bool = False,
) -> pd.DataFrame:
    """Collect one CSV-ready row per sample for a target task.

    ``source_weights`` are conditional-on-transfer weights for RGCER.  Legacy
    Prompt-only models expose ``routing_weights`` and are accepted as-is.
    """

    model.eval()
    records = []
    for batch_index, batch in enumerate(dataloader):
        if batch.get("is_empty", False):
            continue
        if device is not None:
            batch = batch.to(device)
        if trainer is not None:
            predictions, diagnostics = _unpack(trainer.predict_all_tasks(batch, return_aux=True))
            if isinstance(diagnostics, dict):
                diagnostics = diagnostics.get(target_task, diagnostics)
        else:
            predictions, diagnostics = _unpack(model(batch, task_name=target_task, return_aux=True))
        batch_size = predictions[target_task].size(0)
        weights = _as_batch_weights(diagnostics, batch_size, len(task_names), predictions[target_task].device)
        weights = weights.detach().cpu()
        null = _as_null_weight(diagnostics, batch_size, weights.device, weights.dtype).detach().cpu()
        final_decoded = _decode_output(
            predictions[target_task], target_task, prediction_mode, trainer, decode_fn, apply_conformal
        )
        final_prediction = final_decoded["median"].reshape(-1).detach().cpu()
        base_raw = _diag_value(diagnostics, "base_raw")
        route_raw = _diag_value(diagnostics, "route_raw")
        base_decoded = _decode_output(
            base_raw if base_raw is not None else predictions[target_task],
            target_task,
            prediction_mode,
            trainer,
            decode_fn,
            False,
        )
        route_decoded = _decode_output(
            route_raw if route_raw is not None else predictions[target_task],
            target_task,
            prediction_mode,
            trainer,
            decode_fn,
            False,
        )
        base_prediction = base_decoded["median"].reshape(-1).detach().cpu()
        route_prediction = route_decoded["median"].reshape(-1).detach().cpu()
        interval_lower = final_decoded["lower"].reshape(-1).detach().cpu()
        interval_upper = final_decoded["upper"].reshape(-1).detach().cpu()
        target = getattr(batch, "y", None)
        target = target.reshape(-1).detach().cpu() if target is not None else None
        for sample_index in range(batch_size):
            row: Dict[str, object] = {
                "batch_index": batch_index,
                "sample_index": sample_index,
                "target_task": target_task,
                "null_weight": float(null[sample_index, 0]),
                "transfer_mass": float(1.0 - null[sample_index, 0]),
                "base_prediction": float(base_prediction[sample_index]),
                "hps_base_prediction": float(base_prediction[sample_index]),
                "route_prediction": float(route_prediction[sample_index]),
                "final_prediction": float(final_prediction[sample_index]),
                "baseline_final_prediction": float(final_prediction[sample_index]),
                "interval_lower": float(interval_lower[sample_index]),
                "interval_upper": float(interval_upper[sample_index]),
                "interval_width": float(interval_upper[sample_index] - interval_lower[sample_index]),
                # Compatibility alias for existing downstream notebooks.
                "prediction": float(final_prediction[sample_index]),
            }
            if target is not None:
                row["target"] = float(target[sample_index])
                row["route_regret"] = abs(row["route_prediction"] - row["target"]) - abs(
                    row["base_prediction"] - row["target"]
                )
            if hasattr(batch, "sample_id"):
                row["sample_id"] = batch.sample_id[sample_index]
            if hasattr(batch, "smiles"):
                row["smiles"] = batch.smiles[sample_index]
            for source_index, source_task in enumerate(task_names):
                row[f"route::{source_task}"] = float(weights[sample_index, source_index])
            active_sources = torch.where(weights[sample_index] > 0)[0]
            if active_sources.numel():
                top_source = active_sources[weights[sample_index, active_sources].argmax()]
                row["top_source_1"] = task_names[int(top_source)]
                row["top_source_weight_1"] = float(weights[sample_index, top_source])
            else:
                row["top_source_1"] = None
                row["top_source_weight_1"] = 0.0
            records.append(row)
    return pd.DataFrame(records)


def mean_routing_weights(records: pd.DataFrame, task_names: Optional[List[str]] = None) -> pd.Series:
    """Return dataset-level mean conditional source routing for a target task."""

    if task_names is None:
        task_names = [column.removeprefix("route::") for column in records.columns if column.startswith("route::")]
    columns = [f"route::{task_name}" for task_name in task_names]
    if records.empty:
        return pd.Series(0.0, index=task_names, dtype=float)
    return records[columns].mean(axis=0).set_axis(task_names)


def top_k_routing(records: pd.DataFrame, task_names: List[str], k: int = 5) -> pd.DataFrame:
    """Return sample-level top-k *real* source tasks and weights."""

    if k <= 0:
        raise ValueError("k must be positive")
    route_columns = [f"route::{task_name}" for task_name in task_names]
    rows = []
    for row_index, row in records.iterrows():
        values = row[route_columns].astype(float)
        values = values[values > 0].sort_values(ascending=False).head(k)
        for source, weight in values.items():
            rows.append(
                {
                    "record_index": row_index,
                    "target_task": row.get("target_task"),
                    "null_weight": row.get("null_weight"),
                    "prediction": row.get("prediction"),
                    "source_task": source.removeprefix("route::"),
                    "routing_weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def routing_summary(records: pd.DataFrame, task_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Summarize NULL mass, entropy-compatible sparsity, and route variance."""

    if task_names is None:
        task_names = [column.removeprefix("route::") for column in records.columns if column.startswith("route::")]
    if records.empty:
        return {
            "mean_null": 1.0,
            "mean_transfer_mass": 0.0,
            "mean_active_routes": 0.0,
            "routing_variance": 0.0,
        }
    weights = torch.tensor(records[[f"route::{task}" for task in task_names]].to_numpy(), dtype=torch.float32)
    null = torch.tensor(records.get("null_weight", pd.Series(1.0, index=records.index)).to_numpy(), dtype=torch.float32)
    return {
        "mean_null": float(null.mean()),
        "mean_transfer_mass": float((1.0 - null).mean()),
        "mean_active_routes": float((weights > 0).sum(dim=-1).float().mean()),
        "routing_variance": float(weights.var(dim=0, unbiased=False).mean()),
    }


def make_source_mask(task_names: List[str], excluded_tasks: List[str]) -> torch.Tensor:
    """Build a boolean source mask for a global source deletion."""

    excluded = set(excluded_tasks)
    unknown = excluded.difference(task_names)
    if unknown:
        raise KeyError(f"Unknown source task(s): {sorted(unknown)}")
    return torch.tensor([task_name not in excluded for task_name in task_names], dtype=torch.bool)


__all__ = [
    "collect_routing_records",
    "make_source_mask",
    "mean_routing_weights",
    "routing_summary",
    "top_k_routing",
]
