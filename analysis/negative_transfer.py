"""Per-sample routing interventions and shuffled-endpoint stress helpers."""

from __future__ import annotations

import copy
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd
import torch

from architecture.prediction_heads import decode_prediction


def _unpack(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def _diag(diagnostics, name, default=None):
    if diagnostics is None:
        return default
    if hasattr(diagnostics, "as_dict"):
        diagnostics = diagnostics.as_dict()
    if not isinstance(diagnostics, dict):
        diagnostics = vars(diagnostics)
    return diagnostics.get(name, default)


def _median_interval(raw, prediction_mode="quantile"):
    if prediction_mode in {"point", "quantile"}:
        decoded = decode_prediction(raw, mode=prediction_mode)
        return decoded.median, decoded.lower, decoded.upper
    raise ValueError("prediction_mode must be 'point' or 'quantile'")


def _decoded_interval(
    raw,
    task_name,
    prediction_mode,
    trainer=None,
    decode_fn: Callable | None = None,
    apply_conformal=False,
):
    if trainer is not None:
        decoded = trainer.decode_task_output(task_name, raw, apply_conformal=apply_conformal)
        return decoded["median"], decoded["lower"], decoded["upper"]
    if decode_fn is not None:
        try:
            decoded = decode_fn(task_name, raw, apply_conformal=apply_conformal)
        except TypeError:
            decoded = decode_fn(task_name, raw)
        return decoded["median"], decoded["lower"], decoded["upper"]
    return _median_interval(raw, prediction_mode)


def intervention_masks(
    source_weights: torch.Tensor,
    strategies: Iterable[str] = ("top", "random", "lowest"),
    seed: int = 42,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Create per-sample masks for top/random/lowest source deletion.

    Returns ``strategy -> (mask, deleted_index)``.  Samples with no real source
    are left unchanged and receive a deleted index of ``-1``.  The input is
    expected to contain conditional source weights, with zero for unavailable
    endpoints and for the excluded target.
    """

    weights = torch.as_tensor(source_weights)
    if weights.ndim != 2:
        raise ValueError("source_weights must have shape [B, T]")
    batch_size, task_count = weights.shape
    available = weights > 0
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    result = {}
    for strategy in strategies:
        if strategy not in {"top", "random", "lowest"}:
            raise ValueError("strategy must be 'top', 'random', or 'lowest'")
        mask = torch.ones(batch_size, task_count, dtype=torch.bool, device=weights.device)
        deleted = torch.full((batch_size,), -1, dtype=torch.long, device=weights.device)
        for row in range(batch_size):
            candidates = torch.where(available[row])[0]
            if candidates.numel() == 0:
                continue
            values = weights[row, candidates]
            if strategy == "top":
                candidate = candidates[values.argmax()]
            elif strategy == "lowest":
                candidate = candidates[values.argmin()]
            else:
                # Sample on CPU so the same seed is reproducible on CPU/CUDA.
                selected = torch.randint(candidates.numel(), (1,), generator=generator).item()
                candidate = candidates[selected]
            mask[row, candidate] = False
            deleted[row] = candidate
        result[strategy] = (mask, deleted)
    return result


@torch.no_grad()
def source_deletion_analysis(
    model,
    batch,
    target_task: str,
    task_names: Sequence[str],
    prediction_mode: str = "quantile",
    strategies: Iterable[str] = ("top", "random", "lowest"),
    device=None,
    seed: int = 42,
    trainer=None,
    decode_fn: Callable | None = None,
    apply_conformal: bool = True,
) -> pd.DataFrame:
    """Run per-sample source deletion and return a long-form comparison table."""

    model.eval()
    if device is not None:
        batch = batch.to(device)
    baseline_predictions, baseline_diagnostics = _unpack(
        model(batch, task_name=target_task, return_aux=True)
    )
    raw = baseline_predictions[target_task]
    baseline_median, baseline_lower, baseline_upper = _decoded_interval(
        raw,
        target_task,
        prediction_mode,
        trainer=trainer,
        decode_fn=decode_fn,
        apply_conformal=apply_conformal,
    )
    weights = _diag(baseline_diagnostics, "source_weights")
    if weights is None:
        weights = _diag(baseline_diagnostics, "routing_weights")
    if weights is None:
        weights = torch.zeros(raw.size(0), len(task_names), device=raw.device)
    masks = intervention_masks(weights, strategies=strategies, seed=seed)
    target = getattr(batch, "y", None)
    target = target.reshape(-1, 1).to(raw.device) if target is not None else None
    baseline_null = _diag(baseline_diagnostics, "null_weight")
    if baseline_null is None:
        gate = _diag(baseline_diagnostics, "transfer_gate")
        baseline_null = 1.0 - gate if gate is not None else torch.ones(raw.size(0), 1, device=raw.device)
    rows = []
    for strategy, (mask, deleted_indices) in masks.items():
        intervened_predictions, intervened_diagnostics = _unpack(
            model(batch, task_name=target_task, return_aux=True, source_mask=mask)
        )
        intervention_median, intervention_lower, intervention_upper = _decoded_interval(
            intervened_predictions[target_task],
            target_task,
            prediction_mode,
            trainer=trainer,
            decode_fn=decode_fn,
            apply_conformal=apply_conformal,
        )
        intervention_null = _diag(intervened_diagnostics, "null_weight", baseline_null)
        for row in range(raw.size(0)):
            target_value = None if target is None else float(target[row, 0].cpu())
            baseline_error = None if target is None else abs(float(baseline_median[row, 0].cpu()) - target_value)
            intervention_error = None if target is None else abs(float(intervention_median[row, 0].cpu()) - target_value)
            deleted_index = int(deleted_indices[row].cpu())
            rows.append(
                {
                    "sample_index": row,
                    "target_task": target_task,
                    "strategy": strategy,
                    "deleted_source_index": deleted_index,
                    "deleted_source_task": task_names[deleted_index] if deleted_index >= 0 else None,
                    "target": target_value,
                    "base_prediction": float(baseline_median[row, 0].cpu()),
                    "intervention_prediction": float(intervention_median[row, 0].cpu()),
                    "delta_prediction": float(intervention_median[row, 0].cpu() - baseline_median[row, 0].cpu()),
                    "base_abs_error": baseline_error,
                    "intervention_abs_error": intervention_error,
                    "delta_abs_error": None
                    if baseline_error is None
                    else intervention_error - baseline_error,
                    "base_null_weight": float(baseline_null[row, 0].cpu()),
                    "intervention_null_weight": float(intervention_null[row, 0].cpu()),
                    "delta_null_weight": float(intervention_null[row, 0].cpu() - baseline_null[row, 0].cpu()),
                    "base_interval_width": float((baseline_upper[row, 0] - baseline_lower[row, 0]).cpu()),
                    "interval_lower": float(baseline_lower[row, 0].cpu()),
                    "interval_upper": float(baseline_upper[row, 0].cpu()),
                    "interval_width": float((baseline_upper[row, 0] - baseline_lower[row, 0]).cpu()),
                    "intervention_interval_lower": float(intervention_lower[row, 0].cpu()),
                    "intervention_interval_upper": float(intervention_upper[row, 0].cpu()),
                    "intervention_interval_width": float(
                        (intervention_upper[row, 0] - intervention_lower[row, 0]).cpu()
                    ),
                    "delta_interval_width": float(
                        (intervention_upper[row, 0] - intervention_lower[row, 0]).cpu()
                        - (baseline_upper[row, 0] - baseline_lower[row, 0]).cpu()
                    ),
                }
            )
    return pd.DataFrame(rows)


def shuffled_endpoint_name(task_name: str) -> str:
    """Return an analysis-only endpoint name without changing the task registry."""

    return f"shuffled_{task_name}"


def shuffle_endpoint_batch(batch, seed: int):
    """Return a copy with a deterministic label permutation for stress tests."""

    result = copy.deepcopy(batch)
    labels = result.y.reshape(-1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(labels.numel(), generator=generator, device="cpu").to(labels.device)
    result.y = labels[permutation].reshape(-1, 1)
    return result


def _batch_ids(batch, offset):
    sample_ids = getattr(batch, "sample_id", None)
    if sample_ids is None:
        return [f"__position_{offset + index}" for index in range(batch.y.reshape(-1).numel())]
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    return [str(value) for value in sample_ids]


def shuffled_endpoint_batches(dataloader, seed: int = 42):
    """Yield a fixed-seed, split-global shuffled-label view of a loader.

    The permutation is constructed once from sample IDs for the whole split,
    rather than once per mini-batch.  Consequently the same sample receives
    the same shuffled endpoint regardless of batch size or batch boundaries.
    Use a different seed per split when constructing train/validation/
    calibration/test views.  The shuffled endpoint is an analysis replica and
    is not included in the formal ToxAcute macro score.
    """

    batches = list(dataloader)
    if not batches:
        return
    sample_ids = []
    labels = []
    batch_id_lists = []
    offset = 0
    for batch in batches:
        current_ids = _batch_ids(batch, offset)
        current_labels = batch.y.reshape(-1).detach().cpu()
        if len(current_ids) != current_labels.numel():
            raise ValueError("Each sample_id list must have one entry per endpoint label")
        batch_id_lists.append(current_ids)
        sample_ids.extend(current_ids)
        labels.extend(current_labels.tolist())
        offset += len(current_ids)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values must be unique within a shuffled split")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(len(sample_ids), generator=generator).tolist()
    shuffled_label_by_id = {
        sample_ids[target_index]: labels[source_index]
        for target_index, source_index in enumerate(permutation)
    }
    for batch, ids in zip(batches, batch_id_lists):
        result = copy.deepcopy(batch)
        values = torch.tensor(
            [shuffled_label_by_id[sample_id] for sample_id in ids],
            dtype=result.y.dtype,
            device=result.y.device,
        )
        result.y = values.reshape_as(result.y)
        yield result


__all__ = [
    "intervention_masks",
    "source_deletion_analysis",
    "shuffled_endpoint_batches",
    "shuffled_endpoint_name",
    "shuffle_endpoint_batch",
]
