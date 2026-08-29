"""Read-only diagnostic logging for RGCER diagnostic runs (plan §6-§9).

Writes per-epoch CSV/JSON artifacts under ``<save_path>/diagnostics`` without
changing any forward/loss behavior: every input is data that training and
validation passes already materialize (route records, routing summaries,
loss bundles, gradients).  Base/route/final path metrics make it possible to
distinguish "route never learned" from "NULL rejected a good route".
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS
from representation_audit import (
    REPRESENTATION_AGGREGATE_FIELDS,
    REPRESENTATION_METRIC_FIELDS,
)


# Parameter-name prefixes per diagnostic group (plan §6.6).  Groups missing
# from a given architecture (e.g. HPS has no router) simply record NaN.
GRAD_GROUPS = {
    "router_query_projection": ("encoder.task_conditioner.router.query_projection",),
    "router_key_projection": ("encoder.task_conditioner.router.key_projection",),
    "router_null": (
        "encoder.task_conditioner.router.null_token",
        "encoder.task_conditioner.router.null_key_projection",
    ),
    "film_generator": ("encoder.task_conditioner.adapter.film_head",),
    "adapter": (
        "encoder.task_conditioner.adapter.adapter_down",
        "encoder.task_conditioner.adapter.adapter_up",
    ),
    "backbone": ("encoder.backbone",),
    "prediction_heads": ("decoders.",),
}

ROUTING_FIELDS = (
    "mean_null_weight",
    "std_null",
    "mean_transfer_mass",
    "mean_joint_source_mass",
    "mean_routing_entropy",
    "routing_variance",
    "routing_variance_joint",
    "mean_active_sources",
    "route_regret_mean",
    "final_regret_mean",
    "route_negative_transfer_rate",
    "final_negative_transfer_rate",
)

EPOCH_SUMMARY_FIELDS = (
    "epoch",
    "routing_enabled",
    "is_best",
    "train_human3_macro_rmse",
    "train_all59_macro_rmse",
    "val_human3_macro_rmse",
    "val_all59_macro_rmse",
    "val_selection_score",
    "val_selection_scope",
    "loss_final_mean",
    "loss_base_mean",
    "loss_total_mean",
    "loss_final_human3",
    "loss_base_human3",
    "loss_total_human3",
    "base_human3_rmse",
    "route_human3_rmse",
    "final_human3_rmse",
)
EPOCH_SUMMARY_FIELDS += tuple(f"human3_{field}" for field in ROUTING_FIELDS)

PATH_METRIC_FIELDS = ("rmse", "r2", "mae", "n")

PREDICTION_FIELDS = (
    "sample_id",
    "row_index",
    "task",
    "label",
    "base_prediction",
    "route_prediction",
    "final_prediction",
    "base_abs_error",
    "route_abs_error",
    "final_abs_error",
    "route_regret",
    "final_regret",
)

TOP_K_SOURCES = 8


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _stack(values):
    if not values:
        return np.empty(0, dtype=float)
    return torch.cat([torch.as_tensor(value) for value in values]).reshape(-1).numpy().astype(float)


def _path_metrics(prediction, label):
    if prediction.size == 0:
        return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan"), "n": 0}
    error = prediction - label
    result = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float("nan"),
        "mae": float(np.mean(np.abs(error))),
        "n": int(prediction.size),
    }
    if label.size >= 2 and np.unique(label).size >= 2:
        ss_total = float(np.sum((label - label.mean()) ** 2))
        if ss_total > 0:
            result["r2"] = float(1.0 - np.sum(error**2) / ss_total)
    return result


def _group_grad_norms(model):
    """One L2 norm per diagnostic group for the current ``.grad`` state.

    A group whose parameters exist but received no gradient records 0.0
    (diagnostically meaningful — e.g. a frozen/detached path); a group with
    no matching parameters at all (e.g. the router under HPS) records NaN.
    """

    sums = {group: None for group in GRAD_GROUPS}
    matched = {group: False for group in GRAD_GROUPS}
    for name, parameter in model.named_parameters():
        for group, prefixes in GRAD_GROUPS.items():
            if any(name.startswith(prefix) for prefix in prefixes):
                matched[group] = True
                if parameter.grad is not None:
                    total = float(parameter.grad.detach().float().pow(2).sum())
                    sums[group] = total if sums[group] is None else sums[group] + total
                break
    return {
        group: (
            math.sqrt(sums[group])
            if sums[group] is not None
            else (0.0 if matched[group] else float("nan"))
        )
        for group in GRAD_GROUPS
    }


def _write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class RunDiagnosticsWriter:
    """Aggregates per-epoch diagnostic rows and best-epoch per-sample dumps."""

    def __init__(self, output_dir, task_names, sample_row_index=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_names = list(task_names)
        self.human_tasks = [task for task in self.task_names if task in set(HUMAN_TARGET_TASKS)]
        self.sample_row_index = {str(key): int(value) for key, value in (sample_row_index or {}).items()}
        self.epoch_rows: list[dict] = []
        self.path_rows: list[dict] = []
        self.routing_rows: list[dict] = []
        self.gradient_rows: list[dict] = []
        self.representation_rows: list[dict] = []
        self._grad_steps: list[dict] = []
        self._path_by_epoch: dict[int, dict[str, dict]] = {}
        self._predictions_by_epoch: dict[int, list[dict]] = {}
        self._routing_by_epoch: dict[int, list[dict]] = {}
        self._representation_by_epoch: dict[int, list[dict]] = {}
        self.best_epoch: int | None = None

    # ------------------------------------------------------------------
    # Training-loop hooks
    # ------------------------------------------------------------------
    def record_gradients(self, model):
        """Capture one gradient-norm snapshot; call before optimizer.step()."""

        self._grad_steps.append(_group_grad_norms(model))

    def _flush_gradients(self, epoch):
        if not self._grad_steps:
            return
        for group in GRAD_GROUPS:
            values = [step[group] for step in self._grad_steps if math.isfinite(step[group])]
            finite = [value for value in values if math.isfinite(value)]
            self.gradient_rows.append(
                {
                    "epoch": epoch,
                    "group": group,
                    "mean": float(np.mean(finite)) if finite else float("nan"),
                    "max": float(np.max(finite)) if finite else float("nan"),
                    "last": values[-1] if values else float("nan"),
                    "steps": len(self._grad_steps),
                }
            )
        self._grad_steps = []

    @staticmethod
    def _mean_loss(per_task):
        values = [value for value in per_task.values() if math.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    def _human3_loss(self, per_task):
        values = [per_task[task] for task in self.human_tasks if math.isfinite(per_task.get(task, float("nan")))]
        return float(np.mean(values)) if values else float("nan")

    def log_epoch(
        self,
        epoch,
        train_result,
        validation_result,
        route_records,
        *,
        routing_enabled,
        is_best,
        representation_records=None,
    ):
        """Append one epoch's summary/path/routing/gradient/representation rows."""

        self._flush_gradients(epoch)
        routing = validation_result.get("routing", {}) or {}
        for task, values in routing.items():
            self.routing_rows.append(
                {"epoch": epoch, "task": task, **{field: _float(values.get(field)) for field in ROUTING_FIELDS}}
            )

        # D3 §27: per-epoch representation-path aggregates for the human3 tasks.
        representation_rows = self._collect_representation_aggregates(representation_records)
        self._representation_by_epoch[epoch] = self._collect_human3_representation(
            representation_records
        )
        for row in representation_rows:
            self.representation_rows.append({"epoch": epoch, **row})

        epoch_path = self._collect_path_metrics(route_records)
        self._path_by_epoch[epoch] = epoch_path
        for task, per_path in epoch_path.items():
            for path, metrics in per_path.items():
                self.path_rows.append({"epoch": epoch, "task": task, "path": path, **metrics})

        final_losses = train_result.get("final_loss", {}) or {}
        base_losses = train_result.get("base_loss", {}) or {}
        total_losses = train_result.get("loss", {}) or {}
        human3_routing = self._human3_routing_aggregate(routing)
        row = {
            "epoch": epoch,
            "routing_enabled": bool(routing_enabled),
            "is_best": bool(is_best),
            "train_human3_macro_rmse": _float(train_result.get("human3_macro_rmse")),
            "train_all59_macro_rmse": _float(train_result.get("all_task_macro_rmse")),
            "val_human3_macro_rmse": _float(validation_result.get("human3_macro_rmse")),
            "val_all59_macro_rmse": _float(validation_result.get("all_task_macro_rmse")),
            "val_selection_score": _float(validation_result.get("selection_score")),
            "val_selection_scope": validation_result.get("selection_scope", ""),
            "loss_final_mean": self._mean_loss({k: _float(v) for k, v in final_losses.items()}),
            "loss_base_mean": self._mean_loss({k: _float(v) for k, v in base_losses.items()}),
            "loss_total_mean": self._mean_loss({k: _float(v) for k, v in total_losses.items()}),
            "loss_final_human3": self._human3_loss({k: _float(v) for k, v in final_losses.items()}),
            "loss_base_human3": self._human3_loss({k: _float(v) for k, v in base_losses.items()}),
            "loss_total_human3": self._human3_loss({k: _float(v) for k, v in total_losses.items()}),
            "base_human3_rmse": self._human3_macro(epoch_path, "base", "rmse"),
            "route_human3_rmse": self._human3_macro(epoch_path, "route", "rmse"),
            "final_human3_rmse": self._human3_macro(epoch_path, "final", "rmse"),
        }
        row.update({f"human3_{field}": _float(value) for field, value in human3_routing.items()})
        self.epoch_rows.append(row)
        self._write_current_files()

    # ------------------------------------------------------------------
    # Path metrics and per-sample caches
    # ------------------------------------------------------------------
    def _collect_representation_aggregates(self, representation_records):
        if not representation_records:
            return []
        aggregates = []
        for task in self.task_names:
            record = representation_records.get(task)
            if not record or not record["metrics"]:
                continue
            stacked = {
                key: torch.cat([entry[key] for entry in record["metrics"]])
                for key in record["metrics"][0]
            }
            row = {"task": task}
            for field in REPRESENTATION_METRIC_FIELDS:
                values = stacked.get(field)
                row[f"{field}_mean"] = (
                    float(values.float().mean()) if values is not None and values.numel() else float("nan")
                )
            aggregates.append(row)
        return aggregates

    def _collect_human3_representation(self, representation_records):
        if not representation_records:
            return []
        rows = []
        for task in self.human_tasks:
            record = representation_records.get(task)
            if not record or not record["metrics"]:
                continue
            stacked = {
                key: torch.cat([entry[key] for entry in record["metrics"]])
                for key in record["metrics"][0]
            }
            sample_ids = list(record["sample_id"])
            count = len(sample_ids)
            for index in range(count):
                row = {
                    "sample_id": sample_ids[index],
                    "row_index": self.sample_row_index.get(str(sample_ids[index]), ""),
                    "task": task,
                }
                for field in REPRESENTATION_METRIC_FIELDS:
                    values = stacked.get(field)
                    row[field] = (
                        float(values[index]) if values is not None and index < values.numel() else float("nan")
                    )
                rows.append(row)
        return rows

    def _collect_path_metrics(self, route_records):
        if not route_records:
            return {}
        collected = {}
        for task in self.task_names:
            record = route_records.get(task)
            if not record or not record.get("target"):
                continue
            target = _stack(record["target"])
            per_path = {}
            for path in ("base", "route", "final"):
                per_path[path] = _path_metrics(_stack(record.get(path)), target)
            collected[task] = per_path
        return collected

    def _human3_macro(self, epoch_path, path, metric):
        values = [
            per_path[path][metric]
            for task, per_path in epoch_path.items()
            if task in set(self.human_tasks) and math.isfinite(per_path[path][metric])
        ]
        return float(np.mean(values)) if values else float("nan")

    def _human3_routing_aggregate(self, routing):
        aggregate = {}
        for field in ROUTING_FIELDS:
            values = [
                _float(routing[task].get(field))
                for task in self.human_tasks
                if task in routing and math.isfinite(_float(routing[task].get(field)))
            ]
            aggregate[field] = float(np.mean(values)) if values else float("nan")
        return aggregate

    def _collect_human3_samples(self, route_records):
        prediction_rows: list[dict] = []
        routing_rows: list[dict] = []
        for task in self.human_tasks:
            record = (route_records or {}).get(task)
            if not record or not record.get("target"):
                continue
            targets = _stack(record["target"])
            base = _stack(record.get("base"))
            route = _stack(record.get("route"))
            final = _stack(record.get("final"))
            route_regret = _stack(record.get("route_regret"))
            final_regret = _stack(record.get("final_regret"))
            null = _stack(record.get("null"))
            entropy = _stack(record.get("entropy"))
            # source/joint weights arrive as one [batch, task] tensor per batch;
            # concatenate once so per-sample indexing below is global.
            source_weights = record.get("source_weights") or []
            joint_weights = record.get("joint_source_weights") or []
            conditional_all = (
                torch.cat([torch.as_tensor(tensor) for tensor in source_weights])
                if source_weights
                else torch.empty((0, len(self.task_names)))
            )
            joint_all = (
                torch.cat([torch.as_tensor(tensor) for tensor in joint_weights])
                if joint_weights
                else torch.empty((0, len(self.task_names)))
            )
            sample_ids = list(record.get("sample_id") or [])
            count = targets.size
            if sample_ids and len(sample_ids) != count:
                sample_ids = []
            for index in range(count):
                sample_id = sample_ids[index] if index < len(sample_ids) else ""
                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "row_index": self.sample_row_index.get(str(sample_id), ""),
                        "task": task,
                        "label": float(targets[index]),
                        "base_prediction": float(base[index]) if index < base.size else float("nan"),
                        "route_prediction": float(route[index]) if index < route.size else float("nan"),
                        "final_prediction": float(final[index]) if index < final.size else float("nan"),
                        "base_abs_error": abs(float(base[index]) - float(targets[index]))
                        if index < base.size
                        else float("nan"),
                        "route_abs_error": abs(float(route[index]) - float(targets[index]))
                        if index < route.size
                        else float("nan"),
                        "final_abs_error": abs(float(final[index]) - float(targets[index]))
                        if index < final.size
                        else float("nan"),
                        "route_regret": float(route_regret[index]) if index < route_regret.size else float("nan"),
                        "final_regret": float(final_regret[index]) if index < final_regret.size else float("nan"),
                    }
                )
                conditional = conditional_all[index] if index < conditional_all.size(0) else torch.empty(0)
                joint = joint_all[index] if index < joint_all.size(0) else torch.empty(0)
                row = {
                    "sample_id": sample_id,
                    "row_index": self.sample_row_index.get(str(sample_id), ""),
                    "task": task,
                    "null_weight": float(null[index]) if index < null.size else float("nan"),
                    "transfer_mass": 1.0 - float(null[index]) if index < null.size else float("nan"),
                }
                order = torch.argsort(conditional, descending=True)[:TOP_K_SOURCES]
                for rank in range(TOP_K_SOURCES):
                    if rank < order.numel() and float(conditional[order[rank]]) > 0:
                        column = int(order[rank])
                        source_name = (
                            self.task_names[column] if 0 <= column < len(self.task_names) else f"column_{column}"
                        )
                        row[f"top{rank + 1}_source"] = source_name
                        row[f"top{rank + 1}_conditional_weight"] = float(conditional[column])
                        row[f"top{rank + 1}_joint_weight"] = float(joint[column]) if column < joint.numel() else 0.0
                    else:
                        row[f"top{rank + 1}_source"] = ""
                        row[f"top{rank + 1}_conditional_weight"] = 0.0
                        row[f"top{rank + 1}_joint_weight"] = 0.0
                row["routing_entropy"] = float(entropy[index]) if index < entropy.size else float("nan")
                routing_rows.append(row)
        return prediction_rows, routing_rows

    # ------------------------------------------------------------------
    # Final artifacts
    # ------------------------------------------------------------------
    def note_best_epoch(self, epoch, route_records, representation_records=None):
        """Cache the current epoch's per-sample rows when it becomes best."""

        if route_records is None:
            return
        predictions, routing = self._collect_human3_samples(route_records)
        self._predictions_by_epoch[epoch] = predictions
        self._routing_by_epoch[epoch] = routing
        if representation_records is not None:
            self._representation_by_epoch[epoch] = self._collect_human3_representation(
                representation_records
            )

    def write_best_artifacts(self, best_epoch):
        """Write best-validation per-sample CSVs and source frequencies."""

        self.best_epoch = best_epoch
        predictions = self._predictions_by_epoch.get(best_epoch, [])
        routing = self._routing_by_epoch.get(best_epoch, [])
        if predictions:
            _write_csv(self.output_dir / "best_validation_human3_predictions.csv", PREDICTION_FIELDS, predictions)
        if routing:
            routing_fields = list(PREDICTION_FIELDS[:3]) + [
                "null_weight",
                "transfer_mass",
            ]
            routing_fields += [
                f"top{rank}_{suffix}"
                for rank in range(1, TOP_K_SOURCES + 1)
                for suffix in ("source", "conditional_weight", "joint_weight")
            ]
            routing_fields += ["routing_entropy"]
            _write_csv(self.output_dir / "best_validation_human3_routing.csv", routing_fields, routing)
            self._write_source_frequency(routing)
        representation = self._representation_by_epoch.get(best_epoch, [])
        if representation:
            summary_rows = []
            for task in self.human_tasks:
                rows = [row for row in representation if row["task"] == task]
                if not rows:
                    continue
                summary = {"epoch": best_epoch, "task": task, "n": len(rows)}
                summary.update(
                    {
                        f"{field}_mean": RunDiagnosticsWriter._finite_mean_values(
                            [row[field] for row in rows]
                        )
                        for field in REPRESENTATION_METRIC_FIELDS
                    }
                )
                # Plan §11: route-minus-base prediction magnitude alongside the
                # representation chain, joined from the best-epoch predictions.
                prediction_rows = self._predictions_by_epoch.get(best_epoch, [])
                deltas = [
                    abs(float(pred["route_prediction"]) - float(pred["base_prediction"]))
                    for pred in prediction_rows
                    if pred["task"] == task
                    and math.isfinite(float(pred["route_prediction"]))
                    and math.isfinite(float(pred["base_prediction"]))
                ]
                summary["route_minus_base_abs_mean"] = RunDiagnosticsWriter._finite_mean_values(deltas)
                summary_rows.append(summary)
            if summary_rows:
                _write_csv(
                    self.output_dir / "representation_path_summary.csv",
                    ("epoch", "task", "n")
                    + tuple(f"{field}_mean" for field in REPRESENTATION_METRIC_FIELDS)
                    + ("route_minus_base_abs_mean",),
                    summary_rows,
                )

    @staticmethod
    def _finite_mean_values(values):
        finite = [value for value in values if math.isfinite(value)]
        return sum(finite) / len(finite) if finite else float("nan")

    def _write_source_frequency(self, routing_rows):
        frequency = {}
        for task in self.human_tasks:
            rows = [row for row in routing_rows if row["task"] == task]
            if not rows:
                continue
            top1_counts = Counter(row["top1_source"] for row in rows if row["top1_source"])
            top3_counts = Counter(
                row[f"top{rank}_source"]
                for row in rows
                for rank in range(1, TOP_K_SOURCES + 1)
                if row[f"top{rank}_source"] and rank <= 3
            )
            frequency[task] = {
                "top1_source_counts": dict(top1_counts.most_common()),
                "top3_source_counts": dict(top3_counts.most_common()),
                "unique_top1_sources": len(top1_counts),
                "samples": len(rows),
            }
        payload = {"best_epoch": self.best_epoch, "tasks": frequency}
        (self.output_dir / "routing_source_frequency.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_current_files(self):
        _write_csv(self.output_dir / "epoch_summary.csv", EPOCH_SUMMARY_FIELDS, self.epoch_rows)
        path_fields = ("epoch", "task", "path") + PATH_METRIC_FIELDS
        _write_csv(self.output_dir / "human3_path_metrics.csv", path_fields, self.path_rows)
        routing_fields = ("epoch", "task") + ROUTING_FIELDS
        _write_csv(self.output_dir / "routing_epoch.csv", routing_fields, self.routing_rows)
        gradient_fields = ("epoch", "group", "mean", "max", "last", "steps")
        _write_csv(self.output_dir / "gradient_norms.csv", gradient_fields, self.gradient_rows)
        representation_fields = ("epoch", "task") + REPRESENTATION_AGGREGATE_FIELDS
        _write_csv(self.output_dir / "representation_epoch.csv", representation_fields, self.representation_rows)
