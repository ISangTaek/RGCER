"""Training, calibration, and evaluation for the three Graphormer variants."""

from __future__ import annotations

import copy
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from conformal import ConformalCalibrator
from architecture.prediction_heads import decode_prediction
from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS, HUMAN_TARGET_TASKS
from loss import QuantileRegressionLoss
from metric import (
    compute_classification_metrics,
    compute_interval_metrics,
    compute_regression_metrics,
)
from molecular_features import FEATURE_SCHEMA_VERSION
from split_manifest import load_manifest, manifest_hash
from utils import count_parameters


SOURCE_POLICY_CHOICES = ("all_except_target", "animal56_only")
DEFAULT_SOURCE_POLICY = "animal56_only"


def build_source_policy_mask(task_names, *, policy, allowed_auxiliary=()):
    """Return the router-source allow-list implied by the source policy.

    ``animal56_only`` keeps the animal→human claim honest: for a human
    target only animal endpoints may act as sources.  The router still
    excludes the target itself; shuffled auxiliary stress endpoints stay
    usable only when they are explicitly allowed by name.
    """

    if policy not in SOURCE_POLICY_CHOICES:
        raise ValueError(f"Unknown rgcer_source_policy: {policy!r}")
    allowed_auxiliary = [str(name) for name in (allowed_auxiliary or [])]
    unknown = [name for name in allowed_auxiliary if name not in task_names]
    if unknown:
        raise ValueError(
            f"explicitly_allowed_auxiliary_sources references tasks outside this run: {unknown}"
        )
    mask = []
    for name in task_names:
        if policy == "all_except_target":
            mask.append(True)
        else:
            mask.append((name in ANIMAL_SOURCE_TASKS) or (name in set(allowed_auxiliary)))
    return mask


class Trainer:
    """Task-wise trainer with one use of every available task batch per epoch."""

    def __init__(
        self,
        task_dict,
        weighting,
        architecture,
        encoder_class,
        decoders,
        optim_param,
        args,
        save_path=None,
        load_path=None,
        **kwargs,
    ):
        self.args = args
        self.task_dict = task_dict
        self.task_name = list(task_dict)
        self.task_num = len(self.task_name)
        self.save_path = Path(save_path) if save_path else None
        self.load_path = load_path
        self.kwargs = kwargs
        self.seed = int(getattr(args, "seed", 42))
        self.selection_scope = str(getattr(args, "selection_scope", "human3"))
        self.source_policy = str(getattr(args, "rgcer_source_policy", DEFAULT_SOURCE_POLICY))
        if self.source_policy not in SOURCE_POLICY_CHOICES:
            raise ValueError(f"Unknown rgcer_source_policy: {self.source_policy!r}")
        self.allowed_auxiliary_sources = tuple(
            str(name)
            for name in (getattr(args, "explicitly_allowed_auxiliary_sources", None) or ())
        )
        build_source_policy_mask(
            self.task_name,
            policy=self.source_policy,
            allowed_auxiliary=self.allowed_auxiliary_sources,
        )
        self._source_mask_by_device: dict[torch.device, torch.Tensor] = {}
        self._set_seed(self.seed)
        self.device = self._resolve_device(args)
        if architecture is None or encoder_class is None:
            raise ValueError("A supported architecture and encoder class are required")
        self.model = architecture(
            self.task_name,
            encoder_class,
            decoders,
            self.device,
            args,
            **kwargs.get("arch_args", {}),
        ).to(self.device)
        self.loss_balancer = weighting()
        self.loss_balancer.task_num = self.task_num
        self.loss_balancer.task_name = self.task_name
        self.loss_balancer.device = self.device
        self.loss_balancer.init_param()
        self.loss_balancer = self.loss_balancer.to(self.device)
        self.optimizer = self._make_optimizer(optim_param)
        self.task_scalers = {}
        self.conformal_calibrator = ConformalCalibrator(
            alpha=getattr(args, "conformal_alpha", 0.10),
            min_calibration_size=getattr(args, "min_calibration_size", 30),
        )
        # Keep the loss object available to unit-level training calls as well as
        # to the full train() lifecycle.  lambda_quantile belongs to the final
        # loss combination, so the quantile components themselves are not
        # scaled a second time inside QuantileRegressionLoss.
        self.quantile_loss = QuantileRegressionLoss(
            lower_quantile=getattr(args, "lower_quantile", 0.05),
            upper_quantile=getattr(args, "upper_quantile", 0.95),
            quantile_weight=1.0,
        )
        self.best_val_score = -float("inf")
        self.best_checkpoint_path = None
        self._best_state = None
        self._best_training_state = None
        self.best_epoch = None
        self.loaded_epoch = None
        self.loaded_routing_enabled = None
        self.best_routing_enabled = None
        self.final_test_result = None
        self.schedule_usage = {}
        self.optimizer_updates = 0
        self.training_cache = {}
        self._is_rgcer = bool(getattr(self.model, "is_rgcer", False))
        self.data_metadata = getattr(args, "datastore_metadata", None)
        if self.data_metadata is None:
            data_store_dir = getattr(args, "data_store_dir", None)
            if data_store_dir:
                try:
                    from toxacute_datastore import ToxAcuteDataStore

                    store = ToxAcuteDataStore.resolve(data_store_dir)
                    self.data_metadata = store.metadata
                    store.close()
                except (FileNotFoundError, ValueError):
                    # Non-ToxAcute unit callers may retain a placeholder path.
                    self.data_metadata = None
        self.max_path_distance = int(
            (self.data_metadata or {}).get("max_path_distance", getattr(args, "max_path_distance", 8) or 8)
        )
        if load_path is not None:
            self.load_checkpoint(load_path)
        elif getattr(args, "mode", "train") in {"test", "batch_inference", "single_inference"}:
            raise FileNotFoundError("test/inference mode requires --load_path to a strict checkpoint")
        count_parameters(self.model)

    @staticmethod
    def _resolve_device(args):
        if args.gpu_id != "cpu" and torch.cuda.is_available():
            return torch.device(f"cuda:{args.gpu_id}")
        return torch.device("cpu")

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_optimizer(self, optim_param):
        optim_name = str(optim_param.get("optim", "adamw")).lower()
        optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optim_name)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {optim_name}")
        parameters = list(self.model.parameters()) + list(self.loss_balancer.parameters())
        return optimizer_class(
            parameters,
            **{key: value for key, value in optim_param.items() if key != "optim"},
        )

    def _is_regression(self, task):
        return "RMSE" in self.task_dict[task].get("metrics", [])

    def _prediction_mode(self, task=None):
        """Return the mode that matches the task head and its loss.

        Classification heads are always point heads.  Keeping this decision in
        the trainer as well as in the CLI prevents a direct API caller from
        accidentally decoding a one-channel classification output as a
        quantile tensor.
        """

        if task is not None and not self._is_regression(task):
            return "point"
        requested = getattr(self.args, "prediction_mode", "point")
        if task is None and self.task_name and all(not self._is_regression(name) for name in self.task_name):
            return "point"
        return requested

    def _fit_task_scalers(self, train_dataloaders_dict):
        for task in self.task_name:
            if not self._is_regression(task):
                self.task_scalers[task] = {"mean": 0.0, "std": 1.0, "count": 0}
                continue
            values = []
            loader = train_dataloaders_dict.get(task)
            if loader is not None:
                for batch in loader:
                    if batch.get("is_empty", False):
                        continue
                    values.append(batch.y.detach().float().reshape(-1).cpu())
            if not values:
                self.task_scalers[task] = {"mean": 0.0, "std": 1.0, "count": 0}
                continue
            labels = torch.cat(values)
            self.task_scalers[task] = {
                "mean": float(labels.mean()),
                "std": max(float(labels.std(unbiased=False)), 1e-6),
                "count": int(labels.numel()),
            }

    def _normalize_target(self, task, labels):
        scaler = self.task_scalers.get(task, {"mean": 0.0, "std": 1.0})
        return (labels - scaler["mean"]) / scaler["std"] if self._is_regression(task) else labels

    def _denormalize_tensor(self, task, values):
        scaler = self.task_scalers.get(task, {"mean": 0.0, "std": 1.0})
        return values * scaler["std"] + scaler["mean"] if self._is_regression(task) else values

    def decode_task_output(self, task, raw, apply_conformal=True):
        mode = self._prediction_mode(task)
        decoded = decode_prediction(raw, mode=mode)
        result = {
            "median": self._denormalize_tensor(task, decoded.median),
            "lower": self._denormalize_tensor(task, decoded.lower),
            "upper": self._denormalize_tensor(task, decoded.upper),
        }
        if apply_conformal and mode == "quantile":
            if task in self.conformal_calibrator.states:
                result["lower"], result["upper"] = self.conformal_calibrator.apply(
                    task, result["lower"], result["upper"]
                )
        return result

    def _task_schedule(self, dataloaders_dict, epoch):
        schedule = []
        for task in self.task_name:
            loader = dataloaders_dict.get(task)
            if loader is not None:
                schedule.extend([task] * len(loader))
        rng = np.random.default_rng(self.seed + int(epoch))
        rng.shuffle(schedule)
        return list(schedule)

    def _loss(self, task, prediction, labels):
        loss_fn = self.task_dict[task]["loss_fn"]
        if hasattr(loss_fn, "compute_loss"):
            return loss_fn.compute_loss(prediction, labels)
        return loss_fn(prediction, labels)

    def _prediction_loss(self, task, raw, labels):
        if not self._is_regression(task):
            return self._loss(task, raw, labels)
        if self._prediction_mode(task) == "quantile":
            return self.quantile_loss.compute_loss(raw, labels)
        return self._loss(task, raw, labels)

    @staticmethod
    def _valid_batch(batch):
        return batch is not None and not batch.get("is_empty", False) and batch.y.numel() > 0

    @staticmethod
    def _diagnostics_dict(diagnostics):
        if diagnostics is None:
            return {}
        if hasattr(diagnostics, "as_dict"):
            return diagnostics.as_dict()
        if isinstance(diagnostics, dict):
            return diagnostics
        return vars(diagnostics)

    def _routing_enabled(self, epoch, routing_enabled_override=None):
        if not self._is_rgcer:
            return True
        if not bool(getattr(self.args, "routing_enabled", True)):
            return False
        if routing_enabled_override is not None:
            return bool(routing_enabled_override)
        # Validation/test callers may pass ``None`` because they are not part
        # of the epoch schedule.  Prefer the state recorded by a loaded
        # checkpoint; this also keeps warm-up checkpoints on the HPS path.
        if epoch is None:
            if self.loaded_routing_enabled is not None:
                return bool(self.loaded_routing_enabled)
            return True
        return epoch >= getattr(self.args, "hps_warmup_epochs", 0)

    def source_policy_mask(self, device=None):
        """Default router-source mask for the current task list and policy.

        Returns ``None`` unless this instance was fully constructed with a
        source policy: minimally wired fixtures keep the previous behaviour
        of passing no mask at all.
        """

        if not self._is_rgcer:
            return None
        if not getattr(self, "source_policy", None) or not getattr(self, "task_name", None):
            return None
        target_device = device if device is not None else getattr(
            self, "device", torch.device("cpu")
        )
        cached = self._source_mask_by_device.get(target_device)
        if cached is None:
            mask = build_source_policy_mask(
                self.task_name,
                policy=self.source_policy,
                allowed_auxiliary=self.allowed_auxiliary_sources,
            )
            cached = torch.tensor(mask, dtype=torch.bool, device=target_device)
            self._source_mask_by_device[target_device] = cached
        return cached

    def effective_source_mask(self, source_mask, device=None):
        """Explicit masks win; otherwise the configured policy applies."""

        if source_mask is not None:
            return source_mask
        return self.source_policy_mask(device)

    def _eligible_for_best(self, epoch: int) -> bool:
        """Only routed Full RGCER epochs may define its best checkpoint."""

        if not self._is_rgcer:
            return True
        if not bool(getattr(self.args, "routing_enabled", True)):
            return True
        return int(epoch) >= int(getattr(self.args, "hps_warmup_epochs", 0))

    def _forward_task(self, batch, task, epoch, return_aux=True, routing_enabled_override=None):
        if self._is_rgcer:
            result = self.model(
                batch,
                task_name=task,
                return_aux=return_aux,
                routing_enabled=self._routing_enabled(epoch, routing_enabled_override),
                source_mask=self.effective_source_mask(None),
            )
        else:
            result = self.model(batch, task_name=task, return_aux=return_aux)
        if return_aux and isinstance(result, tuple):
            return result[0], self._diagnostics_dict(result[1])
        return result, {}

    @torch.no_grad()
    def predict_all_tasks(self, batch, *, return_aux=False, source_mask=None):
        """Run inference with the configured source policy applied by default."""

        self.model.eval()
        default_mask = self.effective_source_mask(source_mask)
        if self._is_rgcer:
            return self.model(
                batch,
                return_all_tasks=True,
                return_aux=return_aux,
                source_mask=default_mask,
                routing_enabled=self._routing_enabled(None),
            )
        return self.model(
            batch,
            return_all_tasks=True,
            return_aux=return_aux,
            source_mask=default_mask,
        )

    def _training_step(self, batch, task, epoch):
        labels = batch.y.reshape(-1, 1).float()
        normalized_labels = self._normalize_target(task, labels)
        output, diagnostics = self._forward_task(batch, task, epoch, return_aux=True)
        final_raw = output[task]
        final_loss = self._prediction_loss(task, final_raw, normalized_labels)
        base_raw = diagnostics.get("base_raw", final_raw)
        routing_enabled = self._routing_enabled(epoch)
        total_loss = getattr(self.args, "lambda_quantile", 1.0) * final_loss
        if routing_enabled and self._is_rgcer:
            base_loss = self._prediction_loss(task, base_raw, normalized_labels)
            if getattr(self.args, "rgcer_use_base_aux_loss", True):
                total_loss = total_loss + getattr(self.args, "lambda_base", 0.25) * base_loss
        else:
            base_loss = final_loss
        return total_loss, {
            "task": task,
            "labels": labels.detach(),
            "final_raw": final_raw,
            "base_raw": base_raw,
            "route_raw": diagnostics.get("route_raw", final_raw),
            "diagnostics": diagnostics,
            "loss": total_loss.detach(),
            "base_loss": base_loss.detach(),
        }

    def _record_training_output(self, bundle):
        task = bundle["task"]
        final = self.decode_task_output(task, bundle["final_raw"].detach(), apply_conformal=False)["median"].cpu()
        base = self.decode_task_output(task, bundle["base_raw"].detach(), apply_conformal=False)["median"].cpu()
        route = self.decode_task_output(task, bundle["route_raw"].detach(), apply_conformal=False)["median"].cpu()
        diagnostics = bundle["diagnostics"]
        self.training_cache.setdefault(
            task,
            {
                "base": [],
                "route": [],
                "final": [],
                "target": [],
                "route_regret": [],
                "final_regret": [],
                "null": [],
                "source_weights": [],
                "entropy": [],
            },
        )
        cache = self.training_cache[task]
        target = bundle["labels"].cpu()
        route_regret = (route - target).abs() - (base - target).abs()
        final_regret = (final - target).abs() - (base - target).abs()
        cache["base"].append(base.cpu())
        cache["route"].append(route.cpu())
        cache["final"].append(final.cpu())
        cache["target"].append(target)
        cache["route_regret"].append(route_regret.cpu())
        cache["final_regret"].append(final_regret.cpu())
        if "null_weight" in diagnostics:
            cache["null"].append(diagnostics["null_weight"].detach().cpu())
        if "source_weights" in diagnostics:
            cache["source_weights"].append(diagnostics["source_weights"].detach().cpu())
        if "routing_entropy" in diagnostics:
            cache["entropy"].append(diagnostics["routing_entropy"].detach().cpu())

    def _routing_summary(self):
        summary = {}
        for task, cache in self.training_cache.items():
            if not cache["target"]:
                continue
            result = {
                "route_regret_mean": float(torch.cat(cache["route_regret"]).mean()),
                "final_regret_mean": float(torch.cat(cache["final_regret"]).mean()),
                "route_negative_transfer_rate": float(
                    (torch.cat(cache["route_regret"]) > 0).float().mean()
                ),
                "final_negative_transfer_rate": float(
                    (torch.cat(cache["final_regret"]) > 0).float().mean()
                ),
            }
            result["negative_transfer_rate"] = result["route_negative_transfer_rate"]
            if cache["null"]:
                null = torch.cat(cache["null"])
                result.update(
                    {
                        "mean_null_weight": float(null.mean()),
                        "mean_null": float(null.mean()),
                        "std_null": float(null.std(unbiased=False)),
                    }
                )
            if cache["source_weights"]:
                weights = torch.cat(cache["source_weights"])
                result.update(
                    {
                        "mean_active_sources": float((weights > 0).sum(dim=-1).float().mean()),
                        "mean_active_routes": float((weights > 0).sum(dim=-1).float().mean()),
                        "routing_variance": float(weights.var(dim=0, unbiased=False).mean()),
                    }
                )
            if cache["entropy"]:
                result["mean_routing_entropy"] = float(torch.cat(cache["entropy"]).mean())
            summary[task] = result
        return summary

    def _train_epoch(self, train_dataloaders_dict, epoch):
        self.model.train()
        self.loss_balancer.train()
        schedule = self._task_schedule(train_dataloaders_dict, epoch)
        self.schedule_usage = {task: 0 for task in self.task_name}
        self.training_cache = {}
        iterators = {
            task: iter(loader)
            for task, loader in train_dataloaders_dict.items()
            if loader is not None and len(loader) > 0
        }
        buffers = {task: {"pred": [], "label": []} for task in self.task_name}
        losses = {task: [] for task in self.task_name}
        self.loss_balancer.epoch = epoch
        max_history = max(epoch + 1, 2)
        self.loss_balancer.train_loss_buffer = getattr(
            self, "train_loss_buffer", np.ones((self.task_num, max_history))
        )
        tasks_per_update = int(getattr(self.args, "tasks_per_update", 1))
        if tasks_per_update <= 0:
            raise ValueError("tasks_per_update must be positive")
        update_count = 0
        for start in range(0, len(schedule), tasks_per_update):
            group = schedule[start : start + tasks_per_update]
            task_loss_groups = {task: [] for task in self.task_name}
            active_mask = torch.zeros(self.task_num, dtype=torch.bool, device=self.device)
            bundles = []
            for task in group:
                batch = next(iterators[task], None)
                if not self._valid_batch(batch):
                    continue
                batch = batch.to(self.device)
                loss, bundle = self._training_step(batch, task, epoch)
                # A shuffled schedule can place two batches of the same task
                # in one optimizer group.  Preserve both forwards and
                # aggregate their losses instead of overwriting one entry in
                # the task loss vector.
                task_loss_groups[task].append(loss)
                bundles.append(bundle)
                active_mask[self.task_name.index(task)] = True
                self.schedule_usage[task] += 1
            if not bundles:
                continue
            self.optimizer.zero_grad(set_to_none=True)
            loss_vector = torch.zeros(self.task_num, device=self.device)
            for task, task_losses in task_loss_groups.items():
                if task_losses:
                    loss_vector[self.task_name.index(task)] = torch.stack(task_losses).mean()
            self.loss_balancer.backward(loss_vector, active_mask=active_mask)
            clip_value = getattr(self.args, "grad_clip", 1.0)
            if clip_value is not None and float(clip_value) > 0:
                clip_grad_norm_(list(self.model.parameters()) + list(self.loss_balancer.parameters()), float(clip_value))
            self.optimizer.step()
            update_count += 1
            for bundle in bundles:
                task = bundle["task"]
                final_median = self.decode_task_output(task, bundle["final_raw"].detach(), apply_conformal=False)["median"]
                buffers[task]["pred"].append(final_median.cpu())
                buffers[task]["label"].append(bundle["labels"].cpu())
                losses[task].append(float(bundle["loss"].cpu()))
                self._record_training_output(bundle)
        result = self._score_buffers(buffers)
        result["loss"] = {task: float(np.mean(losses[task])) if losses[task] else np.nan for task in self.task_name}
        self.optimizer_updates += update_count
        result["updates"] = update_count
        result["schedule_usage"] = dict(self.schedule_usage)
        result["routing"] = self._routing_summary()
        return result

    def _selection_tasks(self):
        """Return ``(tasks, effective_scope)`` used for best-checkpoint selection.

        ``human3`` selects the three human target endpoints, which are the
        endpoints the model is meant to serve.  When the current run contains
        none of them (e.g. an animal56 task scope or a non-ToxAcute dataset)
        the selection falls back to every task of the run.
        """
        if self.selection_scope == "human3":
            selected = [task for task in self.task_name if task in HUMAN_TARGET_TASKS]
            if selected:
                return selected, "human3"
        return list(self.task_name), "all_tasks"

    def _score_buffers(self, buffers):
        task_scores = {}
        primary_values = []
        regression_values = []
        human3_values = []
        selection_tasks, effective_scope = self._selection_tasks()
        selection_values = []
        for task in self.task_name:
            pred = buffers[task]["pred"]
            labels = buffers[task]["label"]
            if not pred:
                task_scores[task] = {name: np.nan for name in self.task_dict[task].get("metrics", [])}
                continue
            pred_array = torch.cat(pred).numpy()
            label_array = torch.cat(labels).numpy()
            values = compute_regression_metrics(pred_array, label_array) if self._is_regression(task) else compute_classification_metrics(pred_array, label_array)
            task_scores[task] = {name: values.get(name, np.nan) for name in self.task_dict[task]["metrics"]}
            primary = task_scores[task][self.task_dict[task]["metrics"][0]]
            weighted = self.task_dict[task]["weight"][0] * primary
            if self.task_dict[task].get("include_in_macro", True) and np.isfinite(primary):
                primary_values.append(weighted)
                if self._is_regression(task):
                    regression_values.append(primary)
            if task in HUMAN_TARGET_TASKS and np.isfinite(primary):
                human3_values.append(primary)
            if task in selection_tasks and np.isfinite(primary):
                selection_values.append(weighted)
        return {
            "tasks": task_scores,
            "score": float(np.mean(primary_values)) if primary_values else -float("inf"),
            "selection_score": float(np.mean(selection_values)) if selection_values else -float("inf"),
            "selection_scope": effective_scope,
            "selection_tasks": list(selection_tasks),
            "human3_macro_rmse": float(np.mean(human3_values)) if human3_values else np.nan,
            "all_task_macro_rmse": float(np.mean(regression_values)) if regression_values else np.nan,
        }

    @staticmethod
    def _print_metrics(mode, epoch, result):
        prefix = mode if epoch is None else f"{mode} epoch={epoch:03d}"
        pieces = [f"{prefix}: score={result['score']:.6f}"]
        if "selection_score" in result:
            pieces.append(
                f"selection[{result.get('selection_scope', '<unset>')}]={result['selection_score']:.6f}"
            )
            for key in ("human3_macro_rmse", "all_task_macro_rmse"):
                value = result.get(key, np.nan)
                if np.isfinite(value):
                    pieces.append(f"{key}={value:.6f}")
        for task, values in result["tasks"].items():
            pieces.append(f"{task}[{', '.join(f'{key}={value:.6f}' for key, value in values.items())}]")
        print(" | ".join(pieces))
        if result.get("schedule_usage"):
            print(f"schedule_usage: {result['schedule_usage']}")
        if result.get("routing"):
            print(f"routing: {result['routing']}")

    def _collect_predictions(
        self,
        dataloaders_dict,
        epoch=0,
        apply_conformal=False,
        routing_enabled_override=None,
    ):
        self.model.eval()
        buffers = {task: {"pred": [], "label": []} for task in self.task_name}
        records = {task: {"lower": [], "upper": [], "target": []} for task in self.task_name}
        route_records = {
            task: {
                "base": [],
                "route": [],
                "final": [],
                "target": [],
                "null": [],
                "source_weights": [],
                "route_regret": [],
                "final_regret": [],
                "entropy": [],
            }
            for task in self.task_name
        }
        with torch.no_grad():
            for task in self.task_name:
                loader = dataloaders_dict.get(task)
                if loader is None:
                    continue
                for batch in loader:
                    if not self._valid_batch(batch):
                        continue
                    batch = batch.to(self.device)
                    output, diagnostics = self._forward_task(
                        batch,
                        task,
                        epoch,
                        return_aux=True,
                        routing_enabled_override=routing_enabled_override,
                    )
                    final_raw = output[task]
                    final_decoded = self.decode_task_output(task, final_raw, apply_conformal=apply_conformal)
                    target = batch.y.reshape(-1, 1).float()
                    buffers[task]["pred"].append(final_decoded["median"].cpu())
                    buffers[task]["label"].append(target.cpu())
                    records[task]["lower"].append(self.decode_task_output(task, final_raw, apply_conformal=False)["lower"].cpu())
                    records[task]["upper"].append(self.decode_task_output(task, final_raw, apply_conformal=False)["upper"].cpu())
                    records[task]["target"].append(target.cpu())
                    base = self.decode_task_output(task, diagnostics.get("base_raw", final_raw), apply_conformal=False)["median"].cpu()
                    route = self.decode_task_output(task, diagnostics.get("route_raw", final_raw), apply_conformal=False)["median"].cpu()
                    final = final_decoded["median"].cpu()
                    rr = (route - target.cpu()).abs() - (base - target.cpu()).abs()
                    fr = (final - target.cpu()).abs() - (base - target.cpu()).abs()
                    route_records[task]["base"].append(base)
                    route_records[task]["route"].append(route)
                    route_records[task]["final"].append(final)
                    route_records[task]["target"].append(target.cpu())
                    route_records[task]["route_regret"].append(rr)
                    route_records[task]["final_regret"].append(fr)
                    if "null_weight" in diagnostics:
                        route_records[task]["null"].append(diagnostics["null_weight"].cpu())
                    if "source_weights" in diagnostics:
                        route_records[task]["source_weights"].append(diagnostics["source_weights"].cpu())
                    if "routing_entropy" in diagnostics:
                        route_records[task]["entropy"].append(diagnostics["routing_entropy"].cpu())
        return buffers, records, route_records

    def _evaluate(self, dataloaders_dict, mode="validation", epoch=0, routing_enabled_override=None):
        buffers, records, route_records = self._collect_predictions(
            dataloaders_dict,
            epoch=epoch,
            apply_conformal=mode == "test",
            routing_enabled_override=routing_enabled_override,
        )
        result = self._score_buffers(buffers)
        if self._prediction_mode() == "quantile":
            interval_values = []
            for task in self.task_name:
                if not self._is_regression(task) or not records[task]["target"]:
                    continue
                # Auxiliary stress-test endpoints may be used as router
                # sources, but they must not affect the formal CQR report.
                if not self.task_dict[task].get("fit_conformal", True):
                    continue
                lower = torch.cat(records[task]["lower"])
                upper = torch.cat(records[task]["upper"])
                target = torch.cat(records[task]["target"])
                if mode == "test" and task in self.conformal_calibrator.states:
                    lower, upper = self.conformal_calibrator.apply(task, lower, upper)
                interval_values.append(compute_interval_metrics(lower, upper, target, self.args.conformal_alpha))
            if interval_values:
                result["interval"] = {
                    key: float(np.nanmean([value[key] for value in interval_values])) for key in interval_values[0]
                }
        result["routing"] = self._evaluation_routing_summary(route_records)
        self._print_metrics(mode, epoch if mode == "validation" else None, result)
        return result

    @staticmethod
    def _rankdata(values):
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=float)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return ranks

    @classmethod
    def _safe_auroc(cls, scores, labels):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=bool).reshape(-1)
        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        if positives == 0 or negatives == 0:
            return float("nan")
        ranks = cls._rankdata(scores)
        positive_rank_sum = float(ranks[labels].sum())
        return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    @staticmethod
    def _safe_auprc(scores, labels):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=bool).reshape(-1)
        positives = int(labels.sum())
        if positives == 0 or positives == labels.size:
            return float("nan") if positives == 0 else 1.0
        order = np.argsort(-scores, kind="mergesort")
        ordered_labels = labels[order].astype(float)
        precision = np.cumsum(ordered_labels) / np.arange(1, labels.size + 1)
        return float((precision * ordered_labels).sum() / positives)

    @classmethod
    def _safe_spearman(cls, left, right):
        left_rank = cls._rankdata(left)
        right_rank = cls._rankdata(right)
        if left_rank.size < 2:
            return float("nan")
        left_centered = left_rank - left_rank.mean()
        right_centered = right_rank - right_rank.mean()
        denominator = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
        if denominator == 0:
            return float("nan")
        return float(np.sum(left_centered * right_centered) / denominator)

    @classmethod
    def _evaluation_routing_summary(cls, route_records):
        summary = {}
        for task, record in route_records.items():
            if not record["target"]:
                continue
            regret = torch.cat(record["route_regret"])
            final_regret = torch.cat(record.get("final_regret", record["route_regret"]))
            values = {
                "route_regret_mean": float(regret.mean()),
                "final_regret_mean": float(final_regret.mean()),
                "route_negative_transfer_rate": float((regret > 0).float().mean()),
                "final_negative_transfer_rate": float((final_regret > 0).float().mean()),
            }
            values["negative_transfer_rate"] = values["route_negative_transfer_rate"]
            if record["null"]:
                null = torch.cat(record["null"])
                values["mean_null_weight"] = float(null.mean())
                values["mean_null"] = float(null.mean())
                values["std_null"] = float(null.std(unbiased=False))
                null_values = null.detach().cpu().numpy().reshape(-1)
                route_regret_values = regret.detach().cpu().numpy().reshape(-1)
                harmful_route = route_regret_values > 0
                values["null_harmful_route_auroc"] = cls._safe_auroc(null_values, harmful_route)
                values["null_harmful_route_auprc"] = cls._safe_auprc(null_values, harmful_route)
                values["null_weight_route_regret_spearman"] = cls._safe_spearman(
                    null_values, route_regret_values
                )
            if record["source_weights"]:
                weights = torch.cat(record["source_weights"])
                values["mean_active_sources"] = float((weights > 0).sum(dim=-1).float().mean())
                values["mean_active_routes"] = float((weights > 0).sum(dim=-1).float().mean())
                values["routing_variance"] = float(weights.var(dim=0, unbiased=False).mean())
            if record.get("entropy"):
                values["mean_routing_entropy"] = float(torch.cat(record["entropy"]).mean())
            summary[task] = values
        return summary

    @staticmethod
    def aggregate_routing_summary(summary):
        """Aggregate per-task routing diagnostics for a compact run artifact."""

        keys = (
            "mean_null_weight",
            "mean_routing_entropy",
            "mean_active_sources",
            "route_regret_mean",
            "final_regret_mean",
            "route_negative_transfer_rate",
            "final_negative_transfer_rate",
        )
        aggregate = {}
        for key in keys:
            values = [float(record[key]) for record in summary.values() if key in record]
            aggregate[key] = float(np.nanmean(values)) if values else float("nan")
        return aggregate

    def _fit_conformal(self, calibration_dataloaders_dict, routing_enabled_override=None):
        if self._prediction_mode() != "quantile" or not getattr(self.args, "fit_conformal", True):
            return
        _, records, _ = self._collect_predictions(
            calibration_dataloaders_dict,
            epoch=10**9,
            apply_conformal=False,
            routing_enabled_override=routing_enabled_override,
        )
        splitting = getattr(self.args, "splitting", None)
        if splitting == "scaffold":
            warnings.warn(
                "Standard split-conformal finite-sample coverage assumes exchangeability; "
                "scaffold-split coverage is reported empirically under structural shift.",
                RuntimeWarning,
                stacklevel=2,
            )
        for task in self.task_name:
            if not self._is_regression(task):
                continue
            if not self.task_dict[task].get("fit_conformal", True):
                continue
            if not records[task]["target"]:
                raise ValueError(f"No calibration samples for {task}.")
            self.conformal_calibrator.fit_task(
                task,
                torch.cat(records[task]["lower"]),
                torch.cat(records[task]["upper"]),
                torch.cat(records[task]["target"]),
            )

    def _conformal_validity(self):
        """Coverage-claim metadata: method, alpha, and split-type assumptions."""

        from conformal import exchangeability_metadata

        return {
            "conformal_method": "taskwise_cqr" if self._prediction_mode() == "quantile" else None,
            "conformal_alpha": float(getattr(self.args, "conformal_alpha", 0.10)),
            **exchangeability_metadata(getattr(self.args, "splitting", None)),
        }

    def _manifest_hash(self):
        if self.data_metadata is not None:
            return self.data_metadata.get("split_manifest_hash")
        directory = getattr(self.args, "preprocessed_data_dir", None)
        if not directory:
            return None
        path = Path(directory) / "split_manifest.json"
        return manifest_hash(load_manifest(path)) if path.exists() else None

    def _architecture_config(self):
        prompt_bank = getattr(getattr(getattr(self.model, "encoder", None), "task_conditioner", None), "prompt_bank", None)
        requested_factorized = getattr(self.args, "use_factorized_prompt", None)
        factorized = bool(prompt_bank.use_factorized_prompt) if prompt_bank is not None else requested_factorized
        return {
            "architecture": self.model.__class__.__name__,
            "task_names": list(self.task_name),
            "prediction_mode": self._prediction_mode(),
            "edge_bias_mode": getattr(self.args, "edge_bias_mode", "path"),
            "spatial_pos_max_clip": getattr(
                self.args, "spatial_pos_clip", getattr(self.args, "spatial_pos_max_clip", 20)
            ),
            "router_top_k": getattr(self.args, "router_top_k", 0),
            "router_temperature": getattr(self.args, "router_temperature", 1.0),
            "exclude_target_from_sources": getattr(self.args, "exclude_target_from_sources", True),
            "use_factorized_prompt": factorized,
            "task_metadata": [
                getattr(item, "__dict__", {})
                for item in getattr(prompt_bank, "metadata", [])
            ],
        }

    def _rgcer_config(self):
        architecture_config = self._architecture_config()
        resolved = getattr(self.args, "effective_rgcer_config", None)
        effective = resolved.get("effective", {}) if isinstance(resolved, dict) else {}

        def _value(name, default):
            return effective.get(name, getattr(self.args, name, default))

        return {
            "rgcer_use_source_response": _value("rgcer_use_source_response", True),
            "rgcer_use_target_response": _value("rgcer_use_target_response", True),
            "rgcer_use_molecule_query": _value("rgcer_use_molecule_query", True),
            "rgcer_use_sparse_routing": _value("rgcer_use_sparse_routing", True),
            "rgcer_use_null_route": _value("rgcer_use_null_route", True),
            "rgcer_use_film": _value("rgcer_use_film", True),
            "rgcer_use_adapter": _value("rgcer_use_adapter", True),
            "rgcer_use_base_aux_loss": _value("rgcer_use_base_aux_loss", True),
            "rgcer_fallback_space": _value("rgcer_fallback_space", "prediction"),
            "rgcer_transfer_mechanism": _value("rgcer_transfer_mechanism", "endpoint_router"),
            "router_top_k": _value("router_top_k", 0),
            "router_temperature": _value("router_temperature", 1.0),
            "rgcer_source_policy": self.source_policy,
            "allowed_auxiliary_sources": list(self.allowed_auxiliary_sources),
            "exclude_target_from_sources": getattr(self.args, "exclude_target_from_sources", True),
            "use_factorized_prompt": architecture_config["use_factorized_prompt"],
            "prediction_mode": self._prediction_mode(),
            "fit_conformal": getattr(self.args, "fit_conformal", True),
            "lambda_base": getattr(self.args, "lambda_base", 0.0),
            "lambda_quantile": getattr(self.args, "lambda_quantile", 1.0),
            "routing_enabled": getattr(self.args, "routing_enabled", True),
            "effective": getattr(self.args, "effective_rgcer_config", None),
        }

    def _data_config(self):
        metadata = self.data_metadata or {}
        return {
            "datastore_format_version": metadata.get("format_version"),
            "datastore_build_id": metadata.get("build_id"),
            "datastore_fingerprint": metadata.get("datastore_fingerprint"),
            "raw_csv_sha256": metadata.get("raw_csv_sha256"),
            "feature_schema_version": metadata.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            "max_path_distance": int(metadata.get("max_path_distance", self.max_path_distance)),
            "splitting": metadata.get("splitting", getattr(self.args, "splitting", None)),
            "split_seed": metadata.get("split_seed", getattr(self.args, "split_seed", None)),
            "split_ratios": metadata.get(
                "split_ratios",
                {
                    "train": 1.0
                    - getattr(self.args, "vs", 0.1)
                    - getattr(self.args, "calibration_size", 0.1)
                    - getattr(self.args, "ts", 0.1),
                    "validation": getattr(self.args, "vs", 0.1),
                    "calibration": getattr(self.args, "calibration_size", 0.1),
                    "test": getattr(self.args, "ts", 0.1),
                },
            ),
            "split_manifest_hash": metadata.get("split_manifest_hash", self._manifest_hash()),
            "max_nodes_filter": getattr(self.args, "max_nodes_filter", None),
            "task_names": list(self.task_name),
        }

    def _checkpoint_payload(self, epoch):
        checkpoint_version = 5 if self.data_metadata is not None else 4
        return {
            "checkpoint_version": checkpoint_version,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "weighting_state": self.loss_balancer.state_dict(),
            "epoch": int(epoch),
            "configuration": vars(self.args) if hasattr(self.args, "__dict__") else {},
            "task_names": list(self.task_name),
            "architecture_config": self._architecture_config(),
            "prediction_mode": self._prediction_mode(),
            "quantile_config": {
                "lower": getattr(self.args, "lower_quantile", 0.05),
                "upper": getattr(self.args, "upper_quantile", 0.95),
            },
            "conformal_state": self.conformal_calibrator.state_dict(),
            "conformal_validity": self._conformal_validity(),
            "hps_warmup_epochs": getattr(self.args, "hps_warmup_epochs", 0),
            "routing_enabled": self._routing_enabled(epoch),
            "rgcer_config": self._rgcer_config(),
            "effective_rgcer_config": getattr(self.args, "effective_rgcer_config", None),
            "task_metadata": self._architecture_config().get("task_metadata", []),
            "task_scalers": self.task_scalers,
            "split_manifest_hash": self._manifest_hash(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "data_config": self._data_config(),
            # These fields are optional when loading an older v3 checkpoint;
            # new checkpoints retain enough optimizer history for a faithful
            # resume.
            "train_loss_buffer": getattr(self, "train_loss_buffer", None),
            "optimizer_updates": int(getattr(self, "optimizer_updates", 0)),
        }

    def _save_checkpoint(self, epoch, filename):
        if self.save_path is None:
            return None
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self.save_path / filename
        torch.save(self._checkpoint_payload(epoch), path)
        return path

    def load_checkpoint(self, path):
        path = Path(path)
        if path.is_dir():
            path = path / f"{getattr(self.args, 'ckpt_name', 'model')}_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        formal_v2 = self.data_metadata is not None
        expected_version = 5 if formal_v2 else 4
        required = {
            "checkpoint_version", "model_state", "optimizer_state", "weighting_state", "epoch",
            "configuration", "task_names", "architecture_config", "prediction_mode", "quantile_config",
            "conformal_state", "hps_warmup_epochs", "rgcer_config", "task_metadata", "task_scalers",
            "split_manifest_hash", "feature_schema_version",
        }
        if formal_v2:
            required.update({"data_config", "effective_rgcer_config", "routing_enabled"})
        missing = required.difference(checkpoint)
        if missing or checkpoint["checkpoint_version"] != expected_version:
            raise ValueError(
                f"Checkpoint is not a strict v{expected_version} checkpoint; missing={sorted(missing)}"
            )
        if list(checkpoint["task_names"]) != self.task_name:
            raise ValueError("Checkpoint task_names do not match the current experiment")
        if checkpoint["prediction_mode"] != self._prediction_mode():
            raise ValueError("Checkpoint prediction mode does not match current configuration")
        stored_quantiles = checkpoint["quantile_config"]
        if not isinstance(stored_quantiles, dict) or any(
            name not in stored_quantiles for name in ("lower", "upper")
        ):
            raise ValueError("Checkpoint quantile_config is incomplete")
        current_quantiles = {
            "lower": getattr(self.args, "lower_quantile", 0.05),
            "upper": getattr(self.args, "upper_quantile", 0.95),
        }
        for name in ("lower", "upper"):
            if float(stored_quantiles.get(name)) != float(current_quantiles[name]):
                raise ValueError(f"Checkpoint quantile setting {name!r} does not match current configuration")
        stored_architecture = checkpoint["architecture_config"]
        if not isinstance(stored_architecture, dict):
            raise ValueError("Checkpoint architecture_config is invalid")
        current_config = self._architecture_config()
        for name in (
            "architecture",
            "prediction_mode",
            "edge_bias_mode",
            "router_top_k",
            "router_temperature",
            "exclude_target_from_sources",
            "use_factorized_prompt",
            "spatial_pos_max_clip",
            "task_metadata",
        ):
            if stored_architecture.get(name) != current_config.get(name):
                raise ValueError(f"Checkpoint setting {name!r} does not match current configuration")
        if int(checkpoint["hps_warmup_epochs"]) != int(getattr(self.args, "hps_warmup_epochs", 0)):
            raise ValueError("Checkpoint hps_warmup_epochs does not match current configuration")
        current_rgcer = self._rgcer_config()
        stored_rgcer = checkpoint["rgcer_config"]
        if not isinstance(stored_rgcer, dict):
            raise ValueError("Checkpoint rgcer_config is invalid")
        for name, value in current_rgcer.items():
            if stored_rgcer.get(name) != value:
                raise ValueError(f"Checkpoint RGCER setting {name!r} does not match current configuration")
        current_manifest_hash = self._manifest_hash()
        if checkpoint["split_manifest_hash"] != current_manifest_hash:
            raise ValueError("Checkpoint split manifest does not match the current experiment")
        if checkpoint["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("Checkpoint feature schema does not match current code")
        if formal_v2:
            stored_data = checkpoint["data_config"]
            current_data = self._data_config()
            if not isinstance(stored_data, dict):
                raise ValueError("Checkpoint data_config is invalid")
            for name in (
                "datastore_format_version",
                "datastore_fingerprint",
                "feature_schema_version",
                "max_path_distance",
                "split_manifest_hash",
                "task_names",
                "max_nodes_filter",
            ):
                if stored_data.get(name) != current_data.get(name):
                    raise ValueError(f"Checkpoint data contract field {name!r} does not match current DataStore")
            if stored_data.get("datastore_format_version") != 2:
                raise ValueError("Checkpoint data_config is not a DataStore V2 contract")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.loss_balancer.load_state_dict(checkpoint["weighting_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.task_scalers = checkpoint["task_scalers"]
        self.conformal_calibrator.load_state_dict(checkpoint["conformal_state"])
        self.loaded_epoch = checkpoint["epoch"]
        stored_routing_enabled = checkpoint.get("routing_enabled")
        if stored_routing_enabled is None:
            stored_routing_enabled = self._routing_enabled(checkpoint["epoch"])
        self.loaded_routing_enabled = bool(stored_routing_enabled)
        if path.name.endswith("_best.pt"):
            self.best_checkpoint_path = path
            self.best_epoch = int(checkpoint["epoch"])
            self.best_routing_enabled = self.loaded_routing_enabled
        if checkpoint.get("train_loss_buffer") is not None:
            self.train_loss_buffer = np.asarray(checkpoint["train_loss_buffer"], dtype=float)
        self.optimizer_updates = int(checkpoint.get("optimizer_updates", 0))
        self.load_path = str(path)
        print(f"Loaded strict v{expected_version} checkpoint: {path}")

    def train(
        self,
        train_dataloaders_dict,
        val_dataloaders_dict,
        calibration_dataloaders_dict=None,
        test_dataloaders_dict=None,
        epochs=1,
        params_main=None,
    ):
        self.final_test_result = None
        if not any(loader is not None and len(loader) > 0 for loader in train_dataloaders_dict.values()):
            raise ValueError("All training dataloaders are empty")
        if not self.task_scalers:
            self._fit_task_scalers(train_dataloaders_dict)
        self.quantile_loss = QuantileRegressionLoss(
            lower_quantile=getattr(self.args, "lower_quantile", 0.05),
            upper_quantile=getattr(self.args, "upper_quantile", 0.95),
            quantile_weight=1.0,
        )
        history = []
        start_epoch = int(self.loaded_epoch) + 1 if self.loaded_epoch is not None else 0
        total_epochs = int(epochs)
        if total_epochs < 0:
            raise ValueError("epochs must be non-negative")
        buffer_size = max(total_epochs, start_epoch + 1, 2)
        existing_buffer = getattr(self, "train_loss_buffer", None)
        if existing_buffer is None or existing_buffer.shape[0] != self.task_num:
            self.train_loss_buffer = np.ones((self.task_num, buffer_size), dtype=float)
        elif existing_buffer.shape[1] < buffer_size:
            expanded = np.ones((self.task_num, buffer_size), dtype=float)
            expanded[:, : existing_buffer.shape[1]] = existing_buffer
            self.train_loss_buffer = expanded
        for epoch in range(start_epoch, total_epochs):
            self.loss_balancer.train_loss_buffer = self.train_loss_buffer
            train_result = self._train_epoch(train_dataloaders_dict, epoch)
            for index, task in enumerate(self.task_name):
                if np.isfinite(train_result["loss"][task]):
                    self.train_loss_buffer[index, epoch] = train_result["loss"][task]
            self._print_metrics("train", epoch, train_result)
            validation_result = self._evaluate(val_dataloaders_dict, mode="validation", epoch=epoch)
            history.append({"epoch": epoch, "train": train_result, "validation": validation_result})
            # The best checkpoint is chosen by the selection scope (human3 by
            # default on ToxAcute), not by the all-task macro score, so an
            # epoch that only improves animal endpoints cannot steal the
            # checkpoint from a better human endpoint epoch.
            if self._eligible_for_best(epoch) and (
                self._best_state is None or validation_result["selection_score"] > self.best_val_score
            ):
                self.best_val_score = validation_result["selection_score"]
                self._best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.best_routing_enabled = self._routing_enabled(epoch)
                self._best_training_state = {
                    "model_state": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
                    "weighting_state": copy.deepcopy(self.loss_balancer.state_dict()),
                    "train_loss_buffer": self.train_loss_buffer.copy(),
                    "optimizer_updates": int(self.optimizer_updates),
                    "epoch": epoch,
                }
                self.best_checkpoint_path = self._save_checkpoint(
                    epoch, f"{getattr(params_main or self.args, 'ckpt_name', 'model')}_best.pt"
                )
            self._save_checkpoint(epoch, f"{getattr(params_main or self.args, 'ckpt_name', 'model')}_last.pt")
        if self._is_rgcer and bool(getattr(self.args, "routing_enabled", True)) and self._best_state is None:
            raise ValueError("Full RGCER requires at least one validation epoch after HPS warm-up")
        if self._best_training_state is not None:
            self.model.load_state_dict(self._best_training_state["model_state"], strict=True)
            self.optimizer.load_state_dict(self._best_training_state["optimizer_state"])
            self.loss_balancer.load_state_dict(self._best_training_state["weighting_state"], strict=True)
            self.train_loss_buffer = self._best_training_state["train_loss_buffer"].copy()
            self.optimizer_updates = self._best_training_state["optimizer_updates"]
            self.loaded_routing_enabled = self.best_routing_enabled
        routing_state = self.best_routing_enabled
        if routing_state is None:
            routing_state = self.loaded_routing_enabled
        if calibration_dataloaders_dict is not None:
            self._fit_conformal(
                calibration_dataloaders_dict,
                routing_enabled_override=routing_state,
            )
            if self.best_checkpoint_path is not None:
                checkpoint_epoch = self.best_epoch
                if checkpoint_epoch is None:
                    checkpoint_epoch = self.loaded_epoch if self.loaded_epoch is not None else max(total_epochs - 1, 0)
                self._save_checkpoint(checkpoint_epoch, self.best_checkpoint_path.name)
                self.load_checkpoint(self.best_checkpoint_path)
        if test_dataloaders_dict and any(loader is not None and len(loader) > 0 for loader in test_dataloaders_dict.values()):
            self.final_test_result = self._evaluate(
                test_dataloaders_dict,
                mode="test",
                epoch=None,
                routing_enabled_override=routing_state,
            )
        return history

    def test(self, dataloaders_dict, epoch=None, mode="test"):
        if not self.task_scalers:
            raise ValueError("Task scalers are missing; load a strict trained checkpoint first")
        if not hasattr(self, "quantile_loss"):
            self.quantile_loss = QuantileRegressionLoss(
                lower_quantile=getattr(self.args, "lower_quantile", 0.05),
                upper_quantile=getattr(self.args, "upper_quantile", 0.95),
                quantile_weight=1.0,
            )
        return self._evaluate(dataloaders_dict, mode=mode, epoch=epoch)
