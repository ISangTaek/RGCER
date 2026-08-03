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
from loss import QuantileRegressionLoss
from metric import (
    compute_classification_metrics,
    compute_interval_metrics,
    compute_regression_metrics,
)
from molecular_features import FEATURE_SCHEMA_VERSION
from split_manifest import load_manifest, manifest_hash
from utils import count_parameters


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
        self.schedule_usage = {}
        self.optimizer_updates = 0
        self.training_cache = {}
        self._is_rgcer = hasattr(getattr(self.model, "encoder", None), "task_conditioner") and hasattr(
            getattr(getattr(self.model, "encoder", None), "task_conditioner", None), "router"
        ) and hasattr(
            getattr(getattr(getattr(self.model, "encoder", None), "task_conditioner", None), "router", None),
            "response_encoder",
        )
        if load_path is not None:
            self.load_checkpoint(load_path)
        elif getattr(args, "mode", "train") in {"test", "batch_inference", "single_inference"}:
            raise FileNotFoundError("test/inference mode requires --load_path to a strict v3 checkpoint")
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

    def _routing_enabled(self, epoch):
        if not self._is_rgcer:
            return True
        if not bool(getattr(self.args, "routing_enabled", True)):
            return False
        # Validation/test callers may pass ``None`` because they are not part
        # of the epoch schedule.  A trained checkpoint should use the routed
        # path in those phases.
        if epoch is None:
            return True
        return epoch >= getattr(self.args, "hps_warmup_epochs", 0)

    def _forward_task(self, batch, task, epoch, return_aux=True):
        if self._is_rgcer:
            result = self.model(
                batch,
                task_name=task,
                return_aux=return_aux,
                routing_enabled=self._routing_enabled(epoch),
            )
        else:
            result = self.model(batch, task_name=task, return_aux=return_aux)
        if return_aux and isinstance(result, tuple):
            return result[0], self._diagnostics_dict(result[1])
        return result, {}

    def _training_step(self, batch, task, epoch):
        labels = batch.y.reshape(-1, 1).float()
        normalized_labels = self._normalize_target(task, labels)
        output, diagnostics = self._forward_task(batch, task, epoch, return_aux=True)
        final_raw = output[task]
        final_loss = self._prediction_loss(task, final_raw, normalized_labels)
        base_raw = diagnostics.get("base_raw", final_raw)
        routing_enabled = self._routing_enabled(epoch)
        if routing_enabled and self._is_rgcer:
            base_loss = self._prediction_loss(task, base_raw, normalized_labels)
            total_loss = getattr(self.args, "lambda_quantile", 1.0) * final_loss + getattr(
                self.args, "lambda_base", 0.25
            ) * base_loss
        else:
            base_loss = final_loss
            total_loss = final_loss
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
        final = self.decode_task_output(task, bundle["final_raw"].detach(), apply_conformal=False)["median"]
        base = self.decode_task_output(task, bundle["base_raw"].detach(), apply_conformal=False)["median"]
        route = self.decode_task_output(task, bundle["route_raw"].detach(), apply_conformal=False)["median"]
        diagnostics = bundle["diagnostics"]
        self.training_cache.setdefault(task, {"base": [], "route": [], "final": [], "target": [], "route_regret": [], "null": [], "source_weights": []})
        cache = self.training_cache[task]
        target = bundle["labels"].cpu()
        route_regret = (route - target).abs() - (base - target).abs()
        cache["base"].append(base.cpu())
        cache["route"].append(route.cpu())
        cache["final"].append(final.cpu())
        cache["target"].append(target)
        cache["route_regret"].append(route_regret.cpu())
        if "null_weight" in diagnostics:
            cache["null"].append(diagnostics["null_weight"].detach().cpu())
        if "source_weights" in diagnostics:
            cache["source_weights"].append(diagnostics["source_weights"].detach().cpu())

    def _routing_summary(self):
        summary = {}
        for task, cache in self.training_cache.items():
            if not cache["target"]:
                continue
            result = {
                "route_regret_mean": float(torch.cat(cache["route_regret"]).mean()),
                "negative_transfer_rate": float((torch.cat(cache["route_regret"]) > 0).float().mean()),
            }
            if cache["null"]:
                null = torch.cat(cache["null"])
                result.update({"mean_null": float(null.mean()), "std_null": float(null.std(unbiased=False))})
            if cache["source_weights"]:
                weights = torch.cat(cache["source_weights"])
                result.update(
                    {
                        "mean_active_routes": float((weights > 0).sum(dim=-1).float().mean()),
                        "routing_variance": float(weights.var(dim=0, unbiased=False).mean()),
                    }
                )
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

    def _score_buffers(self, buffers):
        task_scores = {}
        primary_values = []
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
            if np.isfinite(primary):
                primary_values.append(self.task_dict[task]["weight"][0] * primary)
        return {"tasks": task_scores, "score": float(np.mean(primary_values)) if primary_values else -float("inf")}

    @staticmethod
    def _print_metrics(mode, epoch, result):
        prefix = mode if epoch is None else f"{mode} epoch={epoch:03d}"
        pieces = [f"{prefix}: score={result['score']:.6f}"]
        for task, values in result["tasks"].items():
            pieces.append(f"{task}[{', '.join(f'{key}={value:.6f}' for key, value in values.items())}]")
        print(" | ".join(pieces))
        if result.get("schedule_usage"):
            print(f"schedule_usage: {result['schedule_usage']}")
        if result.get("routing"):
            print(f"routing: {result['routing']}")

    def _collect_predictions(self, dataloaders_dict, epoch=0, apply_conformal=False):
        self.model.eval()
        buffers = {task: {"pred": [], "label": []} for task in self.task_name}
        records = {task: {"lower": [], "upper": [], "target": []} for task in self.task_name}
        route_records = {task: {"base": [], "route": [], "final": [], "target": [], "null": [], "source_weights": [], "route_regret": []} for task in self.task_name}
        with torch.no_grad():
            for task in self.task_name:
                loader = dataloaders_dict.get(task)
                if loader is None:
                    continue
                for batch in loader:
                    if not self._valid_batch(batch):
                        continue
                    batch = batch.to(self.device)
                    output, diagnostics = self._forward_task(batch, task, epoch, return_aux=True)
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
                    route_records[task]["base"].append(base)
                    route_records[task]["route"].append(route)
                    route_records[task]["final"].append(final)
                    route_records[task]["target"].append(target.cpu())
                    route_records[task]["route_regret"].append(rr)
                    if "null_weight" in diagnostics:
                        route_records[task]["null"].append(diagnostics["null_weight"].cpu())
                    if "source_weights" in diagnostics:
                        route_records[task]["source_weights"].append(diagnostics["source_weights"].cpu())
        return buffers, records, route_records

    def _evaluate(self, dataloaders_dict, mode="validation", epoch=0):
        buffers, records, route_records = self._collect_predictions(dataloaders_dict, epoch=epoch, apply_conformal=mode == "test")
        result = self._score_buffers(buffers)
        if self._prediction_mode() == "quantile":
            interval_values = []
            for task in self.task_name:
                if not self._is_regression(task) or not records[task]["target"]:
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
    def _evaluation_routing_summary(route_records):
        summary = {}
        for task, record in route_records.items():
            if not record["target"]:
                continue
            regret = torch.cat(record["route_regret"])
            values = {"route_regret_mean": float(regret.mean()), "negative_transfer_rate": float((regret > 0).float().mean())}
            if record["null"]:
                null = torch.cat(record["null"])
                values["mean_null"] = float(null.mean())
                values["std_null"] = float(null.std(unbiased=False))
            if record["source_weights"]:
                weights = torch.cat(record["source_weights"])
                values["mean_active_routes"] = float((weights > 0).sum(dim=-1).float().mean())
                values["routing_variance"] = float(weights.var(dim=0, unbiased=False).mean())
            summary[task] = values
        return summary

    def _fit_conformal(self, calibration_dataloaders_dict):
        if self._prediction_mode() != "quantile" or not getattr(self.args, "fit_conformal", True):
            return
        _, records, _ = self._collect_predictions(calibration_dataloaders_dict, epoch=10**9, apply_conformal=False)
        for task in self.task_name:
            if not self._is_regression(task):
                continue
            if not records[task]["target"]:
                raise ValueError(f"No calibration samples for {task}.")
            self.conformal_calibrator.fit_task(
                task,
                torch.cat(records[task]["lower"]),
                torch.cat(records[task]["upper"]),
                torch.cat(records[task]["target"]),
            )

    def _manifest_hash(self):
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
            "router_top_k": getattr(self.args, "router_top_k", 0),
            "router_temperature": getattr(self.args, "router_temperature", 1.0),
            "exclude_target_from_sources": getattr(self.args, "exclude_target_from_sources", True),
            "use_factorized_prompt": factorized,
            "task_metadata": [
                getattr(item, "__dict__", {})
                for item in getattr(prompt_bank, "metadata", [])
            ],
        }

    def _checkpoint_payload(self, epoch):
        return {
            "checkpoint_version": 3,
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
            "hps_warmup_epochs": getattr(self.args, "hps_warmup_epochs", 0),
            "rgcer_config": {
                "lambda_base": getattr(self.args, "lambda_base", 0.0),
                "lambda_quantile": getattr(self.args, "lambda_quantile", 1.0),
                "router_top_k": getattr(self.args, "router_top_k", 0),
                "router_temperature": getattr(self.args, "router_temperature", 1.0),
                "routing_enabled": getattr(self.args, "routing_enabled", True),
            },
            "task_metadata": self._architecture_config().get("task_metadata", []),
            "task_scalers": self.task_scalers,
            "split_manifest_hash": self._manifest_hash(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
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
        required = {
            "checkpoint_version", "model_state", "optimizer_state", "weighting_state", "epoch",
            "configuration", "task_names", "architecture_config", "prediction_mode", "quantile_config",
            "conformal_state", "hps_warmup_epochs", "rgcer_config", "task_metadata", "task_scalers",
            "split_manifest_hash", "feature_schema_version",
        }
        missing = required.difference(checkpoint)
        if missing or checkpoint["checkpoint_version"] != 3:
            raise ValueError(f"Checkpoint is not a strict v3 checkpoint; missing={sorted(missing)}")
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
            "task_metadata",
        ):
            if stored_architecture.get(name) != current_config.get(name):
                raise ValueError(f"Checkpoint setting {name!r} does not match current configuration")
        if int(checkpoint["hps_warmup_epochs"]) != int(getattr(self.args, "hps_warmup_epochs", 0)):
            raise ValueError("Checkpoint hps_warmup_epochs does not match current configuration")
        current_rgcer = {
            "lambda_base": getattr(self.args, "lambda_base", 0.0),
            "lambda_quantile": getattr(self.args, "lambda_quantile", 1.0),
            "router_top_k": getattr(self.args, "router_top_k", 0),
            "router_temperature": getattr(self.args, "router_temperature", 1.0),
            "routing_enabled": getattr(self.args, "routing_enabled", True),
        }
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
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.loss_balancer.load_state_dict(checkpoint["weighting_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.task_scalers = checkpoint["task_scalers"]
        self.conformal_calibrator.load_state_dict(checkpoint["conformal_state"])
        self.loaded_epoch = checkpoint["epoch"]
        if path.name.endswith("_best.pt"):
            self.best_checkpoint_path = path
            self.best_epoch = int(checkpoint["epoch"])
        if checkpoint.get("train_loss_buffer") is not None:
            self.train_loss_buffer = np.asarray(checkpoint["train_loss_buffer"], dtype=float)
        self.optimizer_updates = int(checkpoint.get("optimizer_updates", 0))
        self.load_path = str(path)
        print(f"Loaded strict v3 checkpoint: {path}")

    def train(
        self,
        train_dataloaders_dict,
        val_dataloaders_dict,
        calibration_dataloaders_dict=None,
        test_dataloaders_dict=None,
        epochs=1,
        params_main=None,
    ):
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
            if self._best_state is None or validation_result["score"] > self.best_val_score:
                self.best_val_score = validation_result["score"]
                self._best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
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
        if self._best_training_state is not None:
            self.model.load_state_dict(self._best_training_state["model_state"], strict=True)
            self.optimizer.load_state_dict(self._best_training_state["optimizer_state"])
            self.loss_balancer.load_state_dict(self._best_training_state["weighting_state"], strict=True)
            self.train_loss_buffer = self._best_training_state["train_loss_buffer"].copy()
            self.optimizer_updates = self._best_training_state["optimizer_updates"]
        if calibration_dataloaders_dict is not None:
            self._fit_conformal(calibration_dataloaders_dict)
            if self.best_checkpoint_path is not None:
                checkpoint_epoch = self.best_epoch
                if checkpoint_epoch is None:
                    checkpoint_epoch = self.loaded_epoch if self.loaded_epoch is not None else max(total_epochs - 1, 0)
                self._save_checkpoint(checkpoint_epoch, self.best_checkpoint_path.name)
                self.load_checkpoint(self.best_checkpoint_path)
        if test_dataloaders_dict and any(loader is not None and len(loader) > 0 for loader in test_dataloaders_dict.values()):
            self._evaluate(test_dataloaders_dict, mode="test", epoch=None)
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
