"""Training, calibration, and evaluation for the three Graphormer variants."""

from __future__ import annotations

import copy
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from conformal import ConformalCalibrator, resolve_conformal_tasks
from architecture.prediction_heads import decode_prediction
from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS, HUMAN_TARGET_TASKS
from loss import QuantileRegressionLoss
from metric import (
    compute_classification_metrics,
    compute_interval_metrics,
    compute_regression_metrics,
)
from molecular_features import FEATURE_SCHEMA_VERSION
from reproducibility import (
    EPOCH_SEED_SCHEME,
    LOADER_SEED_SCHEME,
    PERSISTENT_WORKER_POLICY,
    SEED_POLICY_VERSION,
    reseed_train_loaders,
    scheduled_batches,
    seed_everything,
    stable_seed,
    state_dict_sha256,
)
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
        self.conformal_scope = str(getattr(args, "conformal_scope", "human3"))
        if self.conformal_scope not in {"human3", "all_tasks"}:
            raise ValueError(f"Unknown conformal_scope: {self.conformal_scope!r}")
        self.conformal_fit_report: list[dict] = []
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
        # Defensive only (review §6): main() seeds before model construction;
        # this re-seed keeps directly-constructed Trainers deterministic.
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
        # Review §7: auditable initialization hash — same seed must produce
        # the same value across fresh processes, different seeds must not.
        self.initial_model_sha256 = state_dict_sha256(self.model)
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
        # Diagnostic-plan §6-§9: read-only per-epoch logging, instantiated by
        # train(); kept None here so unit-level callers never touch it.
        self._diagnostics = None
        self.sample_row_index = {}
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
        seed_everything(seed)

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
            if task not in self.conformal_calibrator.states:
                # Fail fast instead of silently reporting uncalibrated bands
                # as conformal intervals (e.g. --no-fit_conformal runs).
                raise ValueError(
                    f"No conformal state fitted for task {task!r}; re-run with "
                    "--fit_conformal or request apply_conformal=False."
                )
            result["lower"], result["upper"] = self.conformal_calibrator.apply(
                task, result["lower"], result["upper"]
            )
        return result

    def _maybe_update_best(self, validation_result, epoch, params_main=None):
        """Replace the historical best only on a strictly better score.

        Encapsulated so resume tests can verify that a restored
        ``best_val_score`` guards the selection: after resuming from a
        ``last.pt`` with a historical best of 10, a validation score of 5
        must NOT replace it (review-2 §19).
        """

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
            return True
        return False

    def _task_schedule(self, dataloaders_dict, epoch):
        """Per-epoch task batch order (review §38-39).

        Default ``proportional`` keeps the historical data-proportional
        exposure so every baseline trains under the same schedule; the
        recorded diagnostics make that bias visible instead of silent.
        ``human_target_floor`` (ablation-only) additionally appends one full
        extra pass of each human target loader.
        """

        schedule = []
        for task in self.task_name:
            loader = dataloaders_dict.get(task)
            if loader is not None:
                schedule.extend([task] * len(loader))
        sampling = str(getattr(self.args, "task_sampling", "proportional") or "proportional")
        if sampling == "human_target_floor":
            for task in HUMAN_TARGET_TASKS:
                if task not in self.task_name:
                    continue
                loader = dataloaders_dict.get(task)
                if loader is not None and len(loader) > 0:
                    schedule.extend([task] * len(loader))
        elif sampling != "proportional":
            raise ValueError(f"Unknown --task_sampling: {sampling!r}")
        rng = np.random.default_rng(self.seed + int(epoch))
        rng.shuffle(schedule)
        return list(schedule)

    def _schedule_diagnostics(self, schedule, dataloaders_dict):
        """Per-epoch exposure report so data-proportionality stays auditable."""

        human_tasks = [task for task in HUMAN_TARGET_TASKS if task in self.task_name]
        batches: dict[str, int] = {}
        for task in schedule:
            batches[task] = batches.get(task, 0) + 1
        total_updates = max(len(schedule), 1)
        fractions = {task: round(count / total_updates, 6) for task, count in sorted(batches.items())}
        dataset_batches = {
            task: len(loader)
            for task in self.task_name
            for loader in [dataloaders_dict.get(task)]
            if loader is not None
        }
        human_batches = sum(batches.get(task, 0) for task in human_tasks)
        return {
            "task_batches": dict(sorted(batches.items())),
            "loader_batches_per_pass": dataset_batches,
            "fraction_of_updates": fractions,
            "human3_fraction": round(human_batches / total_updates, 6),
            "total_updates": total_updates,
            "task_sampling": str(getattr(self.args, "task_sampling", "proportional")),
        }

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
            "final_loss": final_loss.detach(),
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
                "joint_source_weights": [],
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
        if "joint_source_weights" in diagnostics:
            cache["joint_source_weights"].append(diagnostics["joint_source_weights"].detach().cpu())
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
                        "mean_transfer_mass": float((1.0 - null).mean()),
                    }
                )
            if cache.get("joint_source_weights"):
                joint = torch.cat(cache["joint_source_weights"])
                total_source_mass = joint.sum(dim=-1)
                result.update(
                    {
                        "mean_joint_source_mass": float(total_source_mass.mean()),
                        "routing_variance_joint": float(joint.var(dim=0, unbiased=False).mean()),
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
        # Review §32: the epoch's random trajectory (dropout, any training
        # stochasticity) derives from (base_seed, epoch), never process
        # history; loader shuffles derive from (base_seed, task, epoch).
        seed_everything(stable_seed(self.seed, "train_epoch", epoch))
        reseed_train_loaders(train_dataloaders_dict, self.seed, epoch)
        schedule = self._task_schedule(train_dataloaders_dict, epoch)
        schedule_diagnostics = self._schedule_diagnostics(schedule, train_dataloaders_dict)
        self.schedule_usage = {task: 0 for task in self.task_name}
        self.training_cache = {}
        # Review §15: a task may be scheduled for more passes than its
        # loader provides batches (human_target_floor extra passes).  The
        # iterator restarts only while scheduled-but-unconsumed batches
        # remain, so extra passes are really consumed without unbounded
        # cycling (the old code built one iterator and silently skipped the
        # exhausted extra pass).
        planned_passes: dict[str, int] = {}
        for task in schedule:
            planned_passes[task] = planned_passes.get(task, 0) + 1
        usable_loaders = {
            task: loader
            for task, loader in train_dataloaders_dict.items()
            if loader is not None and len(loader) > 0
        }
        iterators = {task: iter(loader) for task, loader in usable_loaders.items()}
        consumed_batches: dict[str, int] = {task: 0 for task in usable_loaders}

        def next_scheduled_batch(task):
            if task not in iterators:
                return None
            batch = next(iterators[task], None)
            if batch is None and consumed_batches[task] < planned_passes[task]:
                iterators[task] = iter(usable_loaders[task])
                batch = next(iterators[task], None)
            if batch is None:
                return None
            consumed_batches[task] += 1
            return batch

        actual_samples: dict[str, int] = {task: 0 for task in self.task_name}
        buffers = {task: {"pred": [], "label": []} for task in self.task_name}
        losses = {task: [] for task in self.task_name}
        final_losses = {task: [] for task in self.task_name}
        base_losses = {task: [] for task in self.task_name}
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
                batch = next_scheduled_batch(task)
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
            if getattr(self, "_diagnostics", None) is not None:
                self._diagnostics.record_gradients(self.model)
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
                actual_samples[task] += int(bundle["labels"].shape[0])
                losses[task].append(float(bundle["loss"].cpu()))
                final_losses[task].append(float(bundle["final_loss"].cpu()))
                base_losses[task].append(float(bundle["base_loss"].cpu()))
                self._record_training_output(bundle)
        result = self._score_buffers(buffers)
        result["loss"] = {task: float(np.mean(losses[task])) if losses[task] else np.nan for task in self.task_name}
        result["final_loss"] = {
            task: float(np.mean(final_losses[task])) if final_losses[task] else np.nan for task in self.task_name
        }
        result["base_loss"] = {
            task: float(np.mean(base_losses[task])) if base_losses[task] else np.nan for task in self.task_name
        }
        self.optimizer_updates += update_count
        result["updates"] = update_count
        result["schedule_usage"] = dict(self.schedule_usage)
        schedule_diagnostics["completed_updates"] = update_count
        # Review §16: formal reports must quote actual consumption, so the
        # planned exposure stays alongside the real batch/sample counts.
        actual_batches = {task: int(count) for task, count in self.schedule_usage.items() if count}
        actual_total = max(sum(actual_batches.values()), 1)
        schedule_diagnostics["planned_task_batches"] = dict(schedule_diagnostics.get("task_batches", {}))
        schedule_diagnostics["actual_task_batches"] = dict(sorted(actual_batches.items()))
        schedule_diagnostics["actual_task_samples"] = {
            task: int(count) for task, count in actual_samples.items() if count
        }
        schedule_diagnostics["actual_fraction_of_updates"] = {
            task: round(count / actual_total, 6) for task, count in sorted(actual_batches.items())
        }
        human_tasks = [task for task in HUMAN_TARGET_TASKS if task in self.task_name]
        human_actual = sum(actual_batches.get(task, 0) for task in human_tasks)
        schedule_diagnostics["human3_actual_fraction"] = round(human_actual / actual_total, 6)
        result["schedule_diagnostics"] = schedule_diagnostics
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
                "sample_id": [],
                "base": [],
                "route": [],
                "final": [],
                "target": [],
                "null": [],
                "source_weights": [],
                "joint_source_weights": [],
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
                    # Per-task conformal gating (§23): point metrics cover the
                    # full task list, but qhat is only applied where a state
                    # exists — never labelling uncalibrated animal intervals
                    # as conformal.
                    apply_task_conformal = (
                        apply_conformal and task in self.conformal_calibrator.states
                    )
                    final_decoded = self.decode_task_output(task, final_raw, apply_conformal=apply_task_conformal)
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
                    # DataStore sample identity for per-sample diagnostic dumps;
                    # batch index alone cannot be aligned across runs (plan §10).
                    route_records[task]["sample_id"].extend(
                        str(value) for value in (getattr(batch, "sample_id", None) or [])
                    )
                    if "null_weight" in diagnostics:
                        route_records[task]["null"].append(diagnostics["null_weight"].cpu())
                    if "source_weights" in diagnostics:
                        route_records[task]["source_weights"].append(diagnostics["source_weights"].cpu())
                    if "joint_source_weights" in diagnostics:
                        route_records[task]["joint_source_weights"].append(
                            diagnostics["joint_source_weights"].cpu()
                        )
                    if "routing_entropy" in diagnostics:
                        route_records[task]["entropy"].append(diagnostics["routing_entropy"].cpu())
        return buffers, records, route_records

    def _evaluate(self, dataloaders_dict, mode="validation", epoch=0, routing_enabled_override=None):
        buffers, records, route_records = self._collect_predictions(
            dataloaders_dict,
            epoch=epoch,
            apply_conformal=(mode == "test" and bool(getattr(self.args, "fit_conformal", True))),
            routing_enabled_override=routing_enabled_override,
        )
        # Kept for the per-epoch diagnostic writer; the summary below stays the
        # single source of truth for reported routing aggregates.
        self._last_route_records = route_records
        result = self._score_buffers(buffers)
        if self._prediction_mode() == "quantile":
            scoped_tasks = self._conformal_tasks()
            task_interval_metrics = {}
            for task in scoped_tasks:
                if not records[task]["target"]:
                    continue
                lower = torch.cat(records[task]["lower"])
                upper = torch.cat(records[task]["upper"])
                target = torch.cat(records[task]["target"])
                if mode == "test" and task in self.conformal_calibrator.states:
                    lower, upper = self.conformal_calibrator.apply(task, lower, upper)
                # Interval reports keep the per-task view even when the
                # split carries few samples; macro alone would hide which
                # human endpoint is actually covered.
                task_interval_metrics[task] = compute_interval_metrics(
                    lower, upper, target, self.args.conformal_alpha
                )
            if task_interval_metrics:
                macro_values = list(task_interval_metrics.values())
                result["interval"] = {
                    "scope": self.conformal_scope,
                    "tasks": {
                        task: {key: float(value) for key, value in values.items()}
                        for task, values in task_interval_metrics.items()
                    },
                    "macro": {
                        key: float(np.nanmean([value[key] for value in macro_values]))
                        for key in macro_values[0]
                    },
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
                values["mean_transfer_mass"] = float((1.0 - null).mean())
                null_values = null.detach().cpu().numpy().reshape(-1)
                route_regret_values = regret.detach().cpu().numpy().reshape(-1)
                harmful_route = route_regret_values > 0
                values["null_harmful_route_auroc"] = cls._safe_auroc(null_values, harmful_route)
                values["null_harmful_route_auprc"] = cls._safe_auprc(null_values, harmful_route)
                values["null_weight_route_regret_spearman"] = cls._safe_spearman(
                    null_values, route_regret_values
                )
            if record.get("joint_source_weights"):
                joint = torch.cat(record["joint_source_weights"])
                values["mean_joint_source_mass"] = float(joint.sum(dim=-1).mean())
                values["routing_variance_joint"] = float(joint.var(dim=0, unbiased=False).mean())
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

    def _conformal_tasks(self):
        """Tasks that receive CQR states, per ``--conformal_scope``.

        Primary conformal evaluation targets the three human endpoints even
        though training/point evaluation cover all 59; interval reports are
        scoped accordingly instead of silently applying qhat everywhere.
        Scope resolution goes through the shared helper (review §20).
        """

        scoped = resolve_conformal_tasks(self.conformal_scope, self.task_name)
        return [
            task
            for task in scoped
            if self._is_regression(task) and self.task_dict[task].get("fit_conformal", True)
        ]

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
        self.conformal_fit_report = []
        for task in self._conformal_tasks():
            if not records[task]["target"]:
                raise ValueError(f"No calibration samples for {task}.")
            state = self.conformal_calibrator.fit_task(
                task,
                torch.cat(records[task]["lower"]),
                torch.cat(records[task]["upper"]),
                torch.cat(records[task]["target"]),
            )
            from conformal import minimum_calibration_size

            required_minimum = minimum_calibration_size(self.conformal_calibrator.alpha)
            entry = {
                "task": task,
                "alpha": float(self.conformal_calibrator.alpha),
                "calibration_count": int(state.count),
                "required_minimum": int(required_minimum),
                "stability_threshold": int(self.conformal_calibrator.min_calibration_size),
                "qhat": float(state.qhat),
                "status": (
                    "OK"
                    if state.count >= self.conformal_calibrator.min_calibration_size
                    else "LOW_CALIBRATION_STABILITY"
                ),
            }
            self.conformal_fit_report.append(entry)
            print(
                "[conformal] "
                + json.dumps(
                    {key: entry[key] for key in entry if key != "status"}
                    | {"status": entry["status"]}
                )
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
            # §42 provenance: behaviour-affecting sizes/rates that a
            # state_dict alone cannot reveal on resume.
            "hidden_dim": int(getattr(self.args, "hidden_dim", 0) or 0),
            "a_layers": int(getattr(self.args, "a_layers", 0) or 0),
            "a_heads": int(getattr(self.args, "a_heads", 0) or 0),
            "mid_dim": int(getattr(self.args, "mid_dim", 0) or 0),
            "head_hidden_dim": int(getattr(self.args, "head_hidden_dim", 0) or 0),
            "head_dropout": float(getattr(self.args, "head_dropout", 0.0)),
            "adapter_ratio": float(getattr(self.args, "adapter_ratio", 0.25)),
            "response_hidden_dim": int(getattr(self.args, "response_hidden_dim", 0) or 0),
            "task_sampling": str(getattr(self.args, "task_sampling", "proportional")),
            "conformal_scope": self.conformal_scope,
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

    def _checkpoint_payload(self, epoch, *, include_historical_best=False):
        checkpoint_version = 6 if self.data_metadata is not None else 4
        payload = {
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
            "conformal_report": list(self.conformal_fit_report),
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
            # Review-2 §32: the supported resume contract is epoch-boundary
            # determinism via derived seeds; record the policy so a resume
            # under a different scheme fails loudly instead of drifting.
            "reproducibility": {
                "base_seed": int(self.seed),
                "seed_policy_version": SEED_POLICY_VERSION,
                "epoch_seed_scheme": EPOCH_SEED_SCHEME,
                "loader_seed_scheme": LOADER_SEED_SCHEME,
                "persistent_worker_policy": PERSISTENT_WORKER_POLICY,
                "initial_model_sha256": getattr(self, "initial_model_sha256", None),
            },
            # Review-2 §15: model-selection continuity — a resumed run must
            # keep comparing against the historical best, not restart from
            # -inf.  Every checkpoint records it; *_last.pt additionally
            # embeds the historical best training state so it alone is a
            # complete resume artifact (review-2 §16).
            "selection_state": {
                "best_val_score": (
                    float(self.best_val_score) if self.best_val_score is not None else None
                ),
                "best_epoch": self.best_epoch,
                "best_routing_enabled": self.best_routing_enabled,
                "best_checkpoint_name": (
                    self.best_checkpoint_path.name if self.best_checkpoint_path else None
                ),
            },
        }
        if include_historical_best and self._best_training_state is not None:
            payload["best_training_state"] = copy.deepcopy(self._best_training_state)
            if self._best_state is not None:
                payload["best_model_state"] = copy.deepcopy(self._best_state)
        return payload

    def _save_checkpoint(self, epoch, filename, *, include_historical_best=False):
        if self.save_path is None:
            return None
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self.save_path / filename
        # Review-2 §17: only *_last.pt embeds the historical best state, so
        # it alone can resume the full selection contract without a copy of
        # *_best.pt; best.pt must not carry a redundant copy of itself.
        torch.save(
            self._checkpoint_payload(epoch, include_historical_best=include_historical_best),
            path,
        )
        return path

    def load_checkpoint(self, path):
        path = Path(path)
        if path.is_dir():
            path = path / f"{getattr(self.args, 'ckpt_name', 'model')}_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        formal_v2 = self.data_metadata is not None
        expected_version = 6 if formal_v2 else 4
        required = {
            "checkpoint_version", "model_state", "optimizer_state", "weighting_state", "epoch",
            "configuration", "task_names", "architecture_config", "prediction_mode", "quantile_config",
            "conformal_state", "hps_warmup_epochs", "rgcer_config", "task_metadata", "task_scalers",
            "split_manifest_hash", "feature_schema_version",
        }
        if formal_v2:
            required.update({"data_config", "effective_rgcer_config", "routing_enabled"})
            # Review-2 §31: formal v6 checkpoints must carry the full
            # reproducibility and selection contracts — no optional legacy
            # semantics.
            required.update({"reproducibility", "selection_state"})
        missing = required.difference(checkpoint)
        if missing or checkpoint["checkpoint_version"] != expected_version:
            raise ValueError(
                f"Checkpoint is not a strict v{expected_version} checkpoint; missing={sorted(missing)}"
            )
        if list(checkpoint["task_names"]) != self.task_name:
            raise ValueError("Checkpoint task_names do not match the current experiment")
        stored_repro = checkpoint.get("reproducibility")
        if formal_v2 and stored_repro is None:
            raise ValueError("Formal v6 checkpoints must embed a reproducibility block")
        if stored_repro is not None:
            if int(stored_repro.get("seed_policy_version", -1)) != SEED_POLICY_VERSION:
                raise ValueError(
                    "Checkpoint seed policy version "
                    f"{stored_repro.get('seed_policy_version')!r} does not match the current "
                    f"policy ({SEED_POLICY_VERSION}); epoch-boundary resume would not be faithful"
                )
            if int(stored_repro.get("base_seed", -1)) != int(self.seed):
                raise ValueError("Checkpoint base_seed does not match the current --seed")
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
            # §42 provenance: silent-training-behaviour knobs that would not
            # surface as state_dict shape mismatches on resume.
            "hidden_dim",
            "a_layers",
            "a_heads",
            "mid_dim",
            "head_hidden_dim",
            "head_dropout",
            "adapter_ratio",
            "response_hidden_dim",
            "task_sampling",
            "conformal_scope",
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
            if name == "fit_conformal":
                # Provenance-only runtime switch: a checkpoint trained with or
                # without CQR stays usable either way; requesting intervals
                # without fitted states fails at decode time below.
                continue
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
        # Review-2 §10/§18: restore the historical best-selection state so a
        # resumed run keeps comparing against the pre-interruption best
        # instead of treating any new epoch as an improvement over -inf.
        selection = checkpoint.get("selection_state")
        if selection is not None:
            stored_score = selection.get("best_val_score")
            self.best_val_score = (
                float(stored_score) if stored_score is not None else -float("inf")
            )
            if selection.get("best_epoch") is not None:
                self.best_epoch = int(selection["best_epoch"])
            if selection.get("best_routing_enabled") is not None:
                self.best_routing_enabled = bool(selection["best_routing_enabled"])
            if selection.get("best_checkpoint_name"):
                self.best_checkpoint_path = path.parent / str(selection["best_checkpoint_name"])
            if path.name.endswith("_best.pt"):
                # A v6 *_best.pt IS the historical best: its own states are
                # the best states, so rebuild the restore payload from them.
                self.best_checkpoint_path = path
                self._best_state = copy.deepcopy(checkpoint["model_state"])
                best_buffer = checkpoint.get("train_loss_buffer")
                self._best_training_state = {
                    "model_state": copy.deepcopy(checkpoint["model_state"]),
                    "optimizer_state": copy.deepcopy(checkpoint["optimizer_state"]),
                    "weighting_state": copy.deepcopy(checkpoint["weighting_state"]),
                    "train_loss_buffer": (
                        np.asarray(best_buffer, dtype=float).copy()
                        if best_buffer is not None
                        else np.zeros((self.task_num, max(int(checkpoint["epoch"]) + 1, 2)))
                    ),
                    "optimizer_updates": int(checkpoint.get("optimizer_updates", 0)),
                    "epoch": int(checkpoint["epoch"]),
                }
        embedded_best = checkpoint.get("best_training_state")
        if embedded_best is not None:
            # Review-2 §16: *_last.pt is self-contained — restore the
            # embedded historical best training state over the current one.
            self._best_training_state = copy.deepcopy(embedded_best)
            self._best_state = copy.deepcopy(
                checkpoint.get("best_model_state") or embedded_best["model_state"]
            )
            if selection is None:
                self.best_val_score = -float("inf")
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
        self._diagnostics = self._make_diagnostics_writer()
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
            best_updated = self._maybe_update_best(validation_result, epoch, params_main=params_main)
            if self._diagnostics is not None:
                # Plan §6-§9/§18: log every epoch, including warm-up epochs
                # (routing_enabled=False) so the collapse trajectory is visible.
                route_records = getattr(self, "_last_route_records", None)
                if best_updated:
                    self._diagnostics.note_best_epoch(epoch, route_records)
                self._diagnostics.log_epoch(
                    epoch,
                    train_result,
                    validation_result,
                    route_records=route_records,
                    routing_enabled=self._routing_enabled(epoch),
                    is_best=bool(best_updated),
                )
            self._save_checkpoint(
                epoch,
                f"{getattr(params_main or self.args, 'ckpt_name', 'model')}_last.pt",
                include_historical_best=True,
            )
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
        if self._diagnostics is not None and self.best_epoch is not None:
            self._diagnostics.write_best_artifacts(self.best_epoch)
        return history

    def _make_diagnostics_writer(self):
        """Build the read-only diagnostic writer when the run requests one."""

        if not bool(getattr(self.args, "diagnostics", True)):
            return None
        save_path = getattr(self, "save_path", None)
        if save_path is None:
            return None
        from run_diagnostics import RunDiagnosticsWriter

        return RunDiagnosticsWriter(
            output_dir=Path(save_path) / "diagnostics",
            task_names=list(self.task_name),
            sample_row_index=getattr(self, "sample_row_index", None),
        )

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
