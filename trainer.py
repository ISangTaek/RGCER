"""Training and evaluation loop for the stateless Phase 1--2 models."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from metric import compute_classification_metrics, compute_regression_metrics
from molecular_features import FEATURE_SCHEMA_VERSION
from split_manifest import load_manifest, manifest_hash
from utils import count_parameters


class Trainer:
    """Task-wise trainer that never combines unrelated molecular batches."""

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

        if args.gpu_id != "cpu" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.gpu_id}")
        else:
            self.device = torch.device("cpu")

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
        self.best_val_score = -float("inf")
        self.best_checkpoint_path = None
        self._best_state = None

        if load_path is not None:
            self.load_checkpoint(load_path)
        elif getattr(args, "mode", "train") in {"test", "batch_inference", "single_inference"}:
            raise FileNotFoundError("test/inference mode requires --load_path to a strict checkpoint")

        count_parameters(self.model)

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_optimizer(self, optim_param):
        optim_name = str(optim_param.get("optim", "adam")).lower()
        if optim_name == "adam":
            optimizer_class = torch.optim.Adam
        elif optim_name == "adamw":
            optimizer_class = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")
        parameters = list(self.model.parameters()) + list(self.loss_balancer.parameters())
        return optimizer_class(
            parameters,
            **{key: value for key, value in optim_param.items() if key != "optim"},
        )

    def _is_regression(self, task):
        return "RMSE" in self.task_dict[task].get("metrics", [])

    def _fit_task_scalers(self, train_dataloaders_dict):
        """Fit regression scalers using labels from train split only."""
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
            if values:
                labels = torch.cat(values)
                mean = float(labels.mean())
                std = float(labels.std(unbiased=False))
                std = max(std, 1e-6)
                self.task_scalers[task] = {"mean": mean, "std": std, "count": int(labels.numel())}
            else:
                self.task_scalers[task] = {"mean": 0.0, "std": 1.0, "count": 0}

    def _normalize_target(self, task, labels):
        scaler = self.task_scalers.get(task, {"mean": 0.0, "std": 1.0})
        return (labels - scaler["mean"]) / scaler["std"] if self._is_regression(task) else labels

    def _denormalize_prediction(self, task, prediction):
        scaler = self.task_scalers.get(task, {"mean": 0.0, "std": 1.0})
        return prediction * scaler["std"] + scaler["mean"] if self._is_regression(task) else prediction

    def _task_schedule(self, dataloaders_dict, epoch):
        schedule = []
        for task in self.task_name:
            loader = dataloaders_dict.get(task)
            if loader is not None:
                schedule.extend([task] * len(loader))
        rng = np.random.default_rng(self.seed + epoch)
        rng.shuffle(schedule)
        return schedule

    def _loss(self, task, prediction, labels):
        loss_fn = self.task_dict[task]["loss_fn"]
        if hasattr(loss_fn, "compute_loss"):
            return loss_fn.compute_loss(prediction, labels)
        return loss_fn(prediction, labels)

    @staticmethod
    def _valid_batch(batch):
        return batch is not None and not batch.get("is_empty", False) and batch.y.numel() > 0

    def _train_epoch(self, train_dataloaders_dict, epoch):
        self.model.train()
        self.loss_balancer.train()
        iterators = {
            task: iter(loader)
            for task, loader in train_dataloaders_dict.items()
            if loader is not None and len(loader) > 0
        }
        active_tasks = list(iterators)
        max_steps = max((len(train_dataloaders_dict[task]) for task in active_tasks), default=0)
        buffers = {task: {"pred": [], "label": []} for task in self.task_name}
        losses = {task: [] for task in self.task_name}
        updates = 0
        self.loss_balancer.epoch = epoch
        self.loss_balancer.train_loss_buffer = getattr(
            self, "train_loss_buffer", np.ones((self.task_num, max(epoch, 2)))
        )

        for _ in range(max_steps):
            task_losses = []
            active_mask = torch.zeros(self.task_num, dtype=torch.bool, device=self.device)
            step_outputs = []
            for task_index, task in enumerate(self.task_name):
                if task not in iterators:
                    task_losses.append(torch.zeros((), device=self.device))
                    step_outputs.append(None)
                    continue
                batch = next(iterators[task], None)
                if batch is None:
                    iterators[task] = iter(train_dataloaders_dict[task])
                    batch = next(iterators[task], None)
                if not self._valid_batch(batch):
                    task_losses.append(torch.zeros((), device=self.device))
                    step_outputs.append(None)
                    continue
                batch = batch.to(self.device)
                labels = batch.y.reshape(-1, 1).float()
                normalized_labels = self._normalize_target(task, labels)
                output = self.model(batch, task_name=task, mode="train")
                prediction = output[task]
                loss = self._loss(task, prediction, normalized_labels)
                active_mask[task_index] = True
                task_losses.append(loss)
                step_outputs.append((task, prediction, labels, loss.detach()))

            if not bool(active_mask.any()):
                continue
            self.optimizer.zero_grad(set_to_none=True)
            losses_tensor = torch.stack(task_losses)
            self.loss_balancer.backward(losses_tensor, active_mask=active_mask)
            clip_value = getattr(self.args, "grad_clip", 1.0)
            if clip_value is not None and float(clip_value) > 0:
                clip_grad_norm_(
                    list(self.model.parameters()) + list(self.loss_balancer.parameters()),
                    float(clip_value),
                )
            self.optimizer.step()
            for item in step_outputs:
                if item is None:
                    continue
                task, prediction, labels, detached_loss = item
                raw_prediction = self._denormalize_prediction(task, prediction.detach())
                buffers[task]["pred"].append(raw_prediction.cpu())
                buffers[task]["label"].append(labels.detach().cpu())
                losses[task].append(float(detached_loss.cpu()))
            updates += 1

        metrics = self._score_buffers(buffers)
        metrics["loss"] = {
            task: float(np.mean(losses[task])) if losses[task] else np.nan for task in self.task_name
        }
        metrics["updates"] = updates
        return metrics

    def _score_buffers(self, buffers):
        task_scores = {}
        primary_values = []
        for task in self.task_name:
            pred = buffers[task]["pred"]
            labels = buffers[task]["label"]
            if not pred:
                task_scores[task] = {name: np.nan for name in self.task_dict[task].get("metrics", [])}
                continue
            pred = torch.cat(pred).numpy()
            labels = torch.cat(labels).numpy()
            if self._is_regression(task):
                values = compute_regression_metrics(pred, labels)
            else:
                values = compute_classification_metrics(pred, labels)
            task_scores[task] = {name: values.get(name, np.nan) for name in self.task_dict[task]["metrics"]}
            primary = task_scores[task][self.task_dict[task]["metrics"][0]]
            if np.isfinite(primary):
                primary_values.append(self.task_dict[task]["weight"][0] * primary)
        return {"tasks": task_scores, "score": float(np.mean(primary_values)) if primary_values else -float("inf")}

    @staticmethod
    def _print_metrics(mode, epoch, result):
        prefix = f"{mode}" if epoch is None else f"{mode} epoch={epoch:03d}"
        pieces = [f"{prefix}: score={result['score']:.6f}"]
        for task, values in result["tasks"].items():
            rendered = ", ".join(f"{name}={value:.6f}" for name, value in values.items())
            pieces.append(f"{task}[{rendered}]")
        print(" | ".join(pieces))

    def _evaluate(self, dataloaders_dict, mode="validation", epoch=None):
        self.model.eval()
        buffers = {task: {"pred": [], "label": []} for task in self.task_name}
        with torch.no_grad():
            for task in self.task_name:
                loader = dataloaders_dict.get(task)
                if loader is None:
                    continue
                for batch in loader:
                    if not self._valid_batch(batch):
                        continue
                    batch = batch.to(self.device)
                    output = self.model(batch, task_name=task, mode=mode)
                    prediction = self._denormalize_prediction(task, output[task])
                    buffers[task]["pred"].append(prediction.cpu())
                    buffers[task]["label"].append(batch.y.reshape(-1, 1).float().cpu())
        result = self._score_buffers(buffers)
        self._print_metrics(mode, epoch, result)
        return result

    def _manifest_hash(self):
        manifest_path = getattr(self.args, "preprocessed_data_dir", None)
        if not manifest_path:
            return None
        manifest_path = Path(manifest_path) / "split_manifest.json"
        if not manifest_path.exists():
            return None
        return manifest_hash(load_manifest(manifest_path))

    def _checkpoint_payload(self, epoch):
        return {
            "checkpoint_version": 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "weighting_state": self.loss_balancer.state_dict(),
            "epoch": int(epoch),
            "configuration": vars(self.args) if hasattr(self.args, "__dict__") else {},
            "task_names": list(self.task_name),
            "task_scalers": self.task_scalers,
            "split_manifest_hash": self._manifest_hash(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
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
            "model_state",
            "optimizer_state",
            "weighting_state",
            "epoch",
            "configuration",
            "task_names",
            "task_scalers",
            "split_manifest_hash",
            "feature_schema_version",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Checkpoint is not a Phase 1+ checkpoint; missing fields: {sorted(missing)}")
        if list(checkpoint["task_names"]) != self.task_name:
            raise ValueError("Checkpoint task_names do not match the current experiment")
        if checkpoint["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("Checkpoint feature schema does not match the current code")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.loss_balancer.load_state_dict(checkpoint["weighting_state"], strict=True)
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.task_scalers = checkpoint["task_scalers"]
        self.loaded_epoch = checkpoint.get("epoch")
        self.load_path = str(path)
        print(f"Loaded strict checkpoint: {path}")

    def train(self, train_dataloaders_dict, val_dataloaders_dict, test_dataloaders_dict, epochs, params_main=None):
        if not any(loader is not None and len(loader) > 0 for loader in train_dataloaders_dict.values()):
            raise ValueError("All training dataloaders are empty")
        if not self.task_scalers:
            self._fit_task_scalers(train_dataloaders_dict)

        history = []
        self.train_loss_buffer = np.ones((self.task_num, max(int(epochs), 2)), dtype=float)
        for epoch in range(int(epochs)):
            self.loss_balancer.train_loss_buffer = self.train_loss_buffer
            train_result = self._train_epoch(train_dataloaders_dict, epoch)
            for task_index, task in enumerate(self.task_name):
                if np.isfinite(train_result["loss"][task]):
                    self.train_loss_buffer[task_index, epoch] = train_result["loss"][task]
            self._print_metrics("train", epoch, train_result)
            val_result = self._evaluate(val_dataloaders_dict, mode="validation", epoch=epoch)
            history.append({"epoch": epoch, "train": train_result, "validation": val_result})
            if self._best_state is None or val_result["score"] > self.best_val_score:
                self.best_val_score = val_result["score"]
                self._best_state = copy.deepcopy(self.model.state_dict())
                self.best_checkpoint_path = self._save_checkpoint(
                    epoch, f"{getattr(params_main or self.args, 'ckpt_name', 'model')}_best.pt"
                )
            self._save_checkpoint(epoch, f"{getattr(params_main or self.args, 'ckpt_name', 'model')}_last.pt")

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state, strict=True)
            if self.best_checkpoint_path is not None:
                # Reload the exact serialized state, including task scalers, so
                # a later test invocation follows the same reproducible path.
                self.load_checkpoint(self.best_checkpoint_path)
        if test_dataloaders_dict and any(loader is not None and len(loader) > 0 for loader in test_dataloaders_dict.values()):
            self._evaluate(test_dataloaders_dict, mode="test", epoch=None)
        return history

    def test(self, dataloaders_dict, epoch=None, mode="test"):
        return self._evaluate(dataloaders_dict, mode=mode, epoch=epoch)
