"""D8 functional source forgetting + source-retention-triggered adaptation (O6/SRTA).

Plan §20-§28 (functional forgetting), §49-§57/§67-§68 (O6 retention probe and
trigger), §93-§95 (metrics, probe identity).

Everything here is DIAGNOSTIC or a TRAIN-TIME control signal:
- the retention probe is built exclusively from Animal56 TRAIN splits,
- the functional-forgetting evaluation on Animal56 validation is never used
  for checkpoint selection / LR schedules / early stopping.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

BACKBONE_PREFIX = "encoder.backbone."
LAYERS_MARKER = BACKBONE_PREFIX + "layers."


def stable_key(sample_id: str) -> str:
    """Plan §50: deterministic probe ordering key = SHA-256 of the sample id."""

    return hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()


def select_source_retention_probe(store, animal_tasks, per_task: int = 16) -> dict[str, list[str]]:
    """Plan §49-§50: from EACH animal task's TRAIN split take the first
    `per_task` molecules ordered by stable SHA(sample_id).  Never validation,
    never label/prediction-dependent."""

    if per_task <= 0:
        raise ValueError(f"per_task must be > 0, found {per_task!r}")
    probe: dict[str, list[str]] = {}
    for task in animal_tasks:
        indices = store.get_task_indices(task, split="train")
        candidates = sorted(
            (str(store.sample_ids[int(index)]) for index in indices), key=stable_key
        )
        chosen = candidates[:per_task]
        if chosen:
            probe[task] = chosen
    if not probe:
        raise ValueError("source retention probe selection found no train molecules")
    return probe


def build_probe_manifest(probe: dict[str, list[str]]) -> dict:
    """Plan §51: serialised probe identity with a canonical manifest hash."""

    tasks = {
        task: {"sample_ids": list(ids), "probe_n": len(ids)}
        for task, ids in sorted(probe.items())
    }
    canonical = json.dumps(tasks, sort_keys=True).encode("utf-8")
    return {
        "tasks": tasks,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "probe_total": sum(entry["probe_n"] for entry in tasks.values()),
    }


def collect_probe_items(store, animal_tasks, probe: dict[str, list[str]], *, max_nodes=None):
    """Collect the exact dataset items for the probe ids (per task), keeping the
    (task, item) association so each batch is scored with its own decoder."""

    dataset_cache: dict[str, object] = {}
    pairs: list[tuple[str, object]] = []
    for task in animal_tasks:
        ids = probe.get(task)
        if not ids:
            continue
        if task not in dataset_cache:
            from toxacute_datastore import ToxAcuteTaskDataset

            dataset_cache[task] = ToxAcuteTaskDataset(
                store, task, split="train", max_nodes=max_nodes
            )
        dataset = dataset_cache[task]
        position_by_id = {}
        for index in range(len(dataset)):
            position_by_id.setdefault(dataset.get_sample_id(index), index)
        for sample_id in ids:
            if sample_id not in position_by_id:
                raise RuntimeError(
                    f"retention probe id {sample_id!r} not found in train split of {task!r}"
                )
            pairs.append((task, dataset[position_by_id[sample_id]]))
    return pairs


def collate_probe_batches(pairs, collate_fn, batch_size: int = 64):
    """Collate per task so every batch maps to exactly one decoder."""

    batches = []
    by_task: dict[str, list[object]] = {}
    for task, item in pairs:
        by_task.setdefault(task, []).append(item)
    for task in sorted(by_task):
        items = by_task[task]
        for start in range(0, len(items), batch_size):
            batches.append((task, collate_fn(items[start : start + batch_size])))
    return batches


def macro_rmse(values: list[float]) -> float:
    import math

    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        raise ValueError("macro RMSE over an empty task set")
    return sum(finite) / len(finite)


def evaluate_animal_rmse(model, task_batches, device, task_scalers, animal_tasks) -> tuple[float, dict[str, float]]:
    """Plan §21-§23: current backbone + ORIGINAL animal heads, per-task median
    RMSE in raw label units, then the 56-task macro.  Decode mirrors
    Trainer.decode_task_output (quantile median + per-task scaler)."""

    from architecture.prediction_heads import decode_prediction

    was_training = model.training
    model.eval()
    per_task: dict[str, float] = {}
    try:
        with torch.no_grad():
            for task, batch in task_batches:
                batch = batch.to(device)
                representation = model.encoder(batch)
                raw = model.decoders[task](representation)
                median = decode_prediction(raw, mode="quantile").median
                scaler = task_scalers.get(task) or {}
                median = (
                    median * float(scaler.get("std", 1.0)) + float(scaler.get("mean", 0.0))
                )
                target = batch.y.reshape(-1, 1).float()
                if target.shape != median.shape:
                    target = target.reshape(median.shape)
                rmse = float(torch.sqrt(((median - target) ** 2).mean()))
                per_task[task] = rmse
    finally:
        if was_training:
            model.train()
    for task in animal_tasks:
        per_task.setdefault(task, float("nan"))
    ordered = [per_task[task] for task in animal_tasks]
    return macro_rmse(ordered), per_task


def build_hybrid_model(base_animal_model, current_backbone_state: dict):
    """Plan §21: keep the ORIGINAL frozen animal heads, swap in the CURRENT
    fine-tuned backbone.  Returns a model whose state overlays the current
    backbone tensors onto the animal-head baseline."""

    import copy

    hybrid = copy.deepcopy(base_animal_model)
    state = hybrid.state_dict()
    overwritten = []
    for key, value in state.items():
        if key.startswith(BACKBONE_PREFIX) and key in current_backbone_state:
            state[key] = current_backbone_state[key].detach().cpu().to(value.dtype)
            overwritten.append(key)
    if not overwritten:
        raise RuntimeError("hybrid model: no backbone tensors were replaced")
    hybrid.load_state_dict(state)
    return hybrid


def functional_forgetting_stats(
    teacher_task_rmse: dict[str, float],
    current_task_rmse: dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> dict:
    """Plan §23-§24/§27-§28: macro forgetting (abs/relative), per-endpoint
    deltas and the source-task forgetting distribution."""

    import math

    tasks = sorted(set(teacher_task_rmse) & set(current_task_rmse))
    if not tasks:
        raise ValueError("functional forgetting: no overlapping tasks")
    deltas = {task: current_task_rmse[task] - teacher_task_rmse[task] for task in tasks}
    teacher_macro = macro_rmse([teacher_task_rmse[task] for task in tasks])
    current_macro = macro_rmse([current_task_rmse[task] for task in tasks])
    abs_forgetting = current_macro - teacher_macro
    sorted_deltas = sorted(deltas.values())
    n = len(sorted_deltas)

    def quantile(q: float) -> float:
        position = min(int(q * (n - 1)), n - 1)
        return sorted_deltas[position]

    improved = sum(1 for value in deltas.values() if value < -tolerance)
    worsened = sum(1 for value in deltas.values() if value > tolerance)
    return {
        "teacher_animal56_macro_rmse": teacher_macro,
        "current_animal56_macro_rmse": current_macro,
        "functional_forgetting_abs": abs_forgetting,
        "functional_forgetting_relative": abs_forgetting / (teacher_macro + 1e-12),
        "functional_forgetting_exact_zero": bool(abs(abs_forgetting) < 1e-7),
        "animal_tasks_improved": improved,
        "animal_tasks_unchanged": n - improved - worsened,
        "animal_tasks_worsened": worsened,
        "delta_q25": quantile(0.25),
        "delta_median": quantile(0.50),
        "delta_q75": quantile(0.75),
        "delta_q90": quantile(0.90),
        "delta_max": sorted_deltas[-1],
        "per_task_delta": deltas,
    }


def freeze_last_block(model) -> int:
    """Plan §55-§56: permanently freeze the last Graphormer block.  Returns the
    number of parameters transitioned to requires_grad=False."""

    layers = model.encoder.backbone.layers
    last_index = len(layers) - 1
    count = 0
    for name, parameter in model.named_parameters():
        if name.startswith(LAYERS_MARKER):
            try:
                layer_index = int(name.split(".")[3])
            except (IndexError, ValueError):
                continue
            if layer_index == last_index and parameter.requires_grad:
                parameter.requires_grad = False
                count += 1
    return count


DRIFT_TRIGGER_FIELDS = (
    "epoch",
    "state",
    "probe_rmse_teacher",
    "probe_rmse_current",
    "retention_damage_train",
    "trigger_threshold",
    "backbone_trainable",
    "triggered",
)


class D8RetentionController:
    """O6/SRTA train-time controller (plan §44-§57, §67-§68).

    Attached as ``trainer.d8_retention_controller``.  The controller owns the
    LAST BLOCK's requires_grad from the moment the retention trigger fires;
    the trainer must call ``pre_epoch`` at each epoch start (re-applies the
    triggered freeze) and ``log_epoch`` after each epoch's validation.
    """

    def __init__(
        self,
        *,
        model,
        teacher_model,
        animal_tasks,
        task_scalers,
        device,
        probe_batches,
        output_dir=None,
        threshold: float = 0.02,
    ):
        if threshold <= 0:
            raise ValueError(f"retention threshold must be > 0, found {threshold!r}")
        self.device = device
        self.animal_tasks = list(animal_tasks)
        self.threshold = float(threshold)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._teacher_model = teacher_model
        self.task_scalers = dict(task_scalers)
        self.probe_batches = probe_batches
        self.teacher_probe_rmse, _ = evaluate_animal_rmse(
            teacher_model, probe_batches, device, task_scalers, self.animal_tasks
        )
        self.triggered = False
        self.trigger_epoch = None
        self.rows: list[dict] = []
        # Baseline row before any update.
        self._append_row(-1, "baseline", model)

    def _append_row(self, epoch: int, state: str, model) -> dict:
        backbone_state = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        if self.triggered:
            current_rmse = self.teacher_probe_rmse
            damage = 0.0
            trainable = False
        else:
            hybrid = build_hybrid_model(self._teacher_model_for_hybrid(), backbone_state)
            current_rmse, _ = evaluate_animal_rmse(
                hybrid, self.probe_batches, self.device, self.task_scalers, self.animal_tasks
            )
            damage = current_rmse / (self.teacher_probe_rmse + 1e-12) - 1.0
            trainable = True
        row = {
            "epoch": int(epoch),
            "state": state,
            "probe_rmse_teacher": f"{self.teacher_probe_rmse:.8f}",
            "probe_rmse_current": f"{current_rmse:.8f}",
            "retention_damage_train": f"{damage:.8f}",
            "trigger_threshold": f"{self.threshold:.4f}",
            "backbone_trainable": int(trainable),
            "triggered": int(self.triggered),
        }
        self.rows.append(row)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "d8_retention_trigger.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(DRIFT_TRIGGER_FIELDS))
                writer.writeheader()
                writer.writerows(self.rows)
        return row

    def _teacher_model_for_hybrid(self):
        return self._teacher_model

    def pre_epoch(self, epoch: int, model) -> None:
        """§55/§68: once triggered, the last block stays frozen from the NEXT
        epoch onwards — re-assert every epoch start (idempotent)."""

        if self.triggered:
            freeze_last_block(model)

    def log_epoch(self, epoch: int, model) -> dict:
        row = self._append_row(epoch, "post_epoch", model)
        damage = float(row["retention_damage_train"])
        if not self.triggered and damage > self.threshold:
            self.triggered = True
            self.trigger_epoch = int(epoch)
            frozen = freeze_last_block(model)
            row["backbone_trainable"] = 0
            row["triggered"] = 1
            print(
                f"O6 source-retention trigger: epoch={epoch} damage={damage:.4f} "
                f"> {self.threshold} — last block frozen from next epoch "
                f"({frozen} params)"
            )
            if self.output_dir is not None:
                self._rewrite()
        return row

    def _rewrite(self):
        if self.output_dir is None:
            return
        with (self.output_dir / "d8_retention_trigger.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(DRIFT_TRIGGER_FIELDS))
            writer.writeheader()
            writer.writerows(self.rows)
