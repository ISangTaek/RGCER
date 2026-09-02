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


def select_source_retention_probe(
    store, animal_tasks, per_task: int = 16, *, max_nodes=None
) -> dict[str, list[str]]:
    """Plan §49-§50 + eighth-review P1-1 (§48-§52): from EACH animal task's
    TRAIN split take the first `per_task` molecules ordered by stable
    SHA(sample_id).  `max_nodes` mirrors the formal loader filter so the probe
    selection universe == the trainable universe (otherwise a >max_nodes
    molecule could be selected and then dropped at collection time).  Never
    validation, never label/prediction-dependent."""

    if per_task <= 0:
        raise ValueError(f"per_task must be > 0, found {per_task!r}")
    probe: dict[str, list[str]] = {}
    for task in animal_tasks:
        indices = store.get_task_indices(task, split="train", max_nodes=max_nodes)
        candidates = sorted(
            (str(store.sample_ids[int(index)]) for index in indices), key=stable_key
        )
        chosen = candidates[:per_task]
        if not chosen:
            raise RuntimeError(
                f"O6 source retention probe has no train samples for {task!r}"
            )
        probe[task] = chosen
    if set(probe) != set(animal_tasks):
        missing = sorted(set(animal_tasks) - set(probe))
        raise RuntimeError(
            f"O6 source retention probe is missing source tasks: {missing[:5]}"
        )
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
    """Collect the exact graph items for the probe ids (per task) directly from
    the store, keeping the (task, item) association so each batch is scored
    with its own decoder.  Uses the store's OWN env cache (same instance as
    the training loaders) — avoids the LMDB double-open that a fresh
    ToxAcuteTaskDataset would trigger (sixth-D8 server finding)."""

    wanted: dict[str, set[str]] = {task: set(ids) for task, ids in probe.items() if ids}
    pairs: list[tuple[str, object]] = []
    seen: set[str] = set()
    for task in animal_tasks:
        ids = wanted.get(task)
        if not ids:
            continue
        indices = store.get_task_indices(task, split="train", max_nodes=max_nodes)
        for index in indices:
            sample_id = str(store.sample_ids[int(index)])
            if sample_id not in ids or sample_id in seen:
                continue
            label = store.get_label(int(index), task)
            if label is None:
                continue
            item = store.get_graph_data(
                int(index), task_name=task, label=float(label)
            )
            pairs.append((task, item))
            seen.add(sample_id)
    all_wanted = set()
    for ids in wanted.values():
        all_wanted |= ids
    missing = sorted(all_wanted - seen)
    if missing:
        raise RuntimeError(
            f"O6 retention probe set incomplete: missing {len(missing)} ids; "
            f"examples={missing[:3]}"
        )
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
    # Sixth-D8 review P0-1: an endpoint usually spans several validation
    # batches — aggregate by SUMMED SQUARED ERROR / COUNT so the reported
    # RMSE covers the whole endpoint set, never "the last batch".
    sum_sq_error = {task: 0.0 for task in animal_tasks}
    count = {task: 0 for task in animal_tasks}
    try:
        with torch.no_grad():
            for task, batch in task_batches:
                batch = batch.to(device)
                try:
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
                    error = median - target
                    sum_sq_error[task] += float(error.pow(2).sum().detach().cpu())
                    count[task] += int(error.numel())
                finally:
                    # PyG `.to()` is in-place: move the batch back so the full
                    # validation set never resides on the GPU at once (P1-3).
                    if getattr(device, "type", None) == "cuda":
                        batch.cpu()
    finally:
        if was_training:
            model.train()
    missing_tasks = [task for task in animal_tasks if count[task] <= 0]
    if missing_tasks:
        raise RuntimeError(
            "Animal56 functional evaluation is incomplete; "
            f"missing {len(missing_tasks)} tasks: {missing_tasks[:5]}"
        )
    per_task: dict[str, float] = {}
    for task in animal_tasks:
        per_task[task] = (sum_sq_error[task] / count[task]) ** 0.5
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
    expected_task_set=None,
) -> dict:
    """Plan §23-§24/§27-§28 + P1-2 §55: macro forgetting (abs/relative),
    per-endpoint deltas and the source-task forgetting distribution.  The
    teacher/current task sets must be IDENTICAL; formal D8 additionally pins
    them to the full 56-task Animal56 set via ``expected_task_set``."""

    import math

    teacher_tasks = set(teacher_task_rmse)
    current_tasks = set(current_task_rmse)
    if teacher_tasks != current_tasks:
        raise ValueError(
            "teacher/current animal task sets differ: "
            f"only-teacher={sorted(teacher_tasks - current_tasks)[:5]} "
            f"only-current={sorted(current_tasks - teacher_tasks)[:5]}"
        )
    tasks = sorted(teacher_tasks)
    if expected_task_set is not None and tasks != sorted(set(expected_task_set)):
        raise ValueError(
            "functional forgetting task set does not match the formal Animal56 "
            f"task set ({len(tasks)} vs {len(set(expected_task_set))})"
        )
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
    # Eighth-review P1-2 (§53-§56): a macro delta can cancel +0.01/-0.01
    # across endpoints — the exact-zero invariant must hold at the ENDPOINT
    # level (all 56 per-task deltas ~0), not just in the macro.
    max_abs_task_delta = max(abs(value) for value in deltas.values())
    return {
        "teacher_animal56_macro_rmse": teacher_macro,
        "current_animal56_macro_rmse": current_macro,
        "functional_forgetting_abs": abs_forgetting,
        "functional_forgetting_relative": abs_forgetting / (teacher_macro + 1e-12),
        "functional_forgetting_exact_zero": bool(max_abs_task_delta < 1e-7),
        "delta_max_abs": max_abs_task_delta,
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
    "trigger_fired_this_epoch",
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
        # Fifth-D8-review P1-1 (§36-§39): the trigger only stops FUTURE
        # updates — the already-drifted last block keeps its real source
        # damage.  Always evaluate the hybrid; never write a synthetic 0.
        hybrid = build_hybrid_model(self._teacher_model_for_hybrid(), backbone_state)
        current_rmse, _ = evaluate_animal_rmse(
            hybrid, self.probe_batches, self.device, self.task_scalers, self.animal_tasks
        )
        damage = current_rmse / (self.teacher_probe_rmse + 1e-12) - 1.0
        trainable = not self.triggered
        row = {
            "epoch": int(epoch),
            "state": state,
            "probe_rmse_teacher": f"{self.teacher_probe_rmse:.8f}",
            "probe_rmse_current": f"{current_rmse:.8f}",
            "retention_damage_train": f"{damage:.8f}",
            "trigger_threshold": f"{self.threshold:.4f}",
            "backbone_trainable": int(trainable),
            "triggered": int(self.triggered),
            "trigger_fired_this_epoch": 0,
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

    def state_dict(self) -> dict:
        """P1-2 (§44): trigger state survives job interruption via the
        checkpoint payload."""

        return {
            "triggered": bool(self.triggered),
            "trigger_epoch": self.trigger_epoch,
            "rows": list(self.rows),
            "threshold": float(self.threshold),
            "teacher_probe_rmse": float(self.teacher_probe_rmse),
        }

    def load_state_dict(self, state: dict) -> None:
        if state is None:
            return
        if float(state["threshold"]) != float(self.threshold):
            raise ValueError(
                f"retention threshold mismatch: checkpoint has "
                f"{state['threshold']!r}, controller runs {self.threshold!r}"
            )
        self.triggered = bool(state["triggered"])
        self.trigger_epoch = state.get("trigger_epoch")
        self.rows = list(state.get("rows") or [])
        if self.triggered:
            print(
                f"O6 retention state restored from checkpoint: triggered at "
                f"epoch {self.trigger_epoch} — last block stays frozen"
            )

    def pre_epoch(self, epoch: int, model) -> None:
        """§55/§68: once triggered, the last block stays frozen from the NEXT
        epoch onwards — re-assert every epoch start (idempotent)."""

        if self.triggered:
            freeze_last_block(model)

    def log_epoch(self, epoch: int, model) -> dict:
        was_triggered = self.triggered
        row = self._append_row(epoch, "post_epoch", model)
        damage = float(row["retention_damage_train"])
        if not was_triggered and damage > self.threshold:
            self.triggered = True
            self.trigger_epoch = int(epoch)
            frozen = freeze_last_block(model)
            # P1-4 (§64): keep the firing epoch visible in the CSV — the row's
            # damage is real and the freeze takes effect from the NEXT epoch.
            row["triggered"] = 1
            row["backbone_trainable"] = 0
            row["trigger_fired_this_epoch"] = 1
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
