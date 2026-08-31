"""Counterfactual shuffled-label provider for D6 Animal56 teachers (plan §61-§63).

Replaces **train-split** labels of the animal endpoints with a fixed,
task-local permutation so that ``Teacher_shuffle`` differs from
``Teacher_real`` only through the loss of real molecule↔label correspondence.
Validation labels stay real (logging only), human labels are never touched.

The permutation is a pure function of (ANIMAL_SHUFFLE_SEED, task name), so the
mapping is identical across model seeds and stages (plan §62).  The full
mapping is retained and hashable (P1-5): ``mapping_sha256`` proves that every
model seed used the same counterfactual alignment, and ``sanity_rows`` records
the per-task label-multiset invariants.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from reproducibility import stable_seed

ANIMAL_SHUFFLE_SEED = 20260831


class ShuffledAnimalTrainLabels:
    """``get_label``-style label provider over one store instance."""

    def __init__(self, store, task_names, shuffle_seed: int = ANIMAL_SHUFFLE_SEED):
        self.store = store
        self.shuffle_seed = int(shuffle_seed)
        self.permutations: dict[str, dict[int, int]] = {}
        self.mappings: dict[str, dict[int, float]] = {}
        self.original_labels: dict[str, dict[int, float]] = {}
        for task in task_names:
            indices = store.get_task_indices(task, split="train")
            labels = np.array([store.get_label(int(index), task) for index in indices], dtype=float)
            generator = np.random.default_rng(stable_seed(self.shuffle_seed, task))
            permutation = generator.permutation(labels.size)
            self.permutations[task] = {
                int(index): int(indices[position])
                for index, position in zip(indices, permutation)
            }
            self.original_labels[task] = {
                int(index): float(value) for index, value in zip(indices, labels)
            }
            self.mappings[task] = {
                int(index): float(labels[position])
                for index, position in zip(indices, permutation)
            }

    def get_label(self, global_index, task, split=None):
        if split == "train" and task in self.mappings:
            value = self.mappings[task].get(int(global_index))
            return None if value is None else float(value)
        if task not in self.mappings:
            return None  # human/unknown endpoints never receive teacher labels
        # validation (logging only) and any other split: real labels.
        return self.store.get_label(int(global_index), task)

    def mapping_sha256(self) -> str:
        """Stable hash over (seed, task, target index, source index) pairs (P1-5)."""

        digest = hashlib.sha256()
        digest.update(json.dumps({"shuffle_seed": self.shuffle_seed}, sort_keys=True).encode("utf-8"))
        for task in sorted(self.permutations):
            digest.update(task.encode("utf-8"))
            mapping = self.permutations[task]
            for target_index in sorted(mapping):
                entry = {
                    "task": task,
                    "target_global_index": int(target_index),
                    "source_global_index": int(mapping[target_index]),
                }
                digest.update(
                    json.dumps(entry, sort_keys=True).encode("utf-8")
                )
        return digest.hexdigest()

    def sanity_rows(self) -> list[dict]:
        """Per-task invariants: multiset equality plus mean/std before/after."""

        rows = []
        for task, mapping in self.mappings.items():
            original = self.original_labels[task]
            before = np.array([original[index] for index in sorted(original)])
            after = np.array([mapping[index] for index in sorted(mapping)])
            rows.append(
                {
                    "task": task,
                    "n": len(mapping),
                    "mean_before": float(before.mean()),
                    "mean_after": float(after.mean()),
                    "std_before": float(before.std()),
                    "std_after": float(after.std()),
                    "label_multiset_equal": bool(
                        np.array_equal(np.sort(before), np.sort(after))
                    ),
                    "mapping_hash": self._task_mapping_hash(task),
                }
            )
        return rows

    def _task_mapping_hash(self, task: str) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps({"shuffle_seed": self.shuffle_seed}, sort_keys=True).encode("utf-8"))
        digest.update(task.encode("utf-8"))
        mapping = self.permutations[task]
        for target_index in sorted(mapping):
            digest.update(
                json.dumps(
                    {
                        "target_global_index": int(target_index),
                        "source_global_index": int(mapping[target_index]),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
        return digest.hexdigest()
