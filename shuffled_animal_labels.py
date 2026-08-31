"""Counterfactual shuffled-label provider for D6 Animal56 teachers (plan §61-§63).

Replaces **train-split** labels of the animal endpoints with a fixed,
task-local permutation so that ``Teacher_shuffle`` differs from
``Teacher_real`` only through the loss of real molecule↔label correspondence.
Validation labels stay real (logging only), human labels are never touched.

The permutation is a pure function of (ANIMAL_SHUFFLE_SEED, task name), so the
mapping is identical across model seeds and stages (plan §62).
"""

from __future__ import annotations

import numpy as np

from reproducibility import stable_seed

ANIMAL_SHUFFLE_SEED = 20260831


class ShuffledAnimalTrainLabels:
    """``get_label``-style label provider over one store instance."""

    def __init__(self, store, task_names, shuffle_seed: int = ANIMAL_SHUFFLE_SEED):
        self.store = store
        self.shuffle_seed = int(shuffle_seed)
        self.mappings: dict[str, dict[int, float]] = {}
        for task in task_names:
            indices = store.get_task_indices(task, split="train")
            labels = np.array([store.get_label(int(index), task) for index in indices], dtype=float)
            generator = np.random.default_rng(stable_seed(self.shuffle_seed, task))
            permutation = generator.permutation(labels.size)
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
