"""In-memory auxiliary endpoint labels for routing stress tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from toxacute_datastore import CODE_TO_SPLIT, SPLIT_CODES, ToxAcuteDataStore


@dataclass(frozen=True)
class AuxiliaryTaskSpec:
    task_name: str
    source_task: str
    include_in_macro: bool = False
    fit_conformal: bool = False
    auxiliary_only: bool = True


class ShuffledEndpointOverlay:
    """Provide a fake endpoint using a split-local permutation of one source."""

    def __init__(self, store: ToxAcuteDataStore, source_task: str, *, seed: int = 42, task_name: str | None = None):
        if source_task not in store.task_names:
            raise KeyError(f"Unknown source task: {source_task}")
        self.store = store
        self.source_task = source_task
        self.seed = int(seed)
        self.task_name = task_name or f"shuffled_{source_task}"
        self.spec = AuxiliaryTaskSpec(self.task_name, source_task)
        self._labels: dict[int, float] = {}
        for split_name, split_code in SPLIT_CODES.items():
            indices = store.get_task_indices(source_task, split=split_name)
            if len(indices) == 0:
                continue
            values = np.asarray([store.get_label(int(index), source_task) for index in indices], dtype=np.float32)
            permutation = np.random.default_rng(self.seed + int(split_code)).permutation(len(indices))
            for target_index, source_position in zip(indices, permutation):
                self._labels[int(target_index)] = float(values[int(source_position)])

    @property
    def task_names(self) -> list[str]:
        return [self.task_name]

    @property
    def registry(self) -> dict[str, dict]:
        return {
            self.task_name: {
                "source_task": self.source_task,
                "include_in_macro": False,
                "fit_conformal": False,
                "auxiliary_only": True,
            }
        }

    def get_label(self, global_index: int, task_name: str, *, split: str | None = None) -> float | None:
        if task_name != self.task_name:
            raise KeyError(f"Unknown auxiliary task: {task_name}")
        index = int(global_index)
        if split is not None:
            normalized = "validation" if split == "val" else split
            if normalized not in CODE_TO_SPLIT.values():
                raise ValueError(f"Unknown split: {split}")
            if int(self.store.split_codes[index]) != SPLIT_CODES[normalized]:
                return None
        return self._labels.get(index)

    def metadata_override(self, task_name: str | None = None) -> dict:
        if task_name is not None and task_name != self.task_name:
            raise KeyError(f"Unknown auxiliary task: {task_name}")
        try:
            from architecture.toxacute_tasks import parse_toxacute_task_name

            inherited = asdict(parse_toxacute_task_name(self.source_task))
        except ValueError:
            inherited = {"task_name": self.source_task}
        inherited.update(
            {
                "task_name": self.task_name,
                "source_task": self.source_task,
            }
        )
        return {
            **inherited,
            "include_in_macro": False,
            "fit_conformal": False,
            "auxiliary_only": True,
        }


__all__ = ["AuxiliaryTaskSpec", "ShuffledEndpointOverlay"]
