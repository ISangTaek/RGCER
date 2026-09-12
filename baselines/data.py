"""Frozen DataStore views for training, validation, and authorized inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from split_manifest import load_manifest, manifest_hash
from toxacute_datastore import CODE_TO_SPLIT, SPLIT_CODES, ToxAcuteDataStore

from .constants import (
    DATASTORE_BUILD_ID,
    DATASTORE_FINGERPRINT,
    HUMAN3_TASKS,
    SMOKE_TRAIN_PER_TASK,
    SMOKE_VALIDATION_PER_TASK,
    SPLIT_MANIFEST_HASH,
    TOXACUTE_TASKS,
)
from .inference_authorization import TestInferenceGrant
from .utils import canonical_sha256


class _LazyGraphStoreOwner:
    """Own one lazily opened DataStore shared by related graph views."""

    def __init__(self, datastore: str | Path):
        self.datastore = str(Path(datastore).resolve())
        self._store: ToxAcuteDataStore | None = None

    @property
    def is_open(self) -> bool:
        return self._store is not None

    def get_graph_record(self, global_index: int) -> dict[str, Any]:
        if self._store is None:
            store = ToxAcuteDataStore.resolve(self.datastore)
            try:
                _validate_store(store)
            except Exception:
                store.close()
                raise
            self._store = store
        return self._store.get_graph_record(int(global_index))

    def close(self) -> None:
        if self._store is not None:
            store = self._store
            self._store = None
            store.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GraphRecordView(Sequence):
    """Lazy, read-only graph records so formal AFP does not retain all graphs in RAM."""

    def __init__(self, owner: _LazyGraphStoreOwner, global_indices: np.ndarray):
        self._owner = owner
        self.global_indices = np.asarray(global_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        return self._owner.get_graph_record(int(self.global_indices[int(index)]))

    def close(self) -> None:
        self._owner.close()


@dataclass(frozen=True)
class BaselinePartition:
    split: str
    global_indices: np.ndarray
    raw_row_indices: np.ndarray
    sample_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    labels_all: np.ndarray
    graph_records: Sequence[dict[str, Any]]

    @property
    def labels_human3(self) -> np.ndarray:
        columns = [TOXACUTE_TASKS.index(name) for name in HUMAN3_TASKS]
        return self.labels_all[:, columns]


SmokePartition = BaselinePartition


@dataclass(frozen=True)
class InferenceSampleExpectation:
    """Independently derived sample/label expectation for output verification."""

    split: str
    sample_ids: tuple[str, ...]
    raw_row_indices: np.ndarray
    labels_human3: np.ndarray


@dataclass(frozen=True)
class BaselineData:
    train: BaselinePartition
    validation: BaselinePartition
    selection_by_task: dict[str, dict[str, list[str]]]
    datastore_root: str
    feature_schema_version: str
    sample_identity_hash: str
    mode: str
    training_task_scope: str
    _graph_store_owner: _LazyGraphStoreOwner | None = field(default=None, repr=False, compare=False)

    @property
    def graph_record_mode(self) -> str:
        return "shared_lazy_datastore" if self._graph_store_owner is not None else "materialized_memory"

    def to_manifest(self) -> dict:
        def partition(value: BaselinePartition) -> dict:
            return {
                "split": value.split,
                "count": len(value.sample_ids),
                "sample_ids": list(value.sample_ids),
                "raw_row_indices": value.raw_row_indices.tolist(),
                "global_indices": value.global_indices.tolist(),
            }

        return {
            "mode": self.mode,
            "training_task_scope": self.training_task_scope,
            "graph_record_mode": self.graph_record_mode,
            "train": partition(self.train),
            "validation": partition(self.validation),
            "selection_by_task": self.selection_by_task,
            "sample_identity_hash": self.sample_identity_hash,
            "feature_schema_version": self.feature_schema_version,
        }

    def close(self) -> None:
        if self._graph_store_owner is not None:
            self._graph_store_owner.close()
            return
        for partition in (self.train, self.validation):
            close = getattr(partition.graph_records, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> "BaselineData":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


Human3SmokeData = BaselineData


def _validate_store(store: ToxAcuteDataStore) -> None:
    expected = {
        "build_id": DATASTORE_BUILD_ID,
        "fingerprint": DATASTORE_FINGERPRINT,
        "split_manifest_hash": SPLIT_MANIFEST_HASH,
    }
    actual = {
        "build_id": store.build_id,
        "fingerprint": store.fingerprint,
        "split_manifest_hash": store.split_manifest_hash,
    }
    if actual != expected:
        raise ValueError(f"Frozen DataStore identity mismatch: {actual!r} != {expected!r}")
    if set(store.task_names) != set(TOXACUTE_TASKS) or len(store.task_names) != len(TOXACUTE_TASKS):
        raise ValueError("DataStore task set differs from the canonical 59-task registry")


def _open_validated(datastore: str | Path):
    store = ToxAcuteDataStore.resolve(datastore)
    _validate_store(store)
    manifest = load_manifest(store.root / "split_manifest.json")
    if manifest_hash(manifest) != SPLIT_MANIFEST_HASH:
        store.close()
        raise ValueError("On-disk split manifest hash differs from the frozen protocol")
    manifest_by_id = {str(record["sample_id"]): record for record in manifest["records"]}
    if len(manifest_by_id) != len(manifest["records"]):
        store.close()
        raise ValueError("split manifest contains duplicate sample_id values")
    global_by_id = {str(sample_id): index for index, sample_id in enumerate(store.sample_ids)}
    if len(global_by_id) != len(store.sample_ids):
        store.close()
        raise ValueError("DataStore index contains duplicate sample_id values")
    if set(global_by_id) != set(manifest_by_id):
        store.close()
        raise ValueError("DataStore and split manifest sample identities differ")
    return store, manifest_by_id, global_by_id


def _materialize(
    store,
    manifest_by_id,
    ordered_ids,
    split: str,
    *,
    lazy_graphs: bool,
    graph_store_owner: _LazyGraphStoreOwner | None = None,
) -> BaselinePartition:
    global_by_id = {str(sample_id): index for index, sample_id in enumerate(store.sample_ids)}
    global_indices = np.asarray([global_by_id[sid] for sid in ordered_ids], dtype=np.int64)
    rows = np.asarray([int(manifest_by_id[sid]["row_index"]) for sid in ordered_ids], dtype=np.int64)
    smiles: list[str] = []
    records: list[dict[str, Any]] = []
    for sid, index in zip(ordered_ids, global_indices):
        record = store.get_graph_record(int(index))
        actual_split = CODE_TO_SPLIT[int(store.split_codes[int(index)])]
        if record["sample_id"] != sid or manifest_by_id[sid]["split"] != split or actual_split != split:
            raise ValueError(f"Cross-source sample/split mismatch for {sid}")
        smiles.append(str(record["raw_smiles"]))
        if not lazy_graphs:
            records.append(record)
    canonical_columns = [store.task_index(task) for task in TOXACUTE_TASKS]
    labels = np.asarray(store.labels[np.ix_(global_indices, canonical_columns)], dtype=np.float32).copy()
    if lazy_graphs:
        owner = graph_store_owner or _LazyGraphStoreOwner(store.root)
        graph_records: Sequence[dict[str, Any]] = GraphRecordView(owner, global_indices)
    else:
        graph_records = tuple(records)
    return BaselinePartition(
        split=split,
        global_indices=global_indices,
        raw_row_indices=rows,
        sample_ids=tuple(ordered_ids),
        smiles=tuple(smiles),
        labels_all=labels,
        graph_records=graph_records,
    )


def _ordered_eligible(store, manifest_by_id, split: str, task_names: tuple[str, ...]) -> list[str]:
    columns = [store.task_index(task) for task in task_names]
    selected = []
    for global_index, sample_id_value in enumerate(store.sample_ids):
        sample_id = str(sample_id_value)
        if CODE_TO_SPLIT[int(store.split_codes[global_index])] != split:
            continue
        if np.isfinite(store.labels[global_index, columns]).any():
            selected.append(sample_id)
    return sorted(selected, key=lambda sid: (int(manifest_by_id[sid]["row_index"]), sid))


def load_human3_smoke(datastore: str | Path, *, lazy_graphs: bool = False) -> Human3SmokeData:
    store, manifest_by_id, global_by_id = _open_validated(datastore)
    graph_store_owner = _LazyGraphStoreOwner(store.root) if lazy_graphs else None
    try:
        selection: dict[str, dict[str, list[str]]] = {task: {} for task in HUMAN3_TASKS}
        unions: dict[str, set[str]] = {"train": set(), "validation": set()}
        for split, limit in (("train", SMOKE_TRAIN_PER_TASK), ("validation", SMOKE_VALIDATION_PER_TASK)):
            for task in HUMAN3_TASKS:
                task_index = store.task_index(task)
                candidates = []
                for sample_id, record in manifest_by_id.items():
                    global_index = global_by_id[sample_id]
                    if record["split"] == split and np.isfinite(store.labels[global_index, task_index]):
                        candidates.append((int(record["row_index"]), sample_id))
                candidates.sort(key=lambda pair: (pair[0], pair[1]))
                chosen = [sample_id for _, sample_id in candidates[:limit]]
                if len(chosen) != limit:
                    raise ValueError(f"Human3 smoke requires {limit} valid {split} rows for {task}, found {len(chosen)}")
                selection[task][split] = chosen
                unions[split].update(chosen)
        ordered_train = sorted(unions["train"], key=lambda sid: (int(manifest_by_id[sid]["row_index"]), sid))
        ordered_validation = sorted(unions["validation"], key=lambda sid: (int(manifest_by_id[sid]["row_index"]), sid))
        train = _materialize(
            store,
            manifest_by_id,
            ordered_train,
            "train",
            lazy_graphs=lazy_graphs,
            graph_store_owner=graph_store_owner,
        )
        validation = _materialize(
            store,
            manifest_by_id,
            ordered_validation,
            "validation",
            lazy_graphs=lazy_graphs,
            graph_store_owner=graph_store_owner,
        )
        if len(train.sample_ids) > 96 or len(validation.sample_ids) > 24:
            raise AssertionError("Human3 union exceeded its protocol maximum")
        identity_payload = {"train": list(train.sample_ids), "validation": list(validation.sample_ids), "selection_by_task": selection}
        return BaselineData(
            train=train,
            validation=validation,
            selection_by_task=selection,
            datastore_root=str(store.root),
            feature_schema_version=str(store.metadata["feature_schema_version"]),
            sample_identity_hash=canonical_sha256(identity_payload),
            mode="human3_smoke",
            training_task_scope="human3_selected_rows_joint59_labels",
            _graph_store_owner=graph_store_owner,
        )
    except Exception:
        if graph_store_owner is not None:
            graph_store_owner.close()
        raise
    finally:
        store.close()


def load_formal_train_validation(datastore: str | Path, method: str) -> BaselineData:
    if method not in {"rf", "afp", "dmpnn", "grover", "toxacol"}:
        raise ValueError(f"Unknown baseline method: {method}")
    store, manifest_by_id, _ = _open_validated(datastore)
    graph_store_owner = _LazyGraphStoreOwner(store.root)
    try:
        train_tasks = TOXACUTE_TASKS if method == "toxacol" else HUMAN3_TASKS
        train_ids = _ordered_eligible(store, manifest_by_id, "train", train_tasks)
        validation_ids = _ordered_eligible(store, manifest_by_id, "validation", HUMAN3_TASKS)
        train = _materialize(
            store,
            manifest_by_id,
            train_ids,
            "train",
            lazy_graphs=True,
            graph_store_owner=graph_store_owner,
        )
        validation = _materialize(
            store,
            manifest_by_id,
            validation_ids,
            "validation",
            lazy_graphs=True,
            graph_store_owner=graph_store_owner,
        )
        for task in HUMAN3_TASKS:
            column = TOXACUTE_TASKS.index(task)
            if not np.isfinite(train.labels_all[:, column]).any() or not np.isfinite(validation.labels_all[:, column]).any():
                raise ValueError(f"Formal train/validation lacks finite labels for {task}")
        identity_payload = {"train": list(train.sample_ids), "validation": list(validation.sample_ids), "training_task_scope": "joint59" if method == "toxacol" else "human3"}
        return BaselineData(
            train=train,
            validation=validation,
            selection_by_task={},
            datastore_root=str(store.root),
            feature_schema_version=str(store.metadata["feature_schema_version"]),
            sample_identity_hash=canonical_sha256(identity_payload),
            mode="formal_train_validation",
            training_task_scope="joint59" if method == "toxacol" else "human3",
            _graph_store_owner=graph_store_owner,
        )
    except Exception:
        graph_store_owner.close()
        raise
    finally:
        store.close()


def _require_inference_split_permission(
    split: str,
    method: str,
    test_grant: TestInferenceGrant | None,
) -> None:
    if split == "calibration":
        raise PermissionError("Calibration prediction remains locked in Stage 3C2A")
    if split == "test":
        if test_grant is None:
            raise PermissionError("Test prediction requires a fully validated authorization grant")
        test_grant.require(split=split, method=method)
        return
    if split != "validation":
        raise ValueError(f"Unknown inference split: {split!r}")


def load_authorized_inference_split(
    datastore: str | Path,
    method: str,
    *,
    split: str = "validation",
    test_grant: TestInferenceGrant | None = None,
) -> BaselinePartition:
    if method not in {"rf", "afp", "dmpnn", "grover", "toxacol"}:
        raise ValueError(f"Unknown baseline method: {method}")
    _require_inference_split_permission(split, method, test_grant)
    store, manifest_by_id, _ = _open_validated(datastore)
    graph_store_owner = _LazyGraphStoreOwner(store.root)
    try:
        ids = _ordered_eligible(store, manifest_by_id, split, HUMAN3_TASKS)
        return _materialize(
            store,
            manifest_by_id,
            ids,
            split,
            lazy_graphs=True,
            graph_store_owner=graph_store_owner,
        )
    except Exception:
        graph_store_owner.close()
        raise
    finally:
        store.close()


def derive_authorized_sample_expectation(
    datastore: str | Path,
    method: str,
    *,
    split: str = "validation",
    test_grant: TestInferenceGrant | None = None,
) -> InferenceSampleExpectation:
    """Derive expected inference samples without reusing the output loader filter."""

    if method not in {"rf", "afp", "dmpnn", "grover", "toxacol"}:
        raise ValueError(f"Unknown baseline method: {method}")
    _require_inference_split_permission(split, method, test_grant)
    store, manifest_by_id, global_by_id = _open_validated(datastore)
    try:
        human_columns = [store.task_index(task) for task in HUMAN3_TASKS]
        selected: list[tuple[int, str, np.ndarray]] = []
        for sample_id, manifest_record in manifest_by_id.items():
            global_index = global_by_id[sample_id]
            manifest_split = str(manifest_record["split"])
            datastore_split = CODE_TO_SPLIT[int(store.split_codes[global_index])]
            if datastore_split != manifest_split:
                raise ValueError(f"Cross-source sample/split mismatch for {sample_id}")
            if manifest_split != split:
                continue
            labels = np.asarray(store.labels[global_index, human_columns], dtype=np.float32).copy()
            if np.isfinite(labels).any():
                selected.append((int(manifest_record["row_index"]), sample_id, labels))
        selected.sort(key=lambda row: (row[0], row[1]))
        return InferenceSampleExpectation(
            split=split,
            sample_ids=tuple(row[1] for row in selected),
            raw_row_indices=np.asarray([row[0] for row in selected], dtype=np.int64),
            labels_human3=np.asarray([row[2] for row in selected], dtype=np.float32).reshape(-1, len(HUMAN3_TASKS)),
        )
    finally:
        store.close()


def load_authorized_validation(datastore: str | Path, method: str, *, split: str = "validation") -> BaselinePartition:
    """Backward-compatible validation-only entry point."""

    if split != "validation":
        raise PermissionError("Only validation prediction is authorized; calibration and test remain locked")
    return load_authorized_inference_split(datastore, method, split=split)
