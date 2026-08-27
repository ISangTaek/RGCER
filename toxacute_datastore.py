"""Versioned, task-independent ToxAcute DataStore V2.

The store keeps one Graphormer graph per valid raw row in LMDB and keeps the
endpoint labels in a dense ``labels.npy`` matrix.  Dataset views only hold
integer row indices, so split membership and sample IDs never require graph
deserialization.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import os
import shutil
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from molecular_features import FEATURE_SCHEMA_VERSION
from split_manifest import (
    MANIFEST_VERSION,
    SPLIT_NAMES,
    create_split_manifest,
    load_manifest,
    manifest_hash,
    validate_manifest,
    write_manifest,
)


DATASTORE_FORMAT = "toxacute_datastore"
DATASTORE_FORMAT_VERSION = 2
GRAPH_RECORD_VERSION = 1
SPLIT_CODES = {"train": 0, "validation": 1, "calibration": 2, "test": 3}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_CODES.items()}
REQUIRED_INDEX_FIELDS = ("sample_ids", "row_indices", "split_codes", "num_nodes")

# Graph payloads live in several bounded LMDB shards instead of one huge
# data.mdb (ToxAcute's ~80k molecules serialize to tens of GB).  The shard
# table in datastore.json lists ``{name, first_index, entries}`` rows that
# must tile [0, num_samples) contiguously; readers open a shard lazily on
# first touch so DataLoader workers never map files they do not need.
GRAPH_LAYOUT_NAME = "sharded_lmdb_v1"
GRAPH_SHARD_DIRNAME = "graph_shards"
SHARD_NAME_TEMPLATE = "shard_{:05d}"
DEFAULT_GRAPH_SHARD_MAX_GB = 4.0
_LMDB_OVERHEAD_BYTES = 256 * 1024 * 1024


def graph_key(index: int) -> bytes:
    """Encode a global row index as a compact, sortable LMDB key."""

    if int(index) < 0:
        raise ValueError("graph index must be non-negative")
    return struct.pack(">Q", int(index))


def shard_name(index: int) -> str:
    return SHARD_NAME_TEMPLATE.format(int(index))


def normalize_shard_table(shards: Sequence[Mapping]) -> list[dict]:
    """Validate the shard table shape and contiguity used by readers."""

    table = []
    cursor = 0
    for position, entry in enumerate(shards):
        try:
            name = str(entry["name"])
            first_index = int(entry["first_index"])
            entries = int(entry["entries"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid graph shard descriptor at position {position}") from exc
        if entries <= 0:
            raise ValueError(f"Graph shard {name!r} declares no entries")
        if first_index != cursor:
            raise ValueError(
                f"Graph shard {name!r} starts at {first_index} but {cursor} was expected"
            )
        expected_name = shard_name(position)
        if Path(name).name != name or name != expected_name:
            raise ValueError(f"Unexpected graph shard name {name!r}; expected {expected_name!r}")
        table.append({"name": name, "first_index": first_index, "entries": entries})
        cursor += entries
    if not table:
        raise ValueError("DataStore declares an empty graph shard table")
    return table


class _GraphShardWriter:
    """Append-only helper writing sequential graph keys into one shard env."""

    def __init__(
        self,
        directory: Path,
        *,
        max_bytes: int,
        initial_map_size: int,
        pending_flush_bytes: int = 128 * 1024 * 1024,
    ):
        self.directory = Path(directory)
        # No artificial floor here: tests exercise sub-megabyte caps, and
        # callers validate graph_shard_max_gb at the CLI boundary.
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("shard max_bytes must be positive")
        # Large molecules serialize to many MB each; a fixed record-count
        # batch alone could pile gigabytes into RAM before the first commit.
        self.pending_flush_bytes = int(pending_flush_bytes)
        if self.pending_flush_bytes <= 0:
            raise ValueError("pending_flush_bytes must be positive")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.env = lmdb.open(
            str(self.directory),
            map_size=max(8 << 20, int(initial_map_size)),
            subdir=True,
            readonly=False,
            lock=True,
            meminit=False,
        )
        self.entries = 0
        self.payload_bytes = 0
        self._pending: list[tuple[bytes, bytes]] = []
        self._pending_bytes = 0

    @property
    def name(self) -> str:
        return self.directory.name

    def full_for(self, payload_size: int) -> bool:
        """Rollover when adding this payload would push past the byte cap."""

        return self.entries > 0 and self.payload_bytes + int(payload_size) > self.max_bytes

    def add(self, key: bytes, payload: bytes, *, commit_every: int) -> None:
        self._pending.append((key, payload))
        self.entries += 1
        self.payload_bytes += len(payload)
        self._pending_bytes += len(payload)
        if (
            len(self._pending) >= max(1, int(commit_every))
            or self._pending_bytes >= self.pending_flush_bytes
        ):
            self._commit_pending()

    def _commit_batch(self, batch: Sequence[tuple[bytes, bytes]]) -> None:
        while True:
            transaction = self.env.begin(write=True)
            try:
                for key, payload in batch:
                    transaction.put(key, payload, overwrite=False)
                transaction.commit()
                return
            except lmdb.MapFullError:
                transaction.abort()
                current_size = int(self.env.info()["map_size"])
                self.env.set_mapsize(max(current_size * 2, current_size + 1))

    def _commit_pending(self) -> None:
        if self._pending:
            self._commit_batch(self._pending)
            self._pending.clear()
            self._pending_bytes = 0

    def flush_and_close(self) -> dict:
        self._commit_pending()
        self.env.sync()
        self.env.close()
        return {"name": self.name, "entries": self.entries, "payload_bytes": self.payload_bytes}


def serialize_graph_record(record: Mapping) -> bytes:
    """Serialize a plain graph dictionary without PyG class identity."""

    buffer = io.BytesIO()
    torch.save(dict(record), buffer)
    return buffer.getvalue()


def deserialize_graph_record(payload: bytes) -> dict:
    """Deserialize one graph record onto CPU."""

    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def _canonical_json(payload: Mapping) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_datastore_fingerprint(
    *,
    raw_csv_sha256: str,
    feature_schema_version: str,
    graph_record_version: int,
    max_path_distance: int,
    task_names: Sequence[str],
    split_manifest_hash: str,
) -> str:
    """Return a path-independent identity for the data artifact."""

    payload = {
        "format_version": DATASTORE_FORMAT_VERSION,
        "raw_csv_sha256": str(raw_csv_sha256),
        "feature_schema_version": str(feature_schema_version),
        "graph_record_version": int(graph_record_version),
        "max_path_distance": int(max_path_distance),
        "task_names": list(task_names),
        "split_manifest_hash": str(split_manifest_hash),
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True)
class DataStoreContext:
    """Small immutable identity object suitable for run metadata."""

    root: str
    build_id: str
    fingerprint: str
    split_manifest_hash: str
    raw_csv_sha256: str
    feature_schema_version: str
    max_path_distance: int
    task_names: tuple[str, ...]


class ToxAcuteDataStore:
    """Read-only access to one validated DataStore V2 build."""

    def __init__(self, root: str | Path, *, validate_on_open: bool = True):
        self.root = Path(root).resolve()
        self._envs: dict[str, Any] = {}
        self._metadata = self._read_json(self.root / "datastore.json")
        if self._metadata.get("graph_layout") != GRAPH_LAYOUT_NAME:
            raise ValueError(
                "DataStore build does not use the sharded graph layout "
                f"({GRAPH_LAYOUT_NAME!r} expected, found "
                f"{self._metadata.get('graph_layout')!r}). Legacy single-file "
                "builds are no longer supported — re-run preprocess_data.py "
                "--build_datastore_v2 to rebuild."
            )
        self.shard_table = normalize_shard_table(self._metadata.get("graph_shards", []))
        self.shard_first_indices = [int(entry["first_index"]) for entry in self.shard_table]
        self.graph_shard_max_bytes = int(self._metadata.get("graph_shard_max_bytes", 0))
        self._labels = np.load(self.root / "labels.npy", mmap_mode="r")
        with np.load(self.root / "index.npz", allow_pickle=False) as index:
            missing = [name for name in REQUIRED_INDEX_FIELDS if name not in index]
            if missing:
                raise ValueError(f"index.npz is missing fields: {missing}")
            self.sample_ids = np.asarray(index["sample_ids"])
            self.row_indices = np.asarray(index["row_indices"], dtype=np.int64)
            self.split_codes = np.asarray(index["split_codes"], dtype=np.int8)
            self.num_nodes = np.asarray(index["num_nodes"], dtype=np.int64)
        if validate_on_open:
            self.validate(strict=False)

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"DataStore metadata not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON metadata: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object in {path}")
        return payload

    @classmethod
    def resolve(cls, root_or_build: str | Path) -> "ToxAcuteDataStore":
        """Resolve either a concrete build directory or a root/CURRENT pair."""

        root = Path(root_or_build).resolve()
        if (root / "datastore.json").exists():
            return cls(root)
        current = root / "CURRENT"
        if not current.exists():
            raise FileNotFoundError(
                f"DataStore V2 root has no datastore.json or CURRENT: {root}"
            )
        build_id = current.read_text(encoding="utf-8").strip()
        if not build_id or Path(build_id).name != build_id:
            raise ValueError(f"Invalid CURRENT build id: {build_id!r}")
        build_root = root / "builds" / build_id
        if not (build_root / "READY").exists():
            raise ValueError(f"CURRENT points to a non-READY build: {build_root}")
        return cls(build_root)

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def task_names(self) -> list[str]:
        return list(self._metadata.get("task_names", []))

    @property
    def build_id(self) -> str:
        return str(self._metadata.get("build_id", self.root.name))

    @property
    def fingerprint(self) -> str:
        return str(self._metadata.get("datastore_fingerprint", ""))

    @property
    def split_manifest_hash(self) -> str:
        return str(self._metadata.get("split_manifest_hash", ""))

    @property
    def raw_csv_sha256(self) -> str:
        return str(self._metadata.get("raw_csv_sha256", ""))

    @property
    def graph_shards_root(self) -> Path:
        return self.root / GRAPH_SHARD_DIRNAME

    def shard_directory(self, name: str) -> Path:
        return self.graph_shards_root / name

    @property
    def context(self) -> DataStoreContext:
        return DataStoreContext(
            root=str(self.root),
            build_id=self.build_id,
            fingerprint=self.fingerprint,
            split_manifest_hash=self.split_manifest_hash,
            raw_csv_sha256=self.raw_csv_sha256,
            feature_schema_version=str(self._metadata.get("feature_schema_version", "")),
            max_path_distance=int(self._metadata.get("max_path_distance", 0)),
            task_names=tuple(self.task_names),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_envs"] = {}
        return state

    def _env_for_index(self, global_index: int):
        """Return the LMDB env owning this row, opening its shard lazily."""

        position = bisect.bisect_right(self.shard_first_indices, int(global_index)) - 1
        if position < 0:
            raise IndexError(f"global graph index out of range: {global_index}")
        entry = self.shard_table[position]
        last_index = entry["first_index"] + entry["entries"] - 1
        if not entry["first_index"] <= int(global_index) <= last_index:
            raise IndexError(f"global graph index out of range: {global_index}")
        env = self._envs.get(entry["name"])
        if env is None:
            env = lmdb.open(
                str(self.shard_directory(entry["name"])),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=512,
                subdir=True,
            )
            self._envs[entry["name"]] = env
        return env

    def close(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs.clear()
        # Windows keeps the labels.npy mapping open until the memmap object is
        # released, which would prevent atomic directory rename after build
        # validation.
        self._labels = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def task_index(self, task_name: str) -> int:
        try:
            return self.task_names.index(task_name)
        except ValueError as exc:
            raise KeyError(f"Unknown task {task_name!r}; available={self.task_names}") from exc

    def get_label(self, global_index: int, task_name: str) -> float:
        value = self._labels[int(global_index), self.task_index(task_name)]
        return float(value)

    def get_task_indices(
        self,
        task_name: str,
        *,
        split: str | None = None,
        max_nodes: int | None = None,
    ) -> np.ndarray:
        task_idx = self.task_index(task_name)
        mask = np.isfinite(self._labels[:, task_idx])
        if split is not None:
            normalized = "validation" if split == "val" else str(split)
            if normalized not in SPLIT_CODES:
                raise ValueError(f"Unknown split: {split!r}")
            mask &= self.split_codes == SPLIT_CODES[normalized]
        if max_nodes is not None:
            if int(max_nodes) <= 0:
                raise ValueError("max_nodes must be positive when provided")
            mask &= self.num_nodes <= int(max_nodes)
        return np.asarray(self.row_indices[mask], dtype=np.int64)

    def lmdb_entry_count(self) -> int:
        """Physical LMDB entry count across every opened graph shard."""

        return sum(
            int(self._env_for_index(entry["first_index"]).stat().get("entries", 0))
            for entry in self.shard_table
        )

    def get_graph_record(self, global_index: int) -> dict:
        index = int(global_index)
        if index < 0 or index >= len(self.row_indices):
            raise IndexError(f"global graph index out of range: {index}")
        env = self._env_for_index(index)
        with env.begin(buffers=True) as transaction:
            payload = transaction.get(graph_key(index))
        if payload is None:
            raise KeyError(f"Graph record not found for global index {index}")
        record = deserialize_graph_record(bytes(payload))
        if not isinstance(record, dict):
            raise ValueError("Graph record must deserialize to a dict")
        if int(record.get("graph_record_version", -1)) != GRAPH_RECORD_VERSION:
            raise ValueError("Graph record version does not match DataStore V2")
        if record.get("feature_schema_version") != self._metadata.get("feature_schema_version"):
            raise ValueError("Graph record feature schema does not match datastore metadata")
        if "y" in record or "task_name" in record:
            raise ValueError("Task-specific labels must not be stored in graph records")
        if str(record.get("sample_id")) != str(self.sample_ids[index]):
            raise ValueError("Graph record sample_id does not match index.npz")
        if int(record.get("num_nodes", -1)) != int(self.num_nodes[index]):
            raise ValueError("Graph record num_nodes does not match index.npz")
        return record

    def get_graph_data(
        self,
        global_index: int,
        *,
        task_name: str | None = None,
        label: float | None = None,
    ) -> Data:
        record = self.get_graph_record(global_index)
        if task_name is not None and label is None:
            label = self.get_label(global_index, task_name)
        if label is None:
            label = 0.0
        return Data(
            x=record["x"],
            in_degree=record["in_degree"],
            out_degree=record["out_degree"],
            spatial_pos=record["spatial_pos"],
            attn_edge_type=record["attn_edge_type"],
            edge_input=record["edge_input"],
            y=torch.tensor([float(label)], dtype=torch.float),
            label=float(label),
            smiles=str(record.get("raw_smiles", "")),
            raw_smiles=str(record.get("raw_smiles", "")),
            canonical_smiles=str(record.get("canonical_smiles", "")),
            scaffold=str(record.get("scaffold", "")),
            scaffold_smiles=str(record.get("scaffold", "")),
            sample_id=str(record.get("sample_id", "")),
            task_name=task_name,
            num_nodes=int(record.get("num_nodes", record["x"].size(0))),
            feature_schema_version=record.get("feature_schema_version"),
            graph_record_version=int(record.get("graph_record_version", GRAPH_RECORD_VERSION)),
        )

    def validate(
        self,
        *,
        strict: bool = False,
        expected_task_names: Sequence[str] | None = None,
        expected_max_path_distance: int | None = None,
        require_ready: bool = True,
    ) -> None:
        required_paths = (
            self.root / "datastore.json",
            self.root / "split_manifest.json",
            self.root / "index.npz",
            self.root / "labels.npy",
            self.root / "task_stats.json",
            self.root / "preprocess_errors.json",
            self.graph_shards_root,
        )
        if require_ready and not (self.root / "READY").exists():
            raise ValueError(f"DataStore build is missing READY: {self.root}")
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete DataStore V2 build; missing={missing}")
        if self._metadata.get("format") != DATASTORE_FORMAT:
            raise ValueError("Unexpected DataStore format")
        if int(self._metadata.get("format_version", -1)) != DATASTORE_FORMAT_VERSION:
            raise ValueError("Unsupported DataStore format version")
        if int(self._metadata.get("graph_record_version", -1)) != GRAPH_RECORD_VERSION:
            raise ValueError("Unsupported graph record version")
        if self._metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("DataStore feature schema does not match current code")
        if expected_task_names is not None:
            unknown_tasks = [task for task in expected_task_names if task not in self.task_names]
            if unknown_tasks:
                raise ValueError(f"DataStore is missing requested task_names: {unknown_tasks}")
        if expected_max_path_distance is not None and int(self._metadata.get("max_path_distance", -1)) != int(
            expected_max_path_distance
        ):
            raise ValueError("DataStore max_path_distance does not match the requested configuration")
        if self._labels.ndim != 2:
            raise ValueError("labels.npy must have shape [num_samples, num_tasks]")
        expected_shape = (len(self.row_indices), len(self.task_names))
        if tuple(self._labels.shape) != expected_shape:
            raise ValueError(f"labels shape {self._labels.shape} does not match {expected_shape}")
        if int(self._metadata.get("num_samples", -1)) != expected_shape[0]:
            raise ValueError("datastore.json num_samples does not match index.npz")
        if int(self._metadata.get("num_tasks", -1)) != expected_shape[1]:
            raise ValueError("datastore.json num_tasks does not match labels.npy")
        if list(self._metadata.get("labels_shape", [])) != list(expected_shape):
            raise ValueError("datastore.json labels_shape does not match labels.npy")
        if int(self._metadata.get("lmdb_entries", -1)) != expected_shape[0]:
            raise ValueError("datastore.json lmdb_entries does not match num_samples")
        if sum(int(entry["entries"]) for entry in self.shard_table) != expected_shape[0]:
            raise ValueError("graph shard table does not cover every sample")
        for entry in self.shard_table:
            data_file = self.shard_directory(entry["name"]) / "data.mdb"
            if not data_file.exists():
                raise FileNotFoundError(f"Graph shard is missing its LMDB data file: {data_file}")
        if len(self.sample_ids) != len(self.row_indices) or len(self.split_codes) != len(self.row_indices):
            raise ValueError("index.npz arrays have inconsistent lengths")
        if np.any(self.row_indices != np.arange(len(self.row_indices), dtype=np.int64)):
            raise ValueError("row_indices must be contiguous global indices")
        if np.any(~np.isin(self.split_codes, np.asarray(list(CODE_TO_SPLIT), dtype=np.int8))):
            raise ValueError("index.npz contains an unknown split code")
        manifest = load_manifest(self.root / "split_manifest.json")
        if int(manifest.get("manifest_version", 1)) != MANIFEST_VERSION:
            raise ValueError(
                f"DataStore requires split manifest version {MANIFEST_VERSION}; "
                f"found {manifest.get('manifest_version')!r} — re-run "
                "preprocess_data.py --build_datastore_v2 to rebuild the split"
            )
        if manifest_hash(manifest) != self.split_manifest_hash:
            raise ValueError("split_manifest_hash does not match split_manifest.json")
        manifest_by_id = {str(record["sample_id"]): record for record in manifest["records"]}
        if set(str(value) for value in self.sample_ids) != set(manifest_by_id):
            raise ValueError("DataStore sample IDs do not match split manifest")
        for index, sample_id in enumerate(self.sample_ids):
            record = manifest_by_id[str(sample_id)]
            if SPLIT_CODES[record["split"]] != int(self.split_codes[index]):
                raise ValueError("index.npz split codes do not match split_manifest.json")
        expected_fingerprint = compute_datastore_fingerprint(
            raw_csv_sha256=self.raw_csv_sha256,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            graph_record_version=GRAPH_RECORD_VERSION,
            max_path_distance=int(self._metadata["max_path_distance"]),
            task_names=self.task_names,
            split_manifest_hash=self.split_manifest_hash,
        )
        if self.fingerprint != expected_fingerprint:
            raise ValueError("datastore_fingerprint is internally inconsistent")
        if strict:
            for entry in self.shard_table:
                env = self._env_for_index(entry["first_index"])
                entries = int(env.stat().get("entries", -1))
                if entries != int(entry["entries"]):
                    raise ValueError(
                        f"Graph shard {entry['name']} has LMDB entries={entries} "
                        f"but metadata declares {entry['entries']}"
                    )
            for index in range(len(self.row_indices)):
                self.get_graph_record(index)


class ToxAcuteTaskDataset(Dataset):
    """A task/split view over the shared graph store."""

    def __init__(
        self,
        store: ToxAcuteDataStore,
        task_name: str,
        *,
        split: str | None,
        max_nodes: int | None = None,
        label_provider: object | Callable[[int, str], float] | None = None,
    ):
        self.store = store
        self.task_name = str(task_name)
        self.split = split
        self.max_nodes = max_nodes
        self.label_provider = label_provider
        if self.task_name in store.task_names:
            self.task_idx = store.task_index(self.task_name)
            self.indices = store.get_task_indices(self.task_name, split=split, max_nodes=max_nodes)
        else:
            if label_provider is None or not hasattr(label_provider, "get_label"):
                raise KeyError(f"Unknown task {self.task_name!r} and no label provider was supplied")
            self.task_idx = None
            self.indices = np.asarray(
                [
                    index
                    for index in range(len(store.row_indices))
                    if label_provider.get_label(index, self.task_name, split=split) is not None
                    and (max_nodes is None or int(store.num_nodes[index]) <= int(max_nodes))
                ],
                dtype=np.int64,
            )

    def __len__(self):
        return int(len(self.indices))

    def _label(self, global_index: int) -> float:
        if self.label_provider is None:
            return self.store.get_label(global_index, self.task_name)
        if callable(self.label_provider):
            return float(self.label_provider(global_index, self.task_name))
        value = self.label_provider.get_label(global_index, self.task_name, split=self.split)
        if value is None:
            raise ValueError(f"Label provider returned None for {self.task_name} index={global_index}")
        return float(value)

    def __getitem__(self, index: int) -> Data:
        global_index = int(self.indices[index])
        return self.store.get_graph_data(
            global_index,
            task_name=self.task_name,
            label=self._label(global_index),
        )

    def get_sample_id(self, index: int) -> str:
        """Return metadata without opening LMDB or deserializing a graph."""

        global_index = int(self.indices[index])
        return str(self.store.sample_ids[global_index])


def _is_valid_label(value) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _task_names_from_input(df, task_names: Sequence[str] | str | None) -> list[str]:
    if task_names is None or task_names == "all":
        candidates = list(df.columns[6:])
    elif isinstance(task_names, str):
        candidates = [item.strip() for item in task_names.split(",") if item.strip()]
    else:
        candidates = [str(item) for item in task_names]
    if not candidates:
        raise ValueError("No task names were provided")
    missing = [name for name in candidates if name not in df.columns]
    if missing:
        raise KeyError(f"Tasks not found in raw CSV: {missing}")
    if len(set(candidates)) != len(candidates):
        raise ValueError("task_names must be unique")
    return candidates


def _split_codes_for_manifest(manifest: Mapping, sample_ids: Sequence[str]) -> np.ndarray:
    validate_manifest(dict(manifest))
    assignments = {str(record["sample_id"]): record["split"] for record in manifest["records"]}
    if set(assignments) != set(str(value) for value in sample_ids):
        raise ValueError("Approved split manifest sample set does not match graph-successful samples")
    return np.asarray([SPLIT_CODES[assignments[str(sample_id)]] for sample_id in sample_ids], dtype=np.int8)


def _write_json(path: Path, payload: Mapping | Sequence) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def build_datastore_v2(
    raw_csv_path: str | Path,
    data_store_dir: str | Path,
    *,
    task_names: Sequence[str] | str | None = None,
    splitting: str = "scaffold",
    valid_size: float = 0.1,
    calibration_size: float = 0.1,
    test_size: float = 0.1,
    split_seed: int = 42,
    max_path_distance: int = 8,
    lmdb_map_size_gb: float = 64.0,
    commit_every: int = 512,
    split_manifest_path: str | Path | None = None,
    graph_shard_max_gb: float = DEFAULT_GRAPH_SHARD_MAX_GB,
) -> Path:
    """Build a new V2 artifact and atomically activate it via ``CURRENT``.

    Graph payloads are streamed into bounded LMDB shards (see
    ``graph_shard_max_gb``), so neither RAM nor any single file on disk has to
    hold the full serialized graph set.
    """

    raw_csv_path = Path(raw_csv_path).resolve()
    root = Path(data_store_dir).resolve()
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV data file not found: {raw_csv_path}")
    if int(max_path_distance) <= 0:
        raise ValueError("max_path_distance must be positive")
    if int(commit_every) <= 0:
        raise ValueError("commit_every must be positive")
    if float(graph_shard_max_gb) <= 0:
        raise ValueError("graph_shard_max_gb must be positive")
    shard_max_bytes = int(float(graph_shard_max_gb) * 1024**3)
    ratios = {
        "train": 1.0 - float(valid_size) - float(calibration_size) - float(test_size),
        "validation": float(valid_size),
        "calibration": float(calibration_size),
        "test": float(test_size),
    }
    if ratios["train"] <= 0 or any(value < 0 for value in ratios.values()):
        raise ValueError("train/validation/calibration/test ratios are invalid")

    import pandas as pd

    from preprocess_data import get_graph_features_from_smiles

    dataframe = pd.read_csv(raw_csv_path)
    tasks = _task_names_from_input(dataframe, task_names)
    raw_sha256 = sha256_file(raw_csv_path)
    root.mkdir(parents=True, exist_ok=True)
    builds_root = root / "builds"
    builds_root.mkdir(parents=True, exist_ok=True)
    temp_build = builds_root / f".tmp-{uuid.uuid4().hex}"
    temp_build.mkdir(parents=True, exist_ok=False)
    active_writer: _GraphShardWriter | None = None
    try:
        # Graph tensors are serialized and committed straight into bounded
        # LMDB shards as rows are processed; only lightweight metadata stays
        # in RAM, so the full graph list never exists in memory at once.
        shards_root = temp_build / GRAPH_SHARD_DIRNAME
        initial_map_size = min(
            max(8 << 20, int(float(lmdb_map_size_gb) * 1024**3)),
            shard_max_bytes + _LMDB_OVERHEAD_BYTES,
        )

        def _open_writer(position: int) -> _GraphShardWriter:
            return _GraphShardWriter(
                shards_root / shard_name(position),
                max_bytes=shard_max_bytes,
                initial_map_size=initial_map_size,
            )

        manifest_records = []
        errors = []
        shard_entries: list[dict] = []
        shard_position = 0
        shard_start_index = 0
        active_writer = _open_writer(shard_position)
        for row_index, row in dataframe.iterrows():
            raw_smiles = row.get("smiles", row.get("SMILES"))
            if raw_smiles is None or (isinstance(raw_smiles, float) and np.isnan(raw_smiles)):
                errors.append({"row_index": int(row_index), "error": "missing_smiles"})
                continue
            try:
                graph = get_graph_features_from_smiles(
                    str(raw_smiles),
                    sample_id=f"row_{int(row_index)}",
                    max_path_distance=int(max_path_distance),
                )
            except Exception as exc:
                errors.append({"row_index": int(row_index), "smiles": str(raw_smiles), "error": str(exc)})
                continue
            payload = serialize_graph_record(graph)
            if active_writer.full_for(len(payload)):
                finished = active_writer.flush_and_close()
                shard_entries.append(
                    {
                        "name": finished["name"],
                        "first_index": shard_start_index,
                        "entries": finished["entries"],
                        "payload_bytes": finished["payload_bytes"],
                    }
                )
                shard_start_index = len(manifest_records)
                shard_position += 1
                active_writer = _open_writer(shard_position)
            active_writer.add(graph_key(len(manifest_records)), payload, commit_every=commit_every)
            manifest_records.append(
                {
                    "sample_id": graph["sample_id"],
                    "row_index": int(row_index),
                    "raw_smiles": graph["raw_smiles"],
                    "canonical_smiles": graph["canonical_smiles"],
                    "scaffold": graph["scaffold"],
                    "num_nodes": int(graph["num_nodes"]),
                }
            )
        if not manifest_records:
            raise ValueError("No valid graph records were produced")
        finished = active_writer.flush_and_close()
        active_writer = None
        shard_entries.append(
            {
                "name": finished["name"],
                "first_index": shard_start_index,
                "entries": finished["entries"],
                "payload_bytes": finished["payload_bytes"],
            }
        )
        validated_shards = normalize_shard_table(shard_entries)
        # Keep byte sizes as diagnostics; the normalized table itself only
        # carries the fields readers depend on.
        payload_by_name = {entry["name"]: int(entry["payload_bytes"]) for entry in shard_entries}
        validated_shards = [
            {**entry, "payload_bytes": payload_by_name[entry["name"]]} for entry in validated_shards
        ]
        if sum(entry["entries"] for entry in validated_shards) != len(manifest_records):
            raise ValueError("Graph shard table does not tile every written record")

        if split_manifest_path is None:
            manifest = create_split_manifest(
                manifest_records,
                splitting=splitting,
                ratios=ratios,
                seed=int(split_seed),
            )
        else:
            manifest = load_manifest(split_manifest_path)
            if int(manifest.get("manifest_version", 1)) != MANIFEST_VERSION:
                raise ValueError(
                    f"Approved split manifest must be version {MANIFEST_VERSION}; "
                    f"found {manifest.get('manifest_version')!r} — regenerate it with "
                    "the current preprocessing code"
                )
            if manifest.get("splitting") != splitting:
                raise ValueError("Approved split manifest splitting does not match the requested splitting")
            if int(manifest.get("seed")) != int(split_seed):
                raise ValueError("Approved split manifest seed does not match split_seed")
            actual_ratios = manifest.get("ratios", {})
            for name, expected in ratios.items():
                if not np.isclose(float(actual_ratios.get(name, np.nan)), expected):
                    raise ValueError(f"Approved split ratio for {name} does not match requested configuration")
            _split_codes_for_manifest(manifest, [record["sample_id"] for record in manifest_records])
        validate_manifest(manifest)
        split_codes = _split_codes_for_manifest(manifest, [record["sample_id"] for record in manifest_records])

        labels = np.full((len(manifest_records), len(tasks)), np.nan, dtype=np.float32)
        observed = 0
        for global_index, record in enumerate(manifest_records):
            row_index = int(record["row_index"])
            for task_index, task_name in enumerate(tasks):
                value = dataframe.at[row_index, task_name]
                if _is_valid_label(value):
                    labels[global_index, task_index] = float(value)
                    observed += 1

        np.save(temp_build / "labels.npy", labels)
        np.savez_compressed(
            temp_build / "index.npz",
            sample_ids=np.asarray([record["sample_id"] for record in manifest_records], dtype=str),
            row_indices=np.arange(len(manifest_records), dtype=np.int64),
            split_codes=split_codes,
            num_nodes=np.asarray([int(record["num_nodes"]) for record in manifest_records], dtype=np.int64),
        )
        write_manifest(manifest, temp_build / "split_manifest.json")
        _write_json(temp_build / "preprocess_errors.json", errors)

        task_stats = {}
        for task_index, task_name in enumerate(tasks):
            stats = {name: int(np.sum(np.isfinite(labels[split_codes == code, task_index]))) for name, code in SPLIT_CODES.items()}
            stats["total"] = int(np.sum(np.isfinite(labels[:, task_index])))
            task_stats[task_name] = stats
        low_calibration_tasks = [
            task for task, stats in task_stats.items() if stats["calibration"] < 30
        ]
        task_stats_payload = {
            "tasks": task_stats,
            "low_calibration_tasks": low_calibration_tasks,
            "empty_train_tasks": [task for task, stats in task_stats.items() if stats["train"] == 0],
            "empty_validation_tasks": [task for task, stats in task_stats.items() if stats["validation"] == 0],
            "empty_calibration_tasks": [task for task, stats in task_stats.items() if stats["calibration"] == 0],
            "empty_test_tasks": [task for task, stats in task_stats.items() if stats["test"] == 0],
            "num_samples": len(manifest_records),
            "num_scaffolds": len({str(record.get("scaffold", "")) for record in manifest_records}),
            "acyclic_count": sum(str(record.get("scaffold", "")) == "__ACYCLIC__" for record in manifest_records),
            "largest_scaffold_group": max(
                (sum(str(record.get("scaffold", "")) == scaffold for record in manifest_records)
                 for scaffold in {str(record.get("scaffold", "")) for record in manifest_records}),
                default=0,
            ),
        }
        _write_json(temp_build / "task_stats.json", task_stats_payload)
        preflight_rows = []
        for task_name, stats in task_stats.items():
            status = "OK"
            if stats["train"] == 0:
                status = "FAIL_EMPTY_TRAIN"
            elif stats["validation"] == 0:
                status = "FAIL_EMPTY_VAL"
            elif stats["calibration"] == 0:
                status = "FAIL_EMPTY_CAL"
            elif stats["test"] == 0:
                status = "FAIL_EMPTY_TEST"
            elif stats["calibration"] < 30:
                status = "LOW_CALIBRATION"
            preflight_rows.append({"task": task_name, **stats, "excluded_nodes": 0, "status": status})
        _write_json(temp_build / "data_preflight.json", preflight_rows)

        split_hash = manifest_hash(manifest)
        fingerprint = compute_datastore_fingerprint(
            raw_csv_sha256=raw_sha256,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            graph_record_version=GRAPH_RECORD_VERSION,
            max_path_distance=int(max_path_distance),
            task_names=tasks,
            split_manifest_hash=split_hash,
        )
        build_id = f"toxacute-v2-{fingerprint[:12]}"
        metadata = {
            "format": DATASTORE_FORMAT,
            "format_version": DATASTORE_FORMAT_VERSION,
            "build_id": build_id,
            "datastore_fingerprint": fingerprint,
            "raw_csv_path_at_build_time": str(raw_csv_path),
            "raw_csv_sha256": raw_sha256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "graph_record_version": GRAPH_RECORD_VERSION,
            "graph_layout": GRAPH_LAYOUT_NAME,
            "graph_shard_max_bytes": shard_max_bytes,
            "graph_shards": validated_shards,
            "max_path_distance": int(max_path_distance),
            "task_names": tasks,
            "num_tasks": len(tasks),
            "num_samples": len(manifest_records),
            "num_observed_labels": int(observed),
            "splitting": splitting,
            "split_seed": int(split_seed),
            "split_ratios": ratios,
            "split_manifest_hash": split_hash,
            "lmdb_entries": len(manifest_records),
            "labels_shape": list(labels.shape),
            "build_complete": True,
        }
        _write_json(temp_build / "datastore.json", metadata)

        validation_store = ToxAcuteDataStore(temp_build, validate_on_open=False)
        validation_store.validate(strict=True, require_ready=False)
        validation_store.close()
        (temp_build / "READY").write_text("ready\n", encoding="utf-8")

        final_build = builds_root / build_id
        if final_build.exists():
            if not (final_build / "READY").exists():
                raise FileExistsError(f"A non-ready build already exists: {final_build}")
            shutil.rmtree(temp_build)
        else:
            os.replace(temp_build, final_build)
        current_tmp = root / f".CURRENT-{uuid.uuid4().hex}"
        current_tmp.write_text(build_id + "\n", encoding="utf-8")
        os.replace(current_tmp, root / "CURRENT")
        return final_build
    except Exception:
        if active_writer is not None:
            try:
                active_writer.env.close()
            except Exception:
                pass
        if temp_build.exists():
            shutil.rmtree(temp_build, ignore_errors=True)
        raise


__all__ = [
    "CODE_TO_SPLIT",
    "DATASTORE_FORMAT",
    "DATASTORE_FORMAT_VERSION",
    "DataStoreContext",
    "DEFAULT_GRAPH_SHARD_MAX_GB",
    "GRAPH_LAYOUT_NAME",
    "GRAPH_RECORD_VERSION",
    "SPLIT_CODES",
    "ToxAcuteDataStore",
    "ToxAcuteTaskDataset",
    "build_datastore_v2",
    "compute_datastore_fingerprint",
    "deserialize_graph_record",
    "graph_key",
    "normalize_shard_table",
    "serialize_graph_record",
    "sha256_file",
    "shard_name",
]
