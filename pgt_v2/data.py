"""PGT inputs and label views. No fitting statistic accepts an unfiltered table."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
import torch

from architecture.graphormer_backbone import MolecularGraphormerBackbone
from baselines.models.toxacol import task_adjacency
from dataset import DataCollator
from molecular_features import PGT_FEATURE_SPEC
from .chemistry import digest, framed_digest


def make_backbone(*, feature_spec=PGT_FEATURE_SPEC, edge_bias_mode="direct", **kwargs):
    if edge_bias_mode != "direct":
        raise ValueError("PGT V2 fixes direct edge bias; path modes belong to the legacy protocol")
    feature_spec.validate_runtime()
    return MolecularGraphormerBackbone(feature_spec=feature_spec, edge_bias_mode="direct", **kwargs)


def validate_graph(item, spec=PGT_FEATURE_SPEC):
    if (getattr(item, "feature_schema_version", None) != spec.version or
            getattr(item, "feature_spec_sha256", None) != spec.sha256 or
            getattr(item, "edge_bias_mode", None) != "direct"):
        raise ValueError("Graph FeatureSpec/direct-mode mismatch")
    n = item.x.shape[0]
    if n < 1 or item.num_nodes != n or item.x.shape != (n, len(spec.atom_fields)):
        raise ValueError("Graph atom count/feature width mismatch")
    for values, cardinalities, shape in (
        (item.x, spec.atom_cardinalities, (n, len(spec.atom_fields))),
        (item.attn_edge_type, spec.bond_cardinalities, (n, n, len(spec.bond_fields))),
    ):
        if values.dtype != torch.long or values.shape != shape:
            raise ValueError("Categorical features must be int64 with the declared shape")
        for col, maximum in enumerate(cardinalities):
            if torch.any(values[..., col] < 0) or torch.any(values[..., col] > maximum):
                raise ValueError("Feature index outside the explicit vocabulary")
    if torch.any(item.x == 0):
        raise ValueError("A real atom cannot contain padding feature indices")
    for name in ("raw_to_model", "model_to_raw"):
        if sorted(getattr(item, name, [])) != list(range(n)):
            raise ValueError("Missing or non-bijective atom map")
    if any(item.model_to_raw[item.raw_to_model[r]] != r for r in range(n)):
        raise ValueError("Atom maps are not inverses")
    if len(item.symmetry_classes) != n or not str(item.model_input_id).startswith("graph_"):
        raise ValueError("Missing chemical identity or symmetry classes")
    if not isinstance(item.raw_row_index, int) or item.raw_row_index < 0 or not str(item.sample_id):
        raise ValueError("Missing source record identity")
    if item.edge_input.shape != (n, n, 1, len(spec.bond_fields)) or not torch.all(item.edge_input == -1):
        raise ValueError("PGT direct inputs must not contain legacy selected-path sequences")
    if item.spatial_pos.shape != (n, n) or item.in_degree.shape != (n,) or item.out_degree.shape != (n,):
        raise ValueError("Invalid graph distance/degree dimensions")


class PGTCollator(DataCollator):
    """Keep full graphs and explicit record/atom identities; support B x T labels."""
    def __init__(self, *, spec=PGT_FEATURE_SPEC, spatial_pos_max_clip=20):
        super().__init__(spatial_pos_max_clip=spatial_pos_max_clip, max_node_filter=None)
        self.spec = spec

    def __call__(self, data_list):
        data_list = list(data_list)
        if not data_list or any(item is None for item in data_list):
            raise ValueError("PGT does not silently discard missing or oversized graph records")
        for item in data_list:
            validate_graph(item, self.spec)
        task_orders = {tuple(getattr(item, "task_names", ())) for item in data_list}
        view_ids = {getattr(item, "training_view_id", None) for item in data_list}
        widths = {item.y.numel() for item in data_list}
        if len(task_orders) != 1 or len(view_ids) != 1 or len(widths) != 1:
            raise ValueError("Mixed task orders or training views in one batch")
        tasks = next(iter(task_orders))
        if (tasks and len(tasks) != next(iter(widths))) or (not tasks and next(iter(widths)) != 1):
            raise ValueError("Multi-task labels require explicit task_names")
        batch = super().__call__(data_list)
        batch.y = torch.stack([item.y.reshape(-1).float() for item in data_list])
        batch.feature_schema_version = self.spec.version
        batch.feature_spec_sha256 = self.spec.sha256
        batch.edge_bias_mode = "direct"
        batch.task_names = list(tasks)
        batch.training_view_id = next(iter(view_ids))
        for name in ("raw_row_index", "model_input_id", "nonstereo_id", "raw_to_model", "model_to_raw", "symmetry_classes"):
            setattr(batch, name, [getattr(item, name) for item in data_list])
        # Offset is optional for fresh graphs and never substituted for source row.
        batch.cache_offset = [getattr(item, "cache_offset", None) for item in data_list]
        return batch


def align_task_columns(values, source_names, target_names):
    source_names, target_names = tuple(source_names), tuple(target_names)
    if (not source_names or len(set(source_names)) != len(source_names) or
            len(set(target_names)) != len(target_names) or set(source_names) != set(target_names)):
        raise ValueError("Task names must be unique and have exactly the same set")
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(source_names):
        raise ValueError("Label matrix width does not match its task names")
    index = {name: i for i, name in enumerate(source_names)}
    return values[:, [index[name] for name in target_names]].copy()


def legacy_cache_row_mapping(sample_ids, manifest_records):
    """Legacy index.row_indices are offsets. Recover source rows through IDs."""
    by_id = {}
    for record in manifest_records:
        key = str(record["sample_id"])
        if key in by_id:
            raise ValueError("Duplicate sample_id in the legacy manifest")
        row = record["row_index"]
        if not isinstance(row, int) or isinstance(row, bool) or row < 0:
            raise ValueError("Invalid original row_index")
        by_id[key] = row
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate cache sample_id")
    result = []
    for offset, sample_id in enumerate(sample_ids):
        if str(sample_id) not in by_id:
            raise ValueError("Cache sample_id missing from source manifest")
        result.append({"cache_offset": offset, "sample_id": str(sample_id), "raw_row_index": by_id[str(sample_id)]})
    if len({r["raw_row_index"] for r in result}) != len(result):
        raise ValueError("Multiple cache records mapped to one original source row")
    return result


def _immutable_array(values, dtype=None):
    array = np.ascontiguousarray(values, dtype=dtype)
    # A bytes-backed array cannot be made writable with setflags(write=True).
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class TrainingView:
    view_id: str
    task_names: tuple[str, ...]
    train_record_ids: tuple[str, ...]
    train_group_ids: tuple[str, ...]
    train_chemical_ids: tuple[str, ...]
    train_raw_row_indices: tuple[int, ...]
    visible_labels: np.ndarray
    visible_mask: np.ndarray
    count: np.ndarray
    group_count: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    scale_fallback: np.ndarray
    adjacency: np.ndarray
    source_sha256: str
    feature_spec_sha256: str

    @cached_property
    def _record_lookup(self):
        return {record: i for i, record in enumerate(self.train_record_ids)}

    def labels_for(self, record_ids, *, standardized=False):
        lookup = self._record_lookup
        if any(record not in lookup for record in record_ids):
            raise ValueError("TrainingView cannot return validation/test/unknown records")
        values = self.visible_labels[[lookup[r] for r in record_ids]].copy()
        return (values - self.mean) / self.scale if standardized else values

    def attach(self, graph, *, standardized=True):
        if graph.sample_id not in self._record_lookup:
            raise ValueError("Graph is not part of this training view")
        row = self._record_lookup[graph.sample_id]
        if (graph.model_input_id != self.train_chemical_ids[row] or
                graph.raw_row_index != self.train_raw_row_indices[row] or
                graph.feature_spec_sha256 != self.feature_spec_sha256 or
                getattr(graph, "source_sha256", None) != self.source_sha256):
            raise ValueError("Training graph/source-row/chemical/FeatureSpec identity mismatch")
        result = graph.clone()
        result.y = torch.tensor(self.labels_for([graph.sample_id], standardized=standardized)[0], dtype=torch.float32)
        result.task_names = list(self.task_names)
        result.training_view_id = self.view_id
        return result


def make_training_view(*, labels, label_task_names, task_names, record_ids, raw_row_indices, group_ids,
                       chemical_ids, roles, visible_mask, source_sha256, fold_id,
                       feature_spec=PGT_FEATURE_SPEC):
    names = tuple(task_names)
    labels = align_task_columns(labels, label_task_names, names).astype(np.float64)
    mask_input = np.asarray(visible_mask)
    if mask_input.dtype != np.bool_ or mask_input.shape != (len(record_ids), len(label_task_names)):
        raise ValueError("Visible mask must be boolean, in the declared label_task_names order")
    visible = align_task_columns(mask_input, label_task_names, names)
    n = len(record_ids)
    if labels.shape[0] != n or any(len(x) != n for x in (raw_row_indices, group_ids, chemical_ids, roles)):
        raise ValueError("Row identities and labels have different lengths")
    if (len(set(raw_row_indices)) != n or
            any(not isinstance(r, int) or isinstance(r, bool) or r < 0 for r in raw_row_indices)):
        raise ValueError("Original source row indices must be unique nonnegative integers")
    if len(set(record_ids)) != n or not all(record_ids) or not all(group_ids) or not all(chemical_ids):
        raise ValueError("Record IDs must be unique; group/chemical IDs cannot be empty")
    if set(roles) - {"train", "validation", "test"}:
        raise ValueError("Unknown data partition")
    train = np.asarray(roles) == "train"
    if not train.any() or visible[~train].any():
        raise ValueError("Only training labels may enter a TrainingView")
    for ids in (group_ids, chemical_ids):
        identity_roles = {}
        for identity, role in zip(ids, roles):
            identity_roles.setdefault(identity, set()).add(role)
        if any(len(v) > 1 for v in identity_roles.values()):
            raise ValueError("A chemical identity or structure group crosses data partitions")
    chemical_groups = {}
    for chemical, group in zip(chemical_ids, group_ids):
        chemical_groups.setdefault(chemical, set()).add(group)
    if any(len(groups) != 1 for groups in chemical_groups.values()):
        raise ValueError("The same chemical identity cannot inflate independent group counts")
    # Hidden values are not inspected or retained, including hidden infinities.
    if not np.isfinite(labels[visible]).all():
        raise ValueError("Visible labels must be finite; absent labels must be masked")
    safe = np.full(labels.shape, np.nan, dtype=np.float64)
    safe[visible] = labels[visible]
    safe = safe[train]
    visible = visible[train]
    count = visible.sum(axis=0)
    if np.any(count == 0):
        raise ValueError("Zero visible labels for tasks: " + str([names[i] for i in np.where(count == 0)[0]]))
    means = np.nanmean(safe, axis=0)
    raw_scale = np.nanstd(safe, axis=0, ddof=0)
    fallback = (count < 2) | (raw_scale <= 1e-8)
    scale = np.where(fallback, 1.0, raw_scale)
    groups = tuple(g for g, keep in zip(group_ids, train) if keep)
    train_records = tuple(r for r, keep in zip(record_ids, train) if keep)
    train_chemicals = tuple(c for c, keep in zip(chemical_ids, train) if keep)
    train_rows = tuple(r for r, keep in zip(raw_row_indices, train) if keep)
    group_count = [len({g for g, keep in zip(groups, visible[:, i]) if keep}) for i in range(len(names))]
    adjacency, _ = task_adjacency(safe, min_shared=15, pcc_threshold=0.75)
    identity = digest({"source_sha256": source_sha256, "feature_spec_sha256": feature_spec.sha256,
        "fold_id": str(fold_id), "task_names": names, "record_ids": list(record_ids), "roles": list(roles),
        "group_ids": list(group_ids), "chemical_ids": list(chemical_ids), "raw_row_indices": list(raw_row_indices),
        "visible_mask": visible.tolist(), "visible_values_row_major": safe[visible].tolist(),
        "scaler": "population_std_fallback_1_at_n_lt_2_or_std_le_1e-8",
        "adjacency": {"min_shared": 15, "pcc_threshold": 0.75}})
    return TrainingView("view_" + identity, names, train_records, groups, train_chemicals, train_rows,
        _immutable_array(safe), _immutable_array(visible), _immutable_array(count), _immutable_array(group_count),
        _immutable_array(means), _immutable_array(scale), _immutable_array(fallback), _immutable_array(adjacency),
        str(source_sha256), feature_spec.sha256)


def sparse_visible_mask(*, observed, task_names, roles, chemical_ids, target_tasks, fraction, fold_id, subset_seed):
    """C01 nested identity-level subsets; labels themselves are never an input."""
    observed = np.asarray(observed)
    task_names = tuple(task_names)
    if observed.dtype != np.bool_ or observed.shape != (len(roles), len(task_names)):
        raise ValueError("Expected boolean label-presence matrix")
    if len(chemical_ids) != len(roles) or not all(chemical_ids) or len(set(task_names)) != len(task_names):
        raise ValueError("Invalid chemical IDs or task order")
    if set(target_tasks) - set(task_names) or not 0 < fraction <= 1:
        raise ValueError("Invalid target tasks or label fraction")
    if set(roles) - {"train", "validation", "test"}:
        raise ValueError("Unknown role")
    result = observed & (np.asarray(roles) == "train")[:, None]
    for task in target_tasks:
        column = task_names.index(task)
        identities = {c for c, keep in zip(chemical_ids, result[:, column]) if keep}
        if not identities:
            raise ValueError("No full-training labels for target " + task)
        ordered = sorted(identities, key=lambda c: (framed_digest([fold_id, task, subset_seed, c]), c))
        selected = set(ordered[:math.ceil(fraction * len(ordered))])
        result[:, column] &= np.asarray([c in selected for c in chemical_ids])
    return result
