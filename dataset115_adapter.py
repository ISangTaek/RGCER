"""Dataset115 split views and train-only scaling; no training authorization.

The published labels already use -log(mol/kg). Preserve CSV values; do not
infer the log base or convert them back to physical dose in this module.
Graph views emit RAW labels because Trainer normalizes labels itself.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baselines.scaling import TaskScaler
from dataset115_contract import (
    ContractError, PRIMARY, SPLITS, audit, label, namespace_id, semantic_digest,
)

LABEL_SCALE = 'dataset115_published_negative_log_mol_per_kg'


@dataclass(frozen=True)
class LabelView:
    route: str
    role: str
    split: str
    tasks: tuple[str, ...]
    sample_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    canonical: tuple[str, ...]
    groups: tuple[str, ...]
    labels: np.ndarray
    input_identity: str
    scale: str = LABEL_SCALE

    def __post_init__(self):
        if self.route not in {'A', 'B'} or self.role not in {'source', 'target'} or self.split not in SPLITS:
            raise ContractError('invalid route/role/split')
        if self.role == 'source' and self.route != 'A':
            raise ContractError('Route B source is ToxAcute, not dataset115')
        if self.scale != LABEL_SCALE:
            raise ContractError('unsupported label scale; no implicit unit conversion')
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ContractError('empty/duplicate tasks')
        if self.role == 'target' and self.tasks != PRIMARY:
            raise ContractError('target must use frozen Primary5 order')
        if self.role == 'source' and any(t.rsplit('_', 2)[0] in {'child', 'human', 'man', 'women'} for t in self.tasks):
            raise ContractError('human source task forbidden')
        n = len(self.sample_ids)
        if len(set(self.sample_ids)) != n or any(not re.fullmatch(r'dataset115:row_(0|[1-9][0-9]*)', s) for s in self.sample_ids):
            raise ContractError('duplicate/wrong-namespace sample identity')
        if any(len(v) != n for v in (self.smiles, self.canonical, self.groups)):
            raise ContractError('row metadata length differs')
        if any(not isinstance(s, str) or not s for seq in (self.smiles, self.canonical, self.groups) for s in seq):
            raise ContractError('missing molecule/group identity')
        values = np.array(self.labels, dtype=np.float64, copy=True)
        if values.shape != (n, len(self.tasks)) or np.isinf(values).any():
            raise ContractError('label shape/nonfinite value invalid')
        if not isinstance(self.input_identity, str) or not re.fullmatch('[0-9a-f]{64}', self.input_identity):
            raise ContractError('input identity missing')
        values.setflags(write=False)
        object.__setattr__(self, 'labels', values)


class Dataset115Table:
    """Full-table loading validates frozen bytes before returning any split view."""

    @classmethod
    def load(cls, csv_path, split_path, tox_path, *, expected_tox_sha):
        report = audit(csv_path, split_path, tox_path, expected_tox_sha=expected_tox_sha)
        manifest = json.loads(Path(split_path).read_text(encoding='utf8'))
        refs = {r['row_index']: r for r in manifest['records']}
        tox = json.loads(Path(tox_path).read_text(encoding='utf8'))
        overlaps = {r['canonical_smiles'] for r in tox['records']}
        tasks = tuple(report['task_names'])
        records = []; values = []
        with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i not in refs or row['smiles'] != refs[i]['raw_smiles']:
                    raise ContractError('row changed after audit')
                records.append(refs[i])
                values.append([np.nan if (v := label(row[t])) is None else v for t in tasks])
        # Guard changes between the audit and second pass, without redoing its counts.
        from dataset115_contract import digest
        if (digest(csv_path) != report['csv_sha256'] or digest(split_path) != report['split_sha256']
                or digest(tox_path) != report['tox_manifest_sha256'] or len(records) != report['rows']):
            raise ContractError('input changed during loading')
        obj = cls()
        obj.report = report
        obj.tasks = tasks
        obj.source_tasks = tuple(report['nonhuman104'])
        obj.records = tuple(records)
        obj.overlaps = frozenset(overlaps)
        obj.values = np.asarray(values, dtype=np.float64)
        obj.values.setflags(write=False)
        obj.identity = semantic_digest({k: report[k] for k in (
            'csv_sha256', 'split_sha256', 'tox_manifest_sha256', 'tox_manifest_semantic_sha256')})
        return obj

    def view(self, route, role, split):
        if route not in {'A', 'B'} or role not in {'source', 'target'} or split not in SPLITS:
            raise ContractError('unknown route/role/split')
        if role == 'source' and route != 'A':
            raise ContractError('Route B source must come from frozen ToxAcute assets')
        tasks = PRIMARY if role == 'target' else self.source_tasks
        columns = [self.tasks.index(t) for t in tasks]
        indices = [i for i, r in enumerate(self.records)
                   if r['split'] == split and (route == 'A' or r['canonical_smiles'] not in self.overlaps)]
        records = [self.records[i] for i in indices]
        return LabelView(route, role, split, tasks,
            tuple(namespace_id('dataset115', r['row_index']) for r in records),
            tuple(r['raw_smiles'] for r in records), tuple(r['canonical_smiles'] for r in records),
            tuple(r['split_group'] for r in records), self.values[np.ix_(indices, columns)], self.identity)


@dataclass(frozen=True)
class TrainOnlyScaler:
    scaler: TaskScaler
    route: str
    role: str
    input_identity: str
    train_ids_sha256: str

    @classmethod
    def fit(cls, view: LabelView):
        if view.split != 'train':
            raise ContractError('scaler fit is train-only')
        # No silently absent tasks; review any insufficient source/target branch.
        scaler = TaskScaler.fit(view.labels, view.tasks, allow_empty=False)
        return cls(scaler, view.route, view.role, view.input_identity,
                   semantic_digest(list(view.sample_ids)))

    def transform(self, view: LabelView):
        if (view.route, view.role, view.input_identity, view.tasks) != (
                self.route, self.role, self.input_identity, self.scaler.task_names):
            raise ContractError('scaler route/role/data/task identity mismatch')
        return self.scaler.transform(view.labels)

    def trainer_scalers(self):
        """Trainer consumes raw graph y and performs exactly one normalization."""
        return {t: {'mean': float(self.scaler.means[i]), 'std': float(self.scaler.stds[i]),
                    'count': int(self.scaler.counts[i])} for i, t in enumerate(self.scaler.task_names)}

    def to_dict(self):
        return dict(scaler=self.scaler.to_dict(), route=self.route, role=self.role,
                    input_identity=self.input_identity, train_ids_sha256=self.train_ids_sha256,
                    fit_split='train', label_scale=LABEL_SCALE)


class GraphTaskView:
    """Lazy in-memory graph conversion; no LMDB build, cache or split mutation."""
    def __init__(self, view: LabelView, task: str, *, max_path_distance=8):
        if task not in view.tasks:
            raise ContractError('task not present in view')
        self.view = view; self.task = task; self.column = view.tasks.index(task)
        self.indices = np.flatnonzero(np.isfinite(view.labels[:, self.column])).tolist()
        self.max_path_distance = max_path_distance

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        from preprocess_data import get_graph_data_from_smiles
        i = self.indices[index]; v = self.view
        graph = get_graph_data_from_smiles(v.smiles[i], float(v.labels[i, self.column]),
            sample_id=v.sample_ids[i], task_name=self.task, max_path_distance=self.max_path_distance)
        if graph.canonical_smiles != v.canonical[i]:
            raise ContractError('current graph canonicalization differs from frozen record')
        graph.split = v.split; graph.split_group = v.groups[i]
        return graph

    def get_sample_id(self, index):
        return self.view.sample_ids[self.indices[index]]
