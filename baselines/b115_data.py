"""B115 joint-label permissions. No test access or training authorization.

The signed P0-B2 audit is an input contract, not a cache: every input and every
allowed observation is checked again on each load. Paths are supplied by the
executing host; no workstation-specific paths are embedded here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
from dataset115_adapter import Dataset115Table, LabelView
from dataset115_contract import ContractError, PRIMARY, digest, semantic_digest
from .models.toxacol import endpoint_feature_matrix, task_adjacency
from .scaling import TaskScaler

AUDIT_SHA = '8cbb3f7bc8d9a31b3cdb123f022fe526cee4cd4dec3ab1b6bf32cce15288919b'


def check_bytes(path, expected):
    if digest(Path(path)) != expected:
        raise ContractError(f'input SHA mismatch: {Path(path).name}')


@dataclass(frozen=True)
class JointTrain:
    route: str
    tasks: tuple[str, ...]
    sample_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    canonical: tuple[str, ...]
    labels: np.ndarray
    input_identity: str

    def __post_init__(self):
        n = len(self.sample_ids)
        x = np.array(self.labels, dtype=np.float64, copy=True)
        if (self.route not in {'A', 'B'} or not n or len(set(self.sample_ids)) != n
                or len(self.smiles) != n or len(self.canonical) != n
                or x.shape != (n, len(self.tasks)) or np.isinf(x).any()
                or not np.isfinite(x).any(axis=1).all()
                or len(set(self.tasks)) != len(self.tasks) or self.tasks[-5:] != PRIMARY):
            raise ContractError('invalid active joint training view')
        x.setflags(write=False)
        object.__setattr__(self, 'labels', x)


def assemble(route, source_tasks, source_ids, source_smiles, source_canonical,
             source_labels, target: LabelView) -> JointTrain:
    """Only train views enter the joint matrix; source permissions are explicit."""
    source_tasks = tuple(source_tasks)
    if target.split != 'train' or target.role != 'target' or target.route != route:
        raise ContractError('joint assembly requires matching target train')
    if (np.asarray(source_labels).shape != (len(source_ids), len(source_tasks))
            or np.isinf(source_labels).any()):
        raise ContractError('source label shape/nonfinite mismatch')
    if route == 'A':
        if (len(source_tasks) != 104 or any(t.rsplit('_', 2)[0] in
                {'child', 'human', 'man', 'women'} for t in source_tasks)):
            raise ContractError('A requires Nonhuman104 only')
        if (tuple(source_ids) != target.sample_ids or tuple(source_canonical) != target.canonical
                or tuple(source_smiles) != target.smiles):
            raise ContractError('A source/target rows must be identically aligned')
        labels = np.column_stack((source_labels, target.labels))
        ids, smiles, canonical = target.sample_ids, target.smiles, target.canonical
    elif route == 'B':
        if source_tasks != tuple(ANIMAL_SOURCE_TASKS):
            raise ContractError('B requires Animal56, not Human3 or Nonhuman104')
        if any(not s.startswith('toxacute:row_') for s in source_ids):
            raise ContractError('B source namespace missing')
        if set(source_canonical).intersection(target.canonical):
            raise ContractError('B source/target canonical overlap')
        labels = np.full((len(source_ids) + len(target.sample_ids), len(source_tasks) + 5), np.nan)
        labels[:len(source_ids), :len(source_tasks)] = source_labels
        labels[len(source_ids):, len(source_tasks):] = target.labels
        ids = tuple(source_ids) + target.sample_ids
        smiles = tuple(source_smiles) + target.smiles
        canonical = tuple(source_canonical) + target.canonical
    else:
        raise ContractError('unknown route')
    if np.asarray(source_labels).shape != (len(source_ids), len(source_tasks)):
        raise ContractError('source label shape mismatch')
    keep = np.flatnonzero(np.isfinite(labels).any(axis=1))
    return JointTrain(route, source_tasks + PRIMARY, tuple(ids[i] for i in keep),
        tuple(smiles[i] for i in keep), tuple(canonical[i] for i in keep), labels[keep], target.input_identity)


def verify_permissions(train, allowlist):
    expected = {}
    with Path(allowlist).open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['sample_id'], row['task'])
            role = 'target' if row['task'] in PRIMARY else 'source'
            if key in expected or row['role'] != role or not np.isfinite(row['label']):
                raise ContractError('duplicate/invalid allowed observation')
            expected[key] = row['label']
    actual = {(sid, train.tasks[j]): float(train.labels[i, j])
              for i, sid in enumerate(train.sample_ids)
              for j in np.flatnonzero(np.isfinite(train.labels[i]))}
    if actual != expected:
        raise ContractError('actual labels differ from frozen observation permissions')


def prepare(train):
    # Count shared canonical structures, not accidental duplicate record rows.
    keys = sorted(set(train.canonical))
    rows = {key: i for i, key in enumerate(keys)}
    graph_labels = np.full((len(keys), len(train.tasks)), np.nan)
    for i, canonical in enumerate(train.canonical):
        columns = np.flatnonzero(np.isfinite(train.labels[i]))
        if np.isfinite(graph_labels[rows[canonical], columns]).any():
            raise ContractError('repeated canonical/task observation needs adjudication')
        graph_labels[rows[canonical], columns] = train.labels[i, columns]
    adjacency, graph = task_adjacency(graph_labels)
    features, vocabulary = endpoint_feature_matrix(train.tasks, extend_vocabulary=True)
    scaler = TaskScaler.fit(train.labels, train.tasks, allow_empty=False)
    contract = dict(route=train.route, tasks=list(train.tasks), input_identity=train.input_identity,
        training_ids_sha256=semantic_digest(list(train.sample_ids)),
        training_labels_sha256=semantic_digest(np.where(np.isfinite(train.labels), train.labels, None).tolist()),
        vocabulary=vocabulary, adjacency=adjacency.tolist(), scaler=scaler.to_dict(),
        rows=len(train.sample_ids), observations=int(np.isfinite(train.labels).sum()), graph=graph,
        fingerprint='Avalon1024', audit_sha256=AUDIT_SHA)
    return adjacency, features, scaler, contract


def load_b115(*, route, audit_path, allowlist, csv_path, split_path, tox_manifest, datastore=None):
    check_bytes(audit_path, AUDIT_SHA)
    audit = json.loads(Path(audit_path).read_text(encoding='utf8'))
    if route not in {'A', 'B'}:
        raise ContractError('unknown route')
    for path, item in zip((csv_path, split_path, tox_manifest), audit['input']):
        check_bytes(path, item['sha256'])
    check_bytes(allowlist, audit[route]['allowlist']['sha256'])
    table = Dataset115Table.load(csv_path, split_path, tox_manifest,
                                expected_tox_sha=audit['input'][2]['sha256'])
    target = table.view(route, 'target', 'train')
    if route == 'A':
        source = table.view('A', 'source', 'train')
        train = assemble(route, source.tasks, source.sample_ids, source.smiles,
                         source.canonical, source.labels, target)
    else:
        if datastore is None:
            raise ContractError('B requires frozen source DataStore')
        build = Path(datastore)
        for name, item in zip(('datastore.json', 'labels.npy', 'index.npz'), audit['B']['source_assets']):
            check_bytes(build / name, item['sha256'])
        metadata = json.loads((build / 'datastore.json').read_text(encoding='utf8'))
        with np.load(build / 'index.npz', allow_pickle=False) as z:
            indices = np.flatnonzero(z['split_codes'] == 0)
            sample_ids = z['sample_ids']  # decompress once, not once per row
            ids = tuple(str(sample_ids[i]) for i in indices)
        refs = {r['sample_id']: r for r in json.loads(Path(tox_manifest).read_text(encoding='utf8'))['records']}
        records = [refs[sid] for sid in ids]
        if any(r['split'] != 'train' for r in records):
            raise ContractError('source split mismatch')
        tasks = tuple(ANIMAL_SOURCE_TASKS)
        columns = [metadata['task_names'].index(t) for t in tasks]
        labels = np.load(build / 'labels.npy', mmap_mode='r', allow_pickle=False)
        train = assemble(route, tasks, tuple('toxacute:' + s for s in ids),
            tuple(r['raw_smiles'] for r in records), tuple(r['canonical_smiles'] for r in records),
            np.array(labels[np.ix_(indices, columns)], dtype=np.float64), target)
    if train.tasks != tuple(audit[route]['tasks']) or train.input_identity != audit['data_identity']:
        raise ContractError('task/data identity differs from audit')
    verify_permissions(train, allowlist)
    adjacency, features, scaler, contract = prepare(train)
    if (contract['observations'] != audit[route]['observations'] or
            contract['graph']['undirected_edges'] != audit[route]['undirected_edges']):
        raise ContractError('recomputed graph/count differs from audited contract')
    for j, task in enumerate(train.tasks):
        old = audit[route]['scaler'][task]
        if old['n'] != int(scaler.counts[j]) or not np.allclose(
                [old['mean'], old['std']], [scaler.means[j], scaler.stds[j]], rtol=1e-12, atol=1e-12):
            raise ContractError('train scaler differs from audited contract')
    validation = table.view(route, 'target', 'validation')
    keep = np.flatnonzero(np.isfinite(validation.labels).any(axis=1))
    validation = LabelView(route, 'target', 'validation', PRIMARY,
        tuple(validation.sample_ids[i] for i in keep), tuple(validation.smiles[i] for i in keep),
        tuple(validation.canonical[i] for i in keep), tuple(validation.groups[i] for i in keep),
        validation.labels[keep], validation.input_identity)
    if set(train.canonical).intersection(validation.canonical):
        raise ContractError('train/validation canonical overlap')
    contract['validation_ids_sha256'] = semantic_digest(list(validation.sample_ids))
    contract['validation_labels_sha256'] = semantic_digest(
        np.where(np.isfinite(validation.labels), validation.labels, None).tolist())
    return train, validation, adjacency, features, scaler, contract
