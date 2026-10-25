"""G02R fixed small-task pilot. Uses existing sparse training labels and no test labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .artifacts import load_arrays, load_manifest, save_view, write_json
from .chemistry import digest
from .config import resolve_run
from .data import make_training_view

TARGETS = (
    'man_oral_TDLo', 'women_oral_TDLo', 'human_oral_TDLo',
    'mammal (species unspecified)_subcutaneous_LD50',
    'frog_subcutaneous_LDLo', 'guinea pig_intravenous_LDLo',
)
SOURCES = (
    'mouse_oral_LD50', 'rat_oral_LD50', 'mouse_subcutaneous_LD50',
    'rat_subcutaneous_LD50', 'guinea pig_intravenous_LD50',
    'guinea pig_subcutaneous_LDLo',
)
PILOT_METHODS = ('M3', 'M5', 'M6', 'N0', 'N1', 'N2', 'N3', 'N4')
SOURCE_CAP = 512
SOURCE_SUBSET_SEED = 20261025


def select_pilot_arrays(train, validation, task_names):
    """Retain all predeclared target observations; cap sources by chemical ID."""
    names = SOURCES + TARGETS
    columns = [task_names.index(name) for name in names]
    labels = train['labels'][:, columns].copy()
    for index, _ in enumerate(SOURCES):
        observed = np.isfinite(labels[:, index])
        identities = np.unique(train['chemical_id'][observed])
        rng = np.random.default_rng(SOURCE_SUBSET_SEED + index)
        selected = rng.choice(identities, size=min(SOURCE_CAP, len(identities)), replace=False)
        labels[~np.isin(train['chemical_id'], selected), index] = np.nan
    active = np.isfinite(labels).any(1)
    train_small = {key: value[active].copy() for key, value in train.items() if key != 'labels'}
    train_small['labels'] = labels[active]
    truth = validation['labels'][:, columns].copy()
    active_val = np.isfinite(truth).any(1)
    val_small = {key: value[active_val].copy() for key, value in validation.items() if key != 'labels'}
    val_small['labels'] = truth[active_val]
    return train_small, val_small


def prepare_pilot(parent_view, output):
    parent = load_manifest(parent_view)
    train = load_arrays(parent_view, 'train.npz', parent)
    validation = load_arrays(parent_view, 'validation.npz', parent)
    train, validation = select_pilot_arrays(train, validation, parent['task_names'])
    tasks = SOURCES + TARGETS
    labels = np.concatenate([train['labels'], validation['labels']])
    roles = ['train'] * len(train['labels']) + ['validation'] * len(validation['labels'])
    records = []
    for part in (train, validation):
        for i in range(len(part['record_id'])):
            records.append({key: (int(part[key][i]) if key in ('raw_row_index', 'cache_offset') else str(part[key][i]))
                            for key in ('record_id', 'cache_offset', 'raw_row_index', 'chemical_id', 'group_id')})
    visible = np.isfinite(labels) & (np.asarray(roles) == 'train')[:, None]
    view = make_training_view(labels=labels, label_task_names=tasks, task_names=tasks,
        record_ids=[r['record_id'] for r in records], raw_row_indices=[r['raw_row_index'] for r in records],
        group_ids=[r['group_id'] for r in records], chemical_ids=[r['chemical_id'] for r in records],
        roles=roles, visible_mask=visible, source_sha256=parent['source_sha256'],
        fold_id=parent['fold'] + '_G02R_mini12')
    manifest = save_view(output, view, cache_manifest={'cache_id': parent['cache_id']},
        records=records, roles=roles, labels=labels, fold=parent['fold'])
    manifest.update(parent_view_id=parent['view_id'], pilot='G02R_mini12',
        target_tasks=list(TARGETS), source_tasks=list(SOURCES), source_chemical_cap=SOURCE_CAP,
        source_subset_seed=SOURCE_SUBSET_SEED, test_population_included=False)
    manifest['manifest_sha256'] = digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'})
    write_json(Path(output) / 'manifest.json', manifest)
    summary = dict(parent_view_id=parent['view_id'], view_id=manifest['view_id'],
        active_training_records=len(train['labels']), validation_records=len(validation['labels']),
        training_labels=int(np.isfinite(train['labels']).sum()),
        validation_labels=int(np.isfinite(validation['labels']).sum()),
        target_training_labels=int(np.isfinite(train['labels'][:, len(SOURCES):]).sum()),
        target_validation_labels=int(np.isfinite(validation['labels'][:, len(SOURCES):]).sum()),
        tasks=[dict(task=name, train_labels=int(view.count[i]), train_groups=int(view.group_count[i]),
            train_chemicals=len(set(train['chemical_id'][np.isfinite(train['labels'][:, i])])),
            validation_labels=int(np.isfinite(validation['labels'][:, i]).sum())) for i,name in enumerate(tasks)])
    write_json(Path(output) / 'selection_summary.json', summary)
    return summary


def resolve_pilot(view_root, output, code_commit):
    manifest = load_manifest(view_root)
    if manifest.get('pilot') != 'G02R_mini12' or manifest['task_names'] != list(SOURCES + TARGETS):
        raise ValueError('G02R requires its fixed twelve-task view')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    order = []
    for method in PILOT_METHODS:
        config_id = 'C0' if method == 'M3' else 'C2'
        for seed in (42, 43):
            run_id = f'PGTV2_G02R_{method}_F0_{config_id}_S{seed}_MINI12'
            row = dict(run_id=run_id, method_id=method, config_id=config_id, init_seed=seed,
                       fold='F0', phase='P2', state='READY', code_commit=code_commit)
            run = resolve_run(row, view_manifest=manifest)
            run['config'].update(max_epochs=120, min_epochs=40, patience=25, selection_tasks=list(TARGETS))
            run['config_sha256'] = digest(run['config'])
            run['run_sha256'] = digest({k:v for k,v in run.items() if k != 'run_sha256'})
            write_json(output / (run_id + '.json'), run)
            order.append(dict(run_id=run_id, method=method, seed=seed, config=config_id,
                              run_file=run_id + '.json'))
    write_json(output / 'run_order.json', order)
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--parent-view', required=True)
    prepare.add_argument('--output', required=True)
    resolve = commands.add_parser('resolve')
    resolve.add_argument('--view', required=True)
    resolve.add_argument('--output', required=True)
    resolve.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_pilot(args.parent_view, args.output)
    else:
        result = resolve_pilot(args.view, args.output, args.commit)
    print(result, flush=True)


if __name__ == '__main__':
    main()
