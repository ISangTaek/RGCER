"""Materialize the approved 5-fold protocol from the accepted graph registry.

Reuses chemistry/cache identities; allocation sees label presence only. This
module does not build a cache, fit a model, or select cases using predictions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from .artifacts import read_json, save_view, write_json
from .cache import file_sha256
from .config import CONFIG_ROOT
from .data import make_training_view, sparse_visible_mask
from .preparation import read_source, write_tsv
from .splits import allocate_groups

PROTOCOL = 'PGTV2_FORMAL5_S3_20261026'
SPLIT_SEED = 20261026
TARGETS = ('man_oral_TDLo', 'women_oral_TDLo', 'human_oral_TDLo',
           'mammal (species unspecified)_subcutaneous_LD50', 'frog_subcutaneous_LDLo',
           'guinea pig_intravenous_LDLo')


def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def choose_by_group(indices, records, count, seed=SPLIT_SEED):
    """One record per accepted structure group, without inspecting label values."""
    groups = defaultdict(list)
    for i in indices:
        groups[records[i]['structure_group_id']].append(int(i))
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    chosen = []
    for pos in rng.permutation(len(names))[:count]:
        members = sorted(groups[names[pos]], key=lambda i: records[i]['record_id'])
        chosen.append(members[int(rng.integers(len(members)))])
    return chosen


def allocate_five(group_ids, observed):
    outer, outer_report = allocate_groups(group_ids, observed, fractions=[.2]*5,
                                         minimum_groups=[2]*5, seed=SPLIT_SEED)
    if not outer_report['coverage_pass']:
        raise ValueError('Five-fold endpoint coverage failed; do not fall back to three folds')
    folds, reports = {}, {}
    for fold in range(5):
        remain = np.asarray([outer[g] != fold for g in group_ids])
        inner, report = allocate_groups([g for g, keep in zip(group_ids, remain) if keep],
            observed[remain], fractions=[.85, .15], minimum_groups=[5, 2], seed=SPLIT_SEED)
        if not report['coverage_pass']:
            raise ValueError(f'Inner endpoint coverage failed in fold {fold}')
        folds[fold] = np.asarray(['test' if outer[g] == fold else
            'train' if inner[g] == 0 else 'validation' for g in group_ids])
        reports[fold] = report
        print(f'Allocated fold {fold}: {report["record_counts"]} non-test train/validation', flush=True)
    return folds, dict(outer=outer_report, inner=reports)


def bind_panels(root, records, labels, tasks, fold_roles, case_root):
    root.mkdir(parents=True, exist_ok=False)
    outer = {r['record_id']: next(f'S{f}' for f in range(5) if fold_roles[f'S{f}'][i] == 'test')
             for i, r in enumerate(records)}
    analogue = read_tsv(Path(case_root)/'analogue_explanation_pairs_reserved.tsv')
    reserved = set()
    for row in analogue:
        a, b = row['record_a'], row['record_b']
        if outer[a] != outer[b]:
            raise ValueError('Reserved analogue pair crosses scaffold test folds')
        row['fold'] = outer[a]
        row['weight_condition'] = 'E02a_S21_P10'
        reserved.update(((a, row['task']), (b, row['task'])))
    write_tsv(root/'analogue_pairs.tsv', analogue, list(analogue[0]))
    pair_population = []
    for row in read_tsv(Path(case_root)/'analogue_pair_candidates.tsv'):
        if row['task'] not in ('guinea pig_intravenous_LD50', 'rat_oral_LD50') or row['group_a'] != row['group_b']:
            continue
        if outer[row['record_a']] != outer[row['record_b']]:
            raise ValueError('Same-group analogue population crosses test folds')
        pair_population.append({**row, 'fold':outer[row['record_a']]})
    write_tsv(root/'analogue_prediction_population.tsv', pair_population, list(pair_population[0]))
    species_tasks = ('mouse_oral_LD50', 'rat_oral_LD50')
    eligible = np.flatnonzero((fold_roles['S0'] == 'test') &
        np.isfinite(labels[:, [tasks.index(t) for t in species_tasks]]).all(1))
    chosen = choose_by_group(eligible, records, 20)
    species = [dict(record_id=records[i]['record_id'], chemical_id=records[i]['chemical_id'],
        group_id=records[i]['structure_group_id'], fold='S0', task=t, weight_condition='E05_K10')
        for i in chosen for t in species_tasks]
    write_tsv(root/'species_explanation.tsv', species, list(species[0]))
    write_tsv(root/'species_prediction_population.tsv', [dict(record_id=records[i]['record_id'],
        group_id=records[i]['structure_group_id'], fold='S0') for i in eligible], ['record_id','group_id','fold'])
    # Reserve case observations before filling the general quota. Cases with
    # different checkpoints still count separately in the 180-observation cap.
    reserved.update((r['record_id'], r['task']) for r in species)
    general, coverage = [], []
    for task, quota in [(t, 15) for t in TARGETS] + [(t, 10) for t in
            ('mouse_oral_LD50', 'rat_oral_LD50', 'rabbit_skin_LD50')]:
        candidates = [i for i in np.flatnonzero(np.isfinite(labels[:, tasks.index(task)]))
                      if (records[i]['record_id'], task) not in reserved]
        chosen = choose_by_group(candidates, records, quota)
        for i in chosen:
            r = records[i]
            general.append(dict(record_id=r['record_id'], chemical_id=r['chemical_id'],
                group_id=r['structure_group_id'], task=task, fold=outer[r['record_id']],
                weight_condition='E02a_S21_P10'))
        coverage.append(dict(task=task, requested=quota, selected=len(chosen)))
    write_tsv(root/'general_explanation.tsv', general, list(general[0]))
    # Training-only references shared across model seeds and label budgets.
    references = {}
    for fold, roles in fold_roles.items():
        chosen = choose_by_group(np.flatnonzero(roles == 'train'), records, 256)
        ids = [records[i]['record_id'] for i in chosen]
        references[fold] = dict(ig_training_mean_ids=ids[:128], functional_relation_ids=ids)
    write_json(root/'training_references.json', references)
    summary = dict(general_observations=len(general), species_observations=len(species),
        analogue_observations=2*len(analogue), total_observations=len(general)+len(species)+2*len(analogue),
        species_prediction_compounds=len(eligible), species_prediction_groups=len({records[i]['structure_group_id'] for i in eligible}),
        general_task_coverage=coverage, selection_seed=SPLIT_SEED, predictions_used=False,
        analogue_test_folds={f'S{f}':sum(r['fold'] == f'S{f}' for r in analogue) for f in range(5)},
        analogue_prediction_pairs={t:sum(r['task'] == t for r in pair_population)
            for t in ('guinea pig_intravenous_LD50', 'rat_oral_LD50')},
        literature_review='PENDING; frozen structural cases are not yet toxicological mechanism evidence')
    write_json(root/'summary.json', summary)
    return summary


def prepare(source, accepted, case_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    accepted = Path(accepted)
    rules = read_json(CONFIG_ROOT/'preparation_rules.json')
    tasks = rules['candidate_task_names']
    if file_sha256(source) != rules['source_sha256']:
        raise ValueError('Source differs from the accepted toxacute.csv')
    raw, all_labels = read_source(source, tasks)
    identity = read_tsv(accepted/'record_identity.tsv')
    cache = read_json(accepted/'cache/cache_manifest.json')
    records = []
    for r in identity:
        row = int(r['raw_row_index']); offset = int(r['cache_offset'])
        if raw[row]['TAID'] != r['TAID'] or cache['records'][offset][1:4] != [r['record_id'], row, r['chemical_id']]:
            raise ValueError('Source row / accepted cache identity mismatch')
        records.append(dict(record_id=r['record_id'], raw_row_index=row, cache_offset=offset,
            chemical_id=r['chemical_id'], nonstereo_id=r['nonstereo_id'], structure_group_id=r['group_id']))
    labels = all_labels[[r['raw_row_index'] for r in records]]
    del raw, all_labels, identity
    observed = np.isfinite(labels)
    counts = observed.sum(0)
    sparse = [t for t, n in zip(tasks, counts) if n < 200]
    if len(tasks) != 59 or len(sparse) != 21:
        raise ValueError('The registered 59 endpoints / S21 population changed')
    write_tsv(output/'record_registry.tsv', records, list(records[0]))
    write_json(output/'endpoint_groups.json', dict(task_names=tasks, S21=sparse,
        M27=[t for t, n in zip(tasks, counts) if 200 <= n < 1000],
        L11=[t for t, n in zip(tasks, counts) if n >= 1000], full_counts=counts.tolist()))
    fold_roles, split_reports, view_rows, count_rows = {}, {}, [], []
    checks = dict(partition_group_overlap=0, partition_chemical_overlap=0,
                  nested_masks=True, source38_unchanged=True, heldout_labels_full=True)
    for split, key in [('R', 'nonstereo_id'), ('S', 'structure_group_id')]:
        print(f'Allocating {split}: {len(records)} records / {len(set(r[key] for r in records))} groups', flush=True)
        groups = [r[key] for r in records]
        roles_by_fold, split_reports[split] = allocate_five(groups, observed)
        view_records = [{**r, 'group_id': r[key]} for r in records]
        for number, roles in roles_by_fold.items():
            fold = f'{split}{number}'; fold_roles[fold] = roles
            previous = None
            for name, fraction in [('S21_P10', .1), ('S21_P30', .3), ('FULL', 1.)]:
                mask = sparse_visible_mask(observed=observed, task_names=tasks, roles=roles,
                    chemical_ids=[r['chemical_id'] for r in records], target_tasks=sparse,
                    fraction=fraction, fold_id=fold, subset_seed=1042)
                if previous is not None and np.any(previous & ~mask):
                    raise ValueError('Label views are not nested')
                source_cols = [i for i, t in enumerate(tasks) if t not in sparse]
                if not np.array_equal(mask[:, source_cols], observed[:, source_cols] & (roles == 'train')[:, None]):
                    raise ValueError('Source supervision was reduced')
                previous = mask
                view = make_training_view(labels=labels, label_task_names=tasks, task_names=tasks,
                    record_ids=[r['record_id'] for r in records], raw_row_indices=[r['raw_row_index'] for r in records],
                    group_ids=groups, chemical_ids=[r['chemical_id'] for r in records], roles=roles.tolist(),
                    visible_mask=mask, source_sha256=rules['source_sha256'], fold_id=fold)
                path = output/'views'/fold/name
                manifest = save_view(path, view, cache_manifest=cache, records=view_records, roles=roles, labels=labels, fold=fold)
                view_rows.append(dict(fold=fold, budget=name, view_id=view.view_id,
                    train_records=int((roles == 'train').sum()), active_train_records=manifest['active_train_rows'],
                    validation_records=int((roles == 'validation').sum()), test_records=int((roles == 'test').sum()),
                    train_labels=int(view.count.sum()), s21_train_labels=int(view.count[[tasks.index(t) for t in sparse]].sum()),
                    source38_train_labels=int(view.count[source_cols].sum()), validation_labels=int(observed[roles == 'validation'].sum()),
                    test_labels=int(observed[roles == 'test'].sum()), scale_fallback_tasks=int(view.scale_fallback.sum())))
                for role in ('train', 'validation', 'test'):
                    visible = mask if role == 'train' else observed & (roles == role)[:, None]
                    for j, task in enumerate(tasks):
                        selected = np.flatnonzero(visible[:, j])
                        count_rows.append(dict(fold=fold, budget=name, role=role, task=task,
                            size_group='S21' if counts[j] < 200 else 'M27' if counts[j] < 1000 else 'L11',
                            labels=len(selected), allocation_groups=len({groups[i] for i in selected}),
                            structure_groups=len({records[i]['structure_group_id'] for i in selected})))
                print(f'Saved {fold}/{name}: train labels={view.count.sum()}, S21={view_rows[-1]["s21_train_labels"]}', flush=True)
            # Save incremental role/report state, useful if interrupted before all views.
            np.savez_compressed(output/'fold_roles.npz', record_id=np.asarray([r['record_id'] for r in records]), **fold_roles)
            write_json(output/'allocation_reports.json', split_reports)
    if any(sum(fold_roles[f'{s}{f}'][i] == 'test' for f in range(5)) != 1
           for s in ('R', 'S') for i in range(len(records))):
        raise ValueError('Each split must test each record exactly once')
    write_tsv(output/'view_counts.tsv', view_rows, list(view_rows[0]))
    write_tsv(output/'endpoint_counts.tsv', count_rows, list(count_rows[0]))
    panels = bind_panels(output/'case_panels', records, labels, tasks, fold_roles, case_root)
    summary = dict(protocol_id=PROTOCOL, status='MATERIALIZED', records=len(records), labels=int(observed.sum()),
        tasks=59, folds_per_split=5, views=len(view_rows), split_seed=SPLIT_SEED, subset_seed=1042,
        init_seeds=[42,43,44], source_sha256=rules['source_sha256'], cache_id=cache['cache_id'], checks=checks,
        random_grouping='accepted nonstereo identity; no verified extra parent links in accepted registry',
        scaffold_grouping='accepted constitutional + Murcko / acyclic Butina groups',
        panels=panels, trained_models=0)
    write_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'accepted', 'case-root', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    prepare(args.source, args.accepted, args.case_root, args.output)
