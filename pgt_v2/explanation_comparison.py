"""Offline, post-development G02I metric revision; never reruns a model."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import read_json, write_json
from .fragment_explanations import attribution_scores, matched_pairs, pair_concordance
from .preparation import write_tsv


def recompute(evidence, output):
    evidence, output = Path(evidence), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    observations = read_json(evidence/'results/observations.json')
    structures = {p.stem: read_json(p) for p in (evidence/'results/structures').glob('*.json')}
    rows = []
    for obs in observations:
        fragments = structures[obs['record_id']]['fragments']
        pairs = matched_pairs(fragments)
        old = obs['fragment_evaluation']
        scores = {'common_perturbation': old['a_mean_absolute_response']}
        for reference, ig in obs['ig'].items():
            scores['ig_'+reference] = attribution_scores(ig['attributions'], fragments)
        if obs['attention'] is not None:
            scores['attention'] = attribution_scores(obs['attention'], fragments)
        for name, values in scores.items():
            ig = obs['ig'].get(name.removeprefix('ig_'), {})
            rows.append(dict(method=obs['method'], seed=obs['seed'], record_id=obs['record_id'],
                group_id=obs['group_id'], task=obs['task'], explanation=name, fixed_pair_count=len(pairs),
                pair_concordance=pair_concordance(values, old['b_mean_absolute_response'], pairs),
                old_capture_gain=old['metrics'][name]['capture_gain'],
                ig_unreliable=ig.get('unreliable'), ig_absolute_residual=abs(ig['completeness_residual']) if ig else None,
                ig_relative_residual=ig.get('normalized_residual')))
    write_tsv(output/'observation_scores.tsv', rows, list(rows[0]))
    # Common actual coverage across every method/seed in a given explainer.
    expected = defaultdict(set)
    cells = defaultdict(list)
    for row in rows:
        expected[row['explanation']].add((row['method'], row['seed']))
        cells[row['explanation'], row['record_id'], row['task']].append(row)
    shared = {key for key, values in cells.items() if
        {(v['method'], v['seed']) for v in values if v['pair_concordance'] is not None} == expected[key[0]]}
    aggregates = defaultdict(list)
    for row in rows:
        if (row['explanation'], row['record_id'], row['task']) in shared:
            aggregates[row['method'], row['seed'], row['explanation'], row['task']].append(row)
    task_rows = []
    for (method, seed, explainer, task), values in sorted(aggregates.items()):
        by_group = defaultdict(list)
        for v in values:
            by_group[v['group_id']].append(v['pair_concordance'])
        qualified = [v['pair_concordance'] for v in values if v['ig_unreliable'] is not True]
        task_rows.append(dict(method=method, seed=seed, explanation=explainer, task=task,
            observations=len(values), structure_groups=len(by_group),
            mean_concordance=float(np.mean([v['pair_concordance'] for v in values])),
            group_mean_concordance=float(np.mean([np.mean(g) for g in by_group.values()])),
            unreliable=sum(v['ig_unreliable'] is True for v in values),
            qualified_mean=float(np.mean(qualified)) if qualified else None))
    write_tsv(output/'task_scores.tsv', task_rows, list(task_rows[0]))
    summaries = []
    for explainer, method_seeds in sorted(expected.items()):
        for method in sorted({m for m, _ in method_seeds}):
            seeds = []
            for seed in sorted(s for m, s in method_seeds if m == method):
                values = [r for r in task_rows if (r['method'], r['seed'], r['explanation']) == (method, seed, explainer)]
                seeds.append(dict(seed=seed, tasks=len(values), observations=sum(v['observations'] for v in values),
                    macro_concordance=float(np.mean([v['mean_concordance'] for v in values])),
                    group_macro_concordance=float(np.mean([v['group_mean_concordance'] for v in values]))))
            summaries.append(dict(method=method, explanation=explainer, seeds=seeds,
                macro_concordance=float(np.mean([s['macro_concordance'] for s in seeds])),
                group_macro_concordance=float(np.mean([s['group_macro_concordance'] for s in seeds]))))
    coverage = []
    first = {(o['record_id'], o['task']): o for o in observations}
    for task in sorted({o['task'] for o in observations}):
        obs = [o for (_, t), o in first.items() if t == task]
        eligible = [o for o in obs if matched_pairs(structures[o['record_id']]['fragments'])]
        actual = [o for o in obs if ('common_perturbation', o['record_id'], task) in shared]
        coverage.append(dict(task=task, total=len(obs), structural_eligible=len(eligible),
            common_actual=len(actual), common_groups=len({o['group_id'] for o in actual})))
    flags = []
    for method in ('N1', 'N1S'):
        for reference in ('zero', 'training_mean'):
            values = [o['ig'][reference] for o in observations if o['method'] == method]
            flags.append(dict(method=method, reference=reference, observations=len(values),
                unreliable=sum(v['unreliable'] for v in values)))
    report = dict(status='OFFLINE_DEVELOPMENT_REANALYSIS', new_model_inference=0, new_training=0,
        source_observations=len(observations), distinct_observations=len(first), coverage=coverage,
        summaries=summaries, original_ig_flags=flags,
        metric='observation mean of sign(s_i-s_j)*sign(b_i-b_j) over all fixed size/aromaticity matched pairs; ties=0',
        interpretation='Descriptive post-development revision; old capture_gain and adverse findings retained; not confirmatory evidence')
    write_json(output/'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    recompute(args.evidence, args.output)
