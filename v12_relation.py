"""C1 prerequisite: grouped four-observation relations inside source banks.

Observed source query labels are deliberately used. This diagnoses relationships,
not a deployable C1 predictor, source-teacher reliability, or human performance.
"""
from itertools import combinations
import hashlib

import numpy as np

from v12_analytic import array, require
from v12_s1_data import fold_id, digest

SPEC = dict(folds=3, min_shared_molecules=12, min_shared_groups=6,
            min_train_molecules=6, min_holdout_molecules=3, min_holdout_groups=2,
            max_train_molecules=128, max_holdout_molecules=64,
            pair_offsets=[1, 7], ridge_penalty=1., condition_seed=42,
            eligibility='same_endpoint_and_route_within_one_source_bank',
            conditions=['Morgan_Tanimoto', 'mean_MolLogP', 'mean_TPSA', 'abs_MolWt_change'],
            label_scale='each_task_standardized_on_this_fold_training_shared_molecules',
            scope='OBSERVED_LABEL_RELATION_ONLY_NOT_DEPLOYABLE_C1_OR_EXTERNAL_VALIDATION')
MODELS = ('GLOBAL', 'CONDITIONAL', 'SHUFFLED_CONDITION')


def order_key(text):
    return hashlib.sha256(('V12S1/pairs/42/'+text).encode()).hexdigest()


def make_pairs(molecules, limit):
    ordered = sorted(set(molecules), key=lambda x: (order_key(x), x))[:limit]
    require(len(ordered) >= 3, 'pair population requires three molecules')
    pairs = set()
    for i, mol in enumerate(ordered):
        for offset in SPEC['pair_offsets']:
            other = ordered[(i+offset) % len(ordered)]
            if other != mol:
                pairs.add(tuple(sorted((mol, other))))
    # Ring offsets bound the degree of every molecule by four.
    return sorted(pairs)


def group_weights(pairs, groups):
    counts = {}
    for a, b in pairs:
        for m in (a, b):
            counts[groups[m]] = counts.get(groups[m], 0) + 1
    return np.array([1/counts[groups[a]] + 1/counts[groups[b]] for a, b in pairs])


def conditions(scene, pairs, mapping):
    names = scene.descriptor_names
    cols = [names.index(n) for n in ('MolLogP', 'TPSA', 'MolWt')]
    result = []
    for a, b in pairs:
        i, j = mapping[a], mapping[b]
        require(not scene.missing[[i, j]][:, cols].any(), 'missing relation descriptor')
        ai, bi = scene.fp_packed[i], scene.fp_packed[j]
        intersection = int(np.unpackbits(np.bitwise_and(ai, bi)).sum())
        union = int(np.unpackbits(np.bitwise_or(ai, bi)).sum())
        av, bv = scene.descriptors[i, cols], scene.descriptors[j, cols]
        result.append([intersection/union if union else 1., (av[0]+bv[0])/2,
                       (av[1]+bv[1])/2, abs(av[2]-bv[2])])
    return np.asarray(result, dtype=np.float64)


def solve_slope(design, response, weight, penalty=1.):
    x, y, w = array(design, 2), array(response, 1), array(weight, 1)
    require(len(x) == len(y) == len(w) and len(x) >= 2 and (w > 0).all()
            and np.isfinite(penalty) and penalty > 0, 'slope dimensions/weights')
    return np.linalg.solve(x.T @ (w[:, None]*x) + penalty*np.eye(x.shape[1]), x.T @ (w*y))


def fit_relation(source_delta, target_delta, chemical_conditions, weights):
    ds, dt = array(source_delta, 1), array(target_delta, 1)
    c = array(chemical_conditions, 2)
    require(c.shape == (len(ds), 4) and len(dt) == len(ds), 'relation dimensions')
    center, scale = c.mean(0), c.std(0)
    scale = np.where(scale > 1e-6, scale, 1.)
    centered = (c-center)/scale
    rng = np.random.default_rng(SPEC['condition_seed'])
    shuffled = centered[rng.permutation(len(centered))]
    designs = dict(GLOBAL=ds[:, None], CONDITIONAL=ds[:, None]*np.column_stack((np.ones(len(ds)), centered)),
                   SHUFFLED_CONDITION=ds[:, None]*np.column_stack((np.ones(len(ds)), shuffled)))
    return dict(center=center.tolist(), scale=scale.tolist(),
                coefficients={m: solve_slope(x, dt, weights, SPEC['ridge_penalty']).tolist() for m, x in designs.items()})


def predict_relation(state, source_delta, chemical_conditions):
    ds, c = array(source_delta, 1), array(chemical_conditions, 2)
    require(c.shape == (len(ds), 4), 'relation prediction dimensions')
    centered = (c-np.asarray(state['center']))/np.asarray(state['scale'])
    designs = {m: ds[:, None] if m == 'GLOBAL' else ds[:, None]*np.column_stack((np.ones(len(ds)), centered)) for m in MODELS}
    result = {m: designs[m] @ np.asarray(state['coefficients'][m]) for m in MODELS}
    require(all(np.isfinite(x).all() for x in result.values()), 'finite relation predictions')
    return result


def bank_index(scene):
    tasks, groups = {t: {} for t in scene.sources}, {}
    for i in scene.indices(scene.sources):
        r = scene.rows[i]
        require(r['canonical'] not in tasks[r['task']], 'duplicate four-cell label')
        tasks[r['task']][r['canonical']] = r['label']
        groups[r['canonical']] = r['group']
    mapping = {c: i for i, c in enumerate(scene.canonical)}
    cols = [scene.descriptor_names.index(n) for n in ('MolLogP', 'TPSA', 'MolWt')]
    eligible, excluded = [], []
    for left, right in combinations(sorted(scene.sources), 2):
        # First-stage scope is intentionally narrower than all cross-endpoint semantics.
        if left.rsplit('_', 2)[1:] != right.rsplit('_', 2)[1:]:
            excluded.append(dict(left=left, right=right, reason='endpoint_or_route_differs_not_semantically_qualified'))
            continue
        common = sorted(set(tasks[left]) & set(tasks[right]))
        common = [m for m in common if not scene.missing[mapping[m], cols].any()]
        if len(common) < SPEC['min_shared_molecules'] or len({groups[m] for m in common}) < SPEC['min_shared_groups']:
            excluded.append(dict(left=left, right=right, reason='insufficient_shared_molecules_or_groups', n=len(common)))
            continue
        folds = []
        for fold in range(3):
            train = [m for m in common if fold_id(groups[m]) != fold]
            held = [m for m in common if fold_id(groups[m]) == fold]
            if len(train) < SPEC['min_train_molecules'] or len(held) < SPEC['min_holdout_molecules'] or len({groups[m] for m in held}) < SPEC['min_holdout_groups']:
                continue
            folds.append(dict(fold=fold, train_molecules=train, holdout_molecules=held))
        if len(folds) != 3:
            excluded.append(dict(left=left, right=right, reason='incomplete_three_fold_coverage', n=len(common)))
            continue
        eligible.append(dict(left=left, right=right, common=common, folds=folds))
    return tasks, groups, mapping, eligible, excluded


def qualification(scene):
    _, _, _, eligible, excluded = bank_index(scene)
    included = sorted({p[k] for p in eligible for k in ('left', 'right')})
    return dict(spec=SPEC, source_tasks=len(scene.sources), eligible_task_pairs=len(eligible),
                directed_fold_cases=6*len(eligible), represented_source_tasks=included,
                unrepresented_source_tasks=sorted(set(scene.sources)-set(included)), exclusions=excluded,
                eligible=[dict(left=p['left'], right=p['right'], shared_molecules=len(p['common']),
                               folds=[dict(fold=f['fold'], train=len(f['train_molecules']), holdout=len(f['holdout_molecules'])) for f in p['folds']]) for p in eligible])


def cases(scene):
    labels, groups, mapping, eligible, _ = bank_index(scene)
    for taskpair in eligible:
        for source, target in ((taskpair['left'], taskpair['right']), (taskpair['right'], taskpair['left'])):
            for fold in taskpair['folds']:
                train = make_pairs(fold['train_molecules'], SPEC['max_train_molecules'])
                held = make_pairs(fold['holdout_molecules'], SPEC['max_holdout_molecules'])
                train_mol = sorted({m for pair in train for m in pair})
                train_groups = {groups[m] for m in train_mol}
                held_groups = {groups[m] for pair in held for m in pair}
                require(train_groups.isdisjoint(held_groups), 'pair endpoint group leakage')
                scalers = {t: dict(mean=float(np.mean([labels[t][m] for m in train_mol])),
                                  std=max(float(np.std([labels[t][m] for m in train_mol])), 1e-6)) for t in (source, target)}
                def delta(pairs, task):
                    return np.array([(labels[task][a]-labels[task][b])/scalers[task]['std'] for a, b in pairs])
                ds, dt = delta(train, source), delta(train, target)
                cs = conditions(scene, train, mapping)
                weight = group_weights(train, groups)
                state = fit_relation(ds, dt, cs, weight)
                hs, ht, hc = delta(held, source), delta(held, target), conditions(scene, held, mapping)
                pred = predict_relation(state, hs, hc)
                hw = group_weights(held, groups)
                score = {m: float(np.sqrt(np.average((p-ht)**2, weights=hw))) for m, p in pred.items()}
                yield dict(source=source, target=target, fold=fold['fold'], state=state, scalers=scalers,
                           train_pairs_sha256=digest(train), train_pair_count=len(train),
                           train_unique_molecules=len(train_mol), train_unique_groups=len(train_groups),
                           holdout_unique_groups=len(held_groups), holdout_pair_count=len(held),
                           train_likelihood_weight=float(weight.sum()), scores=score,
                           predictions=[dict(left=a, right=b, left_group=groups[a], right_group=groups[b],
                                             source_delta=float(hs[i]), target_delta=float(ht[i]),
                                             weight=float(hw[i]), predictions={m: float(p[i]) for m, p in pred.items()})
                                        for i, (a, b) in enumerate(held)])
