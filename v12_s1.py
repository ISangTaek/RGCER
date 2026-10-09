"""V12-S1: source-asset qualification, C3 analytic controls and C1 relations.

This CPU-only mechanism stage is not the subsequent ten-method screen. Real
fits run on the experiment server; local tests use synthetic inputs only.
"""
from pathlib import Path
import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import v12_analytic as analytic
import v12_relation as relation
import v12_s1_data as data

read, write, sha, digest, require = data.read, data.write, data.sha, data.digest, analytic.require
REPO = Path(__file__).resolve().parent
TASK = 'V12_S1_C3_ANALYTIC_AND_C1_RELATION_20261009'
CODE_FILES = ('v12_analytic.py', 'v12_relation.py', 'v12_s1_data.py', 'v12_s1.py',
              'configs/v12_s1_input_lock.json', 'docs/V12_S1_PROTOCOL.md',
              'tests/test_v12_analytic.py', 'tests/test_v12_relation.py', 'tests/test_v12_s1.py')
TESTS = CODE_FILES[-3:]
SPEC = dict(task=TASK, seed=42, stages=['S0_qualification', 'S1_mechanism'],
            scenes=list(data.SCENES), methods=list(analytic.METHODS), penalties=list(analytic.LAMBDAS),
            source_rank=8, fingerprint_projection=32, source_prior_ridge=10., covariance_diagonal_shrinkage=.5,
            covariance_floor=.01, residual_precision_multiplier=10., target_folds=3,
            c3_label_cv='TARGET_LABEL_HOLDOUT_CONDITIONAL_ON_FIXED_SOURCE_ASSET',
            selection='minimum_pooled_target_train_group_CV_macro_RMSE_first_configuration_on_tie',
            reported_cv='TUNING_SCORE_NOT_UNBIASED_OUTER_EVALUATION',
            validation='previously_exposed_development_population_no_new_test',
            target_endpoint_model_fits=520, source_task_prior_fits=216,
            source_neural_training=0, target_neural_training=0, gpu_required=False,
            relation_banks=['ToxAcute', 'A'], relation_spec=relation.SPEC,
            relation_counts={'ToxAcute': {'eligible_task_pairs': 74, 'directed_fold_cases': 444, 'represented_source_tasks': 41},
                             'A': {'eligible_task_pairs': 88, 'directed_fold_cases': 528, 'represented_source_tasks': 64}},
            scientific_acceptance='PENDING_CODEX_REVIEW')


def code_identity(expected_commit):
    require(isinstance(expected_commit, str) and len(expected_commit) == 40
            and all(c in '0123456789abcdef' for c in expected_commit), 'full expected commit required')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    require(actual == expected_commit, 'execution HEAD differs')
    identity = {}
    for name in CODE_FILES:
        blob = subprocess.check_output(['git', 'show', expected_commit+':'+name], cwd=REPO)
        current = (REPO/name).read_bytes().replace(b'\r\n', b'\n')
        require(blob.replace(b'\r\n', b'\n') == current, 'execution code changed: '+name)
        import hashlib
        identity[name] = hashlib.sha256(current).hexdigest()
    return dict(commit=actual, code_sha256_lf=identity, input_lock_sha256=sha(data.LOCK), spec=SPEC)


def source_bank_equivalence(left, right):
    def records(scene):
        return sorted([(r['task'], r['canonical'], r['group'], r['label'])
                       for r in scene.rows if r['task'] in scene.sources])
    require(left.sources == right.sources and records(left) == records(right), 'Tox/B source bank labels differ')
    molecules = sorted({r['canonical'] for r in left.rows if r['task'] in left.sources})
    li, ri = {c:i for i,c in enumerate(left.canonical)}, {c:i for i,c in enumerate(right.canonical)}
    a, b = [li[c] for c in molecules], [ri[c] for c in molecules]
    require(left.descriptor_names == right.descriptor_names and
            all(np.array_equal(getattr(left,k)[a], getattr(right,k)[b]) for k in ('fp_packed','descriptors','missing')),
            'Tox/B source chemistry differs')
    return dict(identical_observation_digest=digest(records(left)), molecules=len(molecules),
                relation_reused_for_B=True, scope='same_source_diagnostic_not_B_target_evidence')


def preflight(reference_root, expected_commit):
    ident = code_identity(expected_commit)
    locked = data.check_inputs(reference_root)
    scenes = {s: data.load_scene(reference_root, s, locked=locked) for s in data.SCENES}
    equivalence = source_bank_equivalence(scenes['ToxAcute'], scenes['B'])
    qualified = {s: relation.qualification(scenes[s]) for s in SPEC['relation_banks']}
    for name, expected in SPEC['relation_counts'].items():
        actual = {k: len(qualified[name][k]) if k == 'represented_source_tasks' else qualified[name][k] for k in expected}
        require(actual == expected, 'frozen relationship qualification matrix: '+name)
    return dict(identity=ident, inputs_checked=len(locked['files']),
                scenes={s: x.qualification() for s,x in scenes.items()},
                relations=qualified,
                source_bank_equivalence=equivalence, real_fitting_executed=False,
                source_crossfit_asset='CURRENT_LOCKED_ASSETS_NOT_STRICT_GROUP_OOF',
                current_scope_can_proceed_without_new_teacher=True)


def prepare_analytic(scene):
    q, chem_identity = data.chemistry_features(scene)
    source = scene.indices(scene.sources)
    # Fit preprocessing to unique source molecules, not repeated task observations.
    by_canonical = {}
    for i in source:
        by_canonical.setdefault(scene.rows[i]['canonical'], int(i))
    unique = [by_canonical[c] for c in sorted(by_canonical)]
    transform = analytic.SourceTransform.fit(scene.functions[unique], q[scene.row_chemical[unique]], SPEC['source_rank'])
    z, chemical = transform.transform(scene.functions, q[scene.row_chemical])
    prior = analytic.source_prior(z[source], [scene.rows[i]['label'] for i in source],
                                  [scene.rows[i]['task'] for i in source], scene.sources)
    shared = dict(transform=transform.state(), prior=prior, chemistry=chem_identity,
                  source_unique_rows_sha256=digest([scene.rows[i] for i in unique]),
                  source_fit_population_sha256=digest([scene.rows[i] for i in source]))
    return z, chemical, shared


def case_plan(scene):
    for method in analytic.METHODS:
        for configuration, penalty in enumerate(analytic.LAMBDAS):
            for fold in (0, 1, 2, None):
                suffix = 'full' if fold is None else f'fold{fold}'
                yield dict(id=f'{method}_c{configuration}_{suffix}', method=method,
                           configuration=configuration, penalty=penalty, fold=fold)


def target_case(scene, case, z, q, prior):
    models, predictions, expected, training = {}, [], [], {}
    for task in scene.targets:
        ids = scene.indices([task])
        if case['fold'] is None:
            fit = ids
            query = scene.indices([task], 'validation')
        else:
            fit = np.array([i for i in ids if data.fold_id(scene.rows[i]['group']) != case['fold']], dtype=np.int64)
            query = np.array([i for i in ids if data.fold_id(scene.rows[i]['group']) == case['fold']], dtype=np.int64)
        require(len(fit) >= 2 and len(query) > 0 and
                {scene.rows[i]['group'] for i in fit}.isdisjoint({scene.rows[i]['group'] for i in query}),
                'target-label fold coverage/group exclusion')
        model = analytic.fit_model(z[fit], q[fit], [scene.rows[i]['label'] for i in fit], prior,
                                   case['method'], case['penalty'])
        models[task] = model
        prediction = analytic.predict_model(model, z[query], q[query])
        current = [scene.rows[i] for i in query]
        expected.extend(current)
        predictions.extend(dict(row, prediction=float(value)) for row,value in zip(current,prediction))
        training[task] = dict(observations=len(fit), groups=len({scene.rows[i]['group'] for i in fit}),
                              fit_identity_sha256=digest([scene.rows[i] for i in fit]),
                              query_identity_sha256=digest(current))
    return dict(case=case, models=models, training=training, predictions=predictions,
                score=data.metrics(predictions, expected), scope=SPEC['c3_label_cv'] if case['fold'] is not None else SPEC['validation'])


def select_cases(scene, results):
    expected = [scene.rows[i] for i in scene.indices(scene.targets)]
    selected = {}
    for method in analytic.METHODS:
        options = []
        for config in range(len(analytic.LAMBDAS)):
            rows = [r for fold in range(3) for r in results[f'{method}_c{config}_fold{fold}']['predictions']]
            score = data.metrics(rows, expected)
            options.append(dict(configuration=config, cv_score=score,
                                validation_score=results[f'{method}_c{config}_full']['score']))
        best = min(options, key=lambda r: (r['cv_score']['macro_rmse'], r['configuration']))
        selected[method] = dict(configuration=best['configuration'], cv_score=best['cv_score'],
                                validation_score=best['validation_score'], options=options)
    return selected


def run_analytic(scene, folder):
    folder.mkdir(parents=True, exist_ok=False)
    z, q, shared = prepare_analytic(scene)
    write(folder/'shared.json', shared)
    results = {}
    for case in case_plan(scene):
        result = target_case(scene, case, z, q, shared['prior'])
        write(folder/(case['id']+'.json'), result)
        results[case['id']] = result
    selected = select_cases(scene, results)
    write(folder/'selection.json', selected)
    write(folder/'folds.json', [dict(task=r['task'], sample_id=r['sample_id'], canonical=r['canonical'],
                                    group=r['group'], fold=data.fold_id(r['group']))
                               for r in scene.rows if r['task'] in scene.targets and r['split'] == 'train'])
    return dict(selection=selected, endpoint_fits=len(results)*len(scene.targets),
                prior_task_fits=len(scene.sources), all_source_tasks_fitted=True, all_target_tasks_fitted=True)


def run_relations(scene, folder):
    folder.mkdir(parents=True, exist_ok=False)
    eligible = relation.qualification(scene)
    write(folder/'qualification.json', eligible)
    summary = []
    with (folder/'cases.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for case in relation.cases(scene):
            stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False)+'\n')
            summary.append({k: case[k] for k in ('source','target','fold','scores','train_unique_groups','holdout_unique_groups','holdout_pair_count')})
    require(len(summary) == eligible['directed_fold_cases'], 'relation case count')
    result = relation_summary(summary)
    write(folder/'summary.json', result)
    return dict(case_count=len(summary), eligible_task_pairs=eligible['eligible_task_pairs'])


def relation_summary(summary):
    return dict(scope=relation.SPEC['scope'], cases=summary, case_count=len(summary),
                conditional_minus_global_mean=float(np.mean([x['scores']['CONDITIONAL']-x['scores']['GLOBAL'] for x in summary])) if summary else None,
                empty_reason=None if summary else 'NO_ELIGIBLE_SOURCE_RELATIONS',
                interpretation='descriptive_task_pair_fold_average_not_independent_sample_inference')


def historical_reference(reference_root, scene):
    entry = read(data.LOCK)['scenes'][scene.name]
    path = data.safe_file(reference_root, entry['reference_predictions'])
    require(sha(path) == read(data.LOCK)['files'][entry['reference_predictions']], 'historical prediction SHA')
    expected = [scene.rows[i] for i in scene.indices(scene.targets, 'validation')]
    return dict(name=path.parent.name, score=data.metrics(read(path), expected), sha256=sha(path),
                scope='HISTORICAL_DEVELOPMENT_ENVELOPE_NOT_ONE_UNIFIED_ARCHITECTURE')


def run(reference_root, output, expected_commit):
    require(not output.exists(), 'existing output; do not overwrite or implicitly resume')
    audit = preflight(reference_root, expected_commit)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'preflight.json', audit)
    start = time.monotonic()
    summary = {}
    for scene_name in data.SCENES:
        print('Analytic mechanism stage: '+scene_name, flush=True)
        scene = data.load_scene(reference_root, scene_name)
        summary[scene_name] = run_analytic(scene, output/'analytic'/scene_name)
        summary[scene_name]['historical_reference'] = historical_reference(reference_root, scene)
        if scene_name in SPEC['relation_banks']:
            print('Observed source relations: '+scene_name, flush=True)
            summary[scene_name]['relations'] = run_relations(scene, output/'relations'/scene_name)
    require(sum(v['endpoint_fits'] for v in summary.values()) == SPEC['target_endpoint_model_fits']
            and sum(v['prior_task_fits'] for v in summary.values()) == SPEC['source_task_prior_fits'], 'fixed full fit matrix')
    receipt = dict(identity=audit['identity'], summary=summary,
                   environment=dict(python=sys.version, numpy=np.__version__, torch=str(torch.__version__),
                                    platform=platform.platform(), device='cpu', numerical_threads=1),
                   elapsed_seconds=time.monotonic()-start, test_evaluated=False,
                   execution_status='FINISHED', content_status='PENDING_INDEPENDENT_REPLAY',
                   acceptance_status='PENDING_CODEX_REVIEW')
    write(output/'receipt.json', receipt)
    write(output/'checksums.json', {p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    return receipt


def close_numbers(actual, expected, path='root'):
    require(type(actual) == type(expected), 'replay type: '+path)
    if isinstance(actual, dict):
        require(set(actual) == set(expected), 'replay keys: '+path)
        for key in actual:
            close_numbers(actual[key], expected[key], path+'/'+key)
    elif isinstance(actual, list):
        require(len(actual) == len(expected), 'replay length: '+path)
        for i, (a, b) in enumerate(zip(actual, expected)):
            close_numbers(a, b, path+'/'+str(i))
    elif isinstance(actual, float):
        require(np.isfinite(actual) and np.isfinite(expected) and np.isclose(actual, expected, rtol=1e-8, atol=1e-8),
                'replay value: '+path)
    else:
        require(actual == expected, 'replay identity: '+path)


def verify(reference_root, output, expected_commit):
    audit = preflight(reference_root, expected_commit)
    close_numbers(read(output/'preflight.json'), audit)
    receipt = read(output/'receipt.json')
    require(receipt['identity'] == audit['identity'] and receipt['test_evaluated'] is False
            and receipt['execution_status'] == 'FINISHED' and receipt['content_status'] == 'PENDING_INDEPENDENT_REPLAY'
            and receipt['acceptance_status'] == 'PENDING_CODEX_REVIEW', 'receipt identity/status')
    manifest = read(output/'checksums.json')
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    require(actual_files == set(manifest) | {'checksums.json'}, 'run output population')
    for name, expected in manifest.items():
        require(sha(data.safe_file(output, name)) == expected, 'run output SHA: '+name)
    expected_files = {'preflight.json', 'receipt.json'}
    for scene_name in data.SCENES:
        expected_files.update(f'analytic/{scene_name}/{name}.json' for name in ('shared','selection','folds'))
        expected_files.update(f'analytic/{scene_name}/{method}_c{config}_{suffix}.json'
                              for method in analytic.METHODS for config in range(2)
                              for suffix in ('fold0','fold1','fold2','full'))
    for name in SPEC['relation_banks']:
        expected_files.update(f'relations/{name}/{file}' for file in ('qualification.json','cases.jsonl','summary.json'))
    require(set(manifest) == expected_files, 'fixed output file allowlist')
    replayed_fits, relation_count = 0, 0
    for name in data.SCENES:
        scene = data.load_scene(reference_root, name)
        folder = output/'analytic'/name
        z, q, shared = prepare_analytic(scene)
        close_numbers(read(folder/'shared.json'), shared)
        results = {}
        for case in case_plan(scene):
            saved = read(folder/(case['id']+'.json'))
            fresh = target_case(scene, case, z, q, shared['prior'])
            close_numbers(saved, fresh)
            # Reload the saved parameters and predict separately from re-fitting.
            for task, model in saved['models'].items():
                expected_rows = [r for r in saved['predictions'] if r['task'] == task]
                mapping = {(r['task'], r['sample_id']): i for i,r in enumerate(scene.rows)}
                ids = [mapping[(r['task'],r['sample_id'])] for r in expected_rows]
                predicted = analytic.predict_model(model, z[ids], q[ids])
                require(np.allclose(predicted, [r['prediction'] for r in expected_rows], rtol=1e-8, atol=1e-8), 'saved model replay')
            results[case['id']] = saved
            replayed_fits += len(scene.targets)
        selected = select_cases(scene, results)
        close_numbers(read(folder/'selection.json'), selected)
        folds = [dict(task=r['task'], sample_id=r['sample_id'], canonical=r['canonical'], group=r['group'],
                      fold=data.fold_id(r['group'])) for r in scene.rows if r['task'] in scene.targets and r['split']=='train']
        close_numbers(read(folder/'folds.json'), folds)
        expected_summary = dict(selection=selected, endpoint_fits=len(results)*len(scene.targets),
                                prior_task_fits=len(scene.sources), all_source_tasks_fitted=True,
                                all_target_tasks_fitted=True, historical_reference=historical_reference(reference_root,scene))
        if name in SPEC['relation_banks']:
            rfolder = output/'relations'/name
            close_numbers(read(rfolder/'qualification.json'), relation.qualification(scene))
            summaries = []
            with (rfolder/'cases.jsonl').open(encoding='utf-8') as stream:
                for fresh in relation.cases(scene):
                    line = stream.readline()
                    require(bool(line), 'missing relation case')
                    saved = json.loads(line)
                    close_numbers(saved, fresh)
                    summaries.append({k: fresh[k] for k in ('source','target','fold','scores','train_unique_groups','holdout_unique_groups','holdout_pair_count')})
                    relation_count += 1
                require(stream.read() == '', 'extra relation cases')
            close_numbers(read(rfolder/'summary.json'), relation_summary(summaries))
            qualified = relation.qualification(scene)
            require(len(summaries) == qualified['directed_fold_cases'], 'relation matrix count')
            expected_summary['relations'] = dict(case_count=len(summaries), eligible_task_pairs=qualified['eligible_task_pairs'])
        close_numbers(receipt['summary'][name], expected_summary)
    require(set(receipt['summary']) == set(data.SCENES) and
            receipt['environment']['device'] == 'cpu' and receipt['environment']['numerical_threads'] == 1
            and type(receipt['elapsed_seconds']) is float and receipt['elapsed_seconds'] >= 0, 'run population/environment')
    require(replayed_fits == SPEC['target_endpoint_model_fits'], 'replayed fit matrix')
    return dict(task=TASK, identity=audit['identity'], content_status='PASS_INDEPENDENT_CPU_REFIT_AND_REPLAY',
                acceptance_status='PENDING_CODEX_REVIEW', target_endpoint_fits=replayed_fits,
                relation_cases=relation_count, no_new_test=True, source_teacher_oof=False,
                original_run_checksums_sha256=sha(output/'checksums.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('preflight', 'run', 'verify'):
        cmd = sub.add_parser(action)
        cmd.add_argument('--reference-root', type=Path, required=True)
        cmd.add_argument('--expected-commit', required=True)
        cmd.add_argument('--output', type=Path, required=True)
        if action == 'verify':
            cmd.add_argument('--receipt', type=Path, required=True)
    sub.add_parser('matrix')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.action == 'matrix':
            print(json.dumps(SPEC, ensure_ascii=False, indent=2)); return 0
        if args.action == 'preflight':
            write(args.output, preflight(args.reference_root, args.expected_commit))
        elif args.action == 'run':
            run(args.reference_root, args.output, args.expected_commit)
        else:
            require(not args.receipt.resolve().is_relative_to(args.output.resolve()), 'verification receipt must be outside immutable run')
            require(not args.receipt.exists(), 'existing verification receipt')
            write(args.receipt, verify(args.reference_root, args.output, args.expected_commit))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
