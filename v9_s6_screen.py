"""V9-S6: ten calibrated transfer methods, frozen policies, full-training refits."""
from copy import deepcopy
from pathlib import Path
import argparse
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch
import v9_s3_screen as s3
import v9_s5_screen as s5
import v9_adaptive_transfer as method

REPO = Path(__file__).resolve().parent
TASK = 'V9_S6_ADAPTIVE_TRANSFER_SCREEN_20261002'
REGISTRY = '.tmp/v9s6_attempt_20261002'
SPEC = dict(method.SPEC, prediction_replay_rtol=.001, prediction_replay_atol=.00001)
TESTS = ('tests/test_v9_adaptive_transfer.py', 'tests/test_v9_s6_screen.py', *s5.TESTS)
TEST_COUNT = 311
read, write, sha, digest, require = s3.read, s3.write, s3.sha, s3.digest, s3.require
SMOKE = (('ToxAcute', .0001), ('A', .001), ('B', .001))


def jobs():
    labels = {.0001: '1e4', .001: '1e3'}
    return [dict(id=f'{s}_lr{labels[lr]}_s42', setting=s, lr=lr, seed=42)
            for s in s3.s2.c0.SETTINGS for lr in SPEC['lrs']]


def scalers_for(trainer, setting):
    return trainer.task_scalers if setting == 'ToxAcute' else trainer.scalers


def evaluate(factory, trainer, setting, expected, indices, scalers, device, split='train'):
    model = trainer.model; model.eval(); lookup = {(r['task'], r['sample_id']): r for r in expected}; rows = []
    with torch.no_grad():
        for task, ids in indices.items():
            size = s3.s2.c0.BATCH_SIZES[setting]
            for start in range(0, len(ids), size):
                batch = s3.s2.batch_for(factory, trainer, setting, split, task, ids[start:start+size], device)
                raw = model(batch, task_name=task)[task]
                require(raw.shape == (len(batch.sample_id), 3) and torch.isfinite(raw).all().item(), 'quantile prediction')
                predictions = raw[:, 0].double()*scalers[task]['std']+scalers[task]['mean']
                for sid, canonical, label, prediction in zip(batch.sample_id, batch.canonical_smiles, batch.y.reshape(-1), predictions):
                    meta = lookup[(task, str(sid))]
                    require(canonical == meta['canonical'] and float(label) == float(torch.tensor(meta['label'], dtype=batch.y.dtype)),
                            'prediction graph chemistry/label')
                    rows.append(dict(meta, prediction=float(prediction)))
    s3.s2.metrics(rows, expected)
    return rows


def select_rows(rows, split):
    result = []
    for task, ids in split.items():
        values = [r for r in rows if r['task'] == task]
        result.extend(values[i] for i in ids)
    return result


def engines(base, source, lr, device):
    return {name: method.build(base, source, name, lr, device) for name in method.ENGINES}


def metric(rows):
    return s3.s2.metrics(rows, [{k: v for k, v in r.items() if k != 'prediction'} for r in rows])


def merged_options(name, states, initial, parameter_names, evaluator):
    a, b = states['TARGET_FULL'], states['JOINT_FULL']
    if name == 'TIES_P':
        b = method.ties_merge(initial, a, b, parameter_names)
    options = []
    for alpha in method.GRID:
        state = method.interpolate(a, b, alpha, depth=name == 'LINES_P')
        rows = evaluator(state)
        options.append(dict(alpha=alpha, rows=rows, score=metric(rows)))
    chosen = min(options, key=lambda r: (r['score']['macro_rmse'], r['alpha']))
    return chosen, options


def snapshots_and_choices(bank, initial, parameter_names, evaluator):
    states = {name: method.state(model) for name, (model, _) in bank.items()}
    rows = {name: evaluator(value) for name, value in states.items()}
    result = {name: dict(rows=rows[name], recipe=dict(engine=name)) for name in (*method.CONTROLS, *method.GRADIENT_METHODS)}
    for name in ('WISE_FT_P', 'LINES_P', 'TIES_P'):
        chosen, options = merged_options(name, states, initial, parameter_names, evaluator)
        result[name] = dict(rows=chosen['rows'], recipe=dict(merge=name, alpha=chosen['alpha']), options=options)
    chosen_target = min(method.CONTROLS[:3], key=lambda name: (metric(rows[name])['macro_rmse'], method.CONTROLS.index(name)))
    result['TARGET_ADAPTIVE'] = dict(rows=rows[chosen_target], recipe=dict(engine=chosen_target))
    coefficients = method.gate_coefficients(rows[chosen_target], rows['JOINT_FULL'])
    result['PROTECTED_GATE'] = dict(rows=method.blend_rows(rows[chosen_target], rows['JOINT_FULL'], coefficients),
                                   recipe=dict(engine=chosen_target, second='JOINT_FULL', coefficients=coefficients))
    # Fork/merge is separate from the independent target-only bank.
    options = []
    for alpha in method.GRID:
        r = evaluator(method.interpolate(states['FORK_TARGET'], states['FORK_JOINT'], alpha))
        options.append(dict(alpha=alpha, rows=r, score=metric(r)))
    chosen = min(options, key=lambda r: (r['score']['macro_rmse'], r['alpha']))
    merged = method.interpolate(states['FORK_TARGET'], states['FORK_JOINT'], chosen['alpha'])
    for name in ('FORK_TARGET', 'FORK_JOINT'):
        bank[name][0].load_state_dict(merged, strict=True)
    result['FORKMERGE_P'] = dict(rows=chosen['rows'], recipe=dict(engine='FORK_TARGET'), options=options,
                                fork_alpha=chosen['alpha'])
    return result


def infer_recipe(recipe, states, initial, parameter_names):
    if 'merge' in recipe:
        b = states['JOINT_FULL']
        if recipe['merge'] == 'TIES_P':
            b = method.ties_merge(initial, states['TARGET_FULL'], b, parameter_names)
        return dict(main=method.interpolate(states['TARGET_FULL'], b, recipe['alpha'], depth=recipe['merge'] == 'LINES_P'))
    result = dict(main=states[recipe['engine']])
    if 'second' in recipe:
        result['second'] = states[recipe['second']]
    return result


def train_case(factory, trainer, source, job, out, commit, device, reference, spec=SPEC, smoke=False):
    require(job in jobs(), 'S6 fixed case')
    start = time.monotonic(); setting = job['setting']; tasks, counts = s3.s2.c0.tasks_and_counts(setting)
    require({t: len(d) for t, d in s3.s2.datasets_for(factory, trainer, setting)['train'].items()} == counts, 'all target tasks/counts')
    training = s3.s2.observations(factory, trainer, setting, 'train')
    validation = s3.s2.observations(factory, trainer, setting, 'validation')
    population = s3.population_check(factory, source, setting, training, validation)
    identity = s3.identity_for(trainer, source, job, training, validation, commit, spec, task_id=TASK)
    s5.pairing(identity, reference)
    split = method.partition(training, spec['inner_fraction']); meta = select_rows(training, split['meta'])
    fit_scalers = method.inner_scalers(training, split); full_scalers = deepcopy(scalers_for(trainer, setting))
    base = deepcopy(trainer.model).cpu(); s3.seed_everything(42, deterministic_algorithms=True)
    bank = engines(base, source, job['lr'], device)
    evaluator_model = method.build(base, source, 'TARGET_FULL', job['lr'], device)[0]
    initial = method.state(bank['TARGET_FULL'][0]); parameter_names = {n: True for n, _ in bank['TARGET_FULL'][0].named_parameters()}
    parameter_names['target_tasks'] = tasks
    for name, value in dict(identity=identity, train_observations=training, validation_observations=validation,
                            source_train_observations=source.rows, inner_split=split, inner_scalers=fit_scalers,
                            full_scalers=full_scalers, population_check=population).items():
        write(out/(name+'.json'), value)
    def inner_evaluator(state):
        evaluator_model.load_state_dict(state, strict=True); trainer.model = evaluator_model
        return evaluate(factory, trainer, setting, meta, split['meta'], fit_scalers, device)
    policies = []; best = {}; controller = {name: {} for name in method.GRADIENT_METHODS}
    batch_size = s3.s2.c0.BATCH_SIZES[setting]
    schedule = s3.aux.Schedule(source.tasks, source.counts, batch_size)
    epochs = spec['max_epochs']; total_steps = 0
    for epoch in range(epochs):
        full_plan = s3.s2.epoch_plan(tasks, counts, batch_size, epoch)
        plan = method.fit_plan(full_plan, split)
        if smoke:
            full_plan, plan = full_plan[:2], plan[:2]
        auxiliary = [schedule.next() for _ in plan]; epoch_policy = []; logs = []
        for item, ai in zip(plan, auxiliary):
            total_steps += 1
            batch = s3.s2.batch_for(factory, trainer, setting, 'train', item['task'], item['indices'], device)
            allowed = method.source_mask(source, ai, training, split, setting)
            meta_batches = [(t, s3.s2.batch_for(factory, trainer, setting, 'train', t,
                             [split['meta'][t][(total_steps-1)%len(split['meta'][t])]], device), fit_scalers[t]) for t in tasks]
            weights = {}
            for name in method.GRADIENT_METHODS:
                weights[name] = (method.learn_weights(bank[name][0], name, batch, item['task'], fit_scalers[item['task']],
                    source.batch(ai['task'], ai['indices'], device), ai['task'], source.scalers[ai['task']],
                    meta_batches, allowed, controller[name], job['lr']) if any(allowed) else [0.]*len(allowed))
            uniform = [1./sum(allowed) if ok else 0. for ok in allowed] if any(allowed) else [0.]*len(allowed)
            for name in method.ENGINES:
                if name not in weights:
                    weights[name] = uniform if name in ('JOINT_FULL', 'FORK_JOINT') else [0.]*len(allowed)
            policy = dict(target=item, full_target=full_plan[len(epoch_policy)], auxiliary=ai, allowed=allowed, weights=weights)
            epoch_policy.append(policy)
            log = {}
            for name, (model, opt) in bank.items():
                log[name] = method.update(model, opt, batch, item['task'], fit_scalers[item['task']], source, ai,
                                          weights[name], device, total_steps)
            logs.append(log)
        choices = snapshots_and_choices(bank, initial, parameter_names, inner_evaluator)
        epoch_record = dict(epoch=epoch+1, full_plan_sha256=digest(full_plan), policy=epoch_policy, losses=logs,
                            choices=choices, optimizer_steps={n: s3.optimizer_steps(m, o) for n, (m, o) in bank.items()})
        write(out/f'inner_epoch_{epoch+1:03d}.json', epoch_record)
        policies.append(dict(steps=epoch_policy, fork_alpha=choices['FORKMERGE_P']['fork_alpha']))
        for name, value in choices.items():
            score = metric(value['rows'])['macro_rmse']
            if name not in best or score < best[name]['inner_macro_rmse']:
                best[name] = dict(epoch=epoch+1, inner_macro_rmse=score, recipe=value['recipe'])
        print(f'{job["id"]} inner epoch={epoch+1}/{epochs}', flush=True)
    # Frozen before any outer prediction/evaluation. Full refit never calls a controller.
    selection = dict(task=TASK, job=job, identity=identity, inner_split_sha256=digest(split),
                     policies_sha256=digest(policies), selected=best, outer_labels_used=False)
    write(out/'frozen_selection.json', selection); write(out/'frozen_policies.json', policies)
    del bank
    s3.seed_everything(42, deterministic_algorithms=True); bank = engines(base, source, job['lr'], device)
    trainer.model = evaluator_model; evaluator_model.load_state_dict(initial, strict=True)
    outer_indices = {t: list(range(len(d))) for t, d in s3.s2.datasets_for(factory, trainer, setting)['validation'].items()}
    initial_rows = evaluate(factory, trainer, setting, validation, outer_indices, full_scalers, device, split='validation')
    s3.paired_initial(initial_rows, reference['initial']); write(out/'initial_validation.json', initial_rows)
    results = {}; refit_steps = 0
    for epoch, policy in enumerate(policies):
        logs = []
        for p in policy['steps']:
            refit_steps += 1; item = p['full_target']; ai = p['auxiliary']
            batch = s3.s2.batch_for(factory, trainer, setting, 'train', item['task'], item['indices'], device)
            log = {}
            for name, (model, opt) in bank.items():
                log[name] = method.update(model, opt, batch, item['task'], full_scalers[item['task']], source, ai,
                                          p['weights'][name], device, refit_steps)
            logs.append(log)
        states = {name: method.state(model) for name, (model, _) in bank.items()}
        merged = method.interpolate(states['FORK_TARGET'], states['FORK_JOINT'], policy['fork_alpha'])
        for name in ('FORK_TARGET', 'FORK_JOINT'):
            bank[name][0].load_state_dict(merged, strict=True); states[name] = merged
        write(out/f'refit_epoch_{epoch+1:03d}.json', dict(epoch=epoch+1, losses=logs, policy_sha256=digest(policy),
              optimizer_steps={n: s3.optimizer_steps(m, o) for n, (m, o) in bank.items()}, full_target_steps=refit_steps))
        for name, chosen in best.items():
            if chosen['epoch'] != epoch+1:
                continue
            model_states = infer_recipe(chosen['recipe'], states, initial, parameter_names)
            control_engine = chosen['recipe'].get('engine', 'TARGET_FULL') if name == 'PROTECTED_GATE' else 'TARGET_FULL'
            payload = dict(identity=identity, selection_sha256=sha(out/'frozen_selection.json'), method=name,
                           chosen=chosen, models=model_states, scalers=full_scalers, epoch=epoch+1,
                           paired_control=states[control_engine], control_engine=control_engine)
            with (out/(name+'.pt')).open('xb') as stream:
                torch.save(payload, stream)
            predictions = []
            for value in model_states.values():
                evaluator_model.load_state_dict(value, strict=True); trainer.model = evaluator_model
                predictions.append(evaluate(factory, trainer, setting, validation, outer_indices, full_scalers, device, split='validation'))
            rows = (predictions[0] if len(predictions) == 1 else method.blend_rows(*predictions, chosen['recipe']['coefficients']))
            write(out/(name+'_validation.json'), rows)
            evaluator_model.load_state_dict(payload['paired_control'], strict=True); trainer.model = evaluator_model
            paired = evaluate(factory, trainer, setting, validation, outer_indices, full_scalers, device, split='validation')
            write(out/(name+'_paired_control.json'), paired)
            results[name] = dict(selected=metric(rows), chosen=chosen, checkpoint_sha256=sha(out/(name+'.pt')),
                                 prediction_sha256=sha(out/(name+'_validation.json')), paired_control=metric(paired),
                                 control_engine=control_engine, control_prediction_sha256=sha(out/(name+'_paired_control.json')))
        print(f'{job["id"]} refit epoch={epoch+1}/{epochs}', flush=True)
    require(set(results) == set(method.METHODS), 'complete ten candidates and controls')
    all_aux = [p['auxiliary'] for policy in policies for p in policy['steps']]
    coverage = dict(visited=s3.aux.coverage(source, all_aux), accepted={})
    for name in method.ENGINES:
        selected = [dict(p['auxiliary'], indices=[i for i, w in zip(p['auxiliary']['indices'], p['weights'][name]) if w > 0])
                    for policy in policies for p in policy['steps'] if any(p['weights'][name])]
        coverage['accepted'][name] = s3.aux.coverage(source, selected)
    if not smoke:
        require(all(v['updates'] > 0 for v in coverage['visited'].values()), 'all source task pool visited')
    write(out/'coverage.json', coverage)
    result = dict(task=TASK, job=job, commit=commit, identity=identity, results=results,
                  target_updates_per_engine=refit_steps, inner_updates_per_engine=total_steps,
                  model_training_runs=2*len(method.ENGINES), model_epochs=2*len(method.ENGINES)*epochs,
                  selection_sha256=sha(out/'frozen_selection.json'), policy_sha256=sha(out/'frozen_policies.json'),
                  test_evaluated=False, scientific_acceptance='PENDING_REVIEW', smoke=smoke,
                  environment=dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda, device=str(device),
                                   gpu_name=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None),
                  parameters={n: dict(total=sum(p.numel() for p in m.parameters()),
                                      requires_grad=sum(p.numel() for p in m.parameters() if p.requires_grad),
                                      updated=sum(p.numel() for p in m.parameters() if p in o.state)) for n, (m, o) in bank.items()},
                  elapsed_seconds=time.monotonic()-start)
    write(out/'receipt.json', result); return result


def choose_lrs(inner_selections):
    """This API deliberately takes no outer predictions or labels."""
    require(len(inner_selections) == 2 and {r['job']['lr'] for r in inner_selections} == set(SPEC['lrs']), 'two LR selections')
    require(len({r['job']['setting'] for r in inner_selections}) == 1, 'selection scene')
    return {name: min(inner_selections, key=lambda r: (r['selected'][name]['inner_macro_rmse'], r['job']['lr']))['job']['id']
            for name in method.METHODS}


def check_losses(log, weights):
    require(set(log) == set(method.ENGINES), 'loss engines')
    for name, value in log.items():
        require(set(value) == {'target', 'auxiliary', 'norms', 'accepted', 'weight_sum'} and
                set(value['norms']) == {'encoder', 'target', 'source'}, 'loss schema')
        require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                    for x in [value['target'], value['auxiliary'], *value['norms'].values()]), 'finite loss/norm')
        require(type(value['accepted']) is int and value['accepted'] == sum(w > 0 for w in weights[name])
                and math.isclose(value['weight_sum'], math.fsum(weights[name]), rel_tol=1e-10, abs_tol=1e-12), 'accepted source count')
        if not any(weights[name]):
            require(value['auxiliary'] == value['norms']['source'] == 0., 'rejected auxiliary updated')


def compare_replay(replayed, recorded, spec):
    require(len(replayed) == len(recorded), 'CPU prediction population')
    for x, y in zip(replayed, recorded):
        require(set(x) == set(y) and all(x[k] == v for k, v in y.items() if k != 'prediction')
                and math.isclose(x['prediction'], y['prediction'], rel_tol=spec['prediction_replay_rtol'],
                                 abs_tol=spec['prediction_replay_atol']), 'CPU raw-data prediction replay')


def verify_case(factory, trainer, source, job, out, commit, reference, spec=SPEC, smoke=False):
    r = read(out/'receipt.json'); setting = job['setting']; tasks, counts = s3.s2.c0.tasks_and_counts(setting)
    training = s3.s2.observations(factory, trainer, setting, 'train')
    validation = s3.s2.observations(factory, trainer, setting, 'validation')
    identity = s3.identity_for(trainer, source, job, training, validation, commit, spec, task_id=TASK)
    s5.pairing(identity, reference)
    require(r['identity'] == read(out/'identity.json') == identity and r['job'] == job and r['commit'] == commit
            and r['task'] == TASK and r['smoke'] is smoke, 'S6 receipt identity')
    split = method.partition(training, spec['inner_fraction']); meta = select_rows(training, split['meta'])
    expected = dict(train_observations=training, validation_observations=validation, source_train_observations=source.rows,
                    inner_split=split, inner_scalers=method.inner_scalers(training, split),
                    full_scalers=scalers_for(trainer, setting), population_check=s3.population_check(factory, source, setting, training, validation))
    for name, value in expected.items():
        require(read(out/(name+'.json')) == value, 'trusted '+name)
    require(r['selection_sha256'] == sha(out/'frozen_selection.json') and r['policy_sha256'] == sha(out/'frozen_policies.json'), 'selection/policy bytes')
    selection = read(out/'frozen_selection.json'); policies = read(out/'frozen_policies.json')
    require(len(policies) == spec['max_epochs'], 'full calibration epochs')
    bank = engines(trainer.model, source, job['lr'], 'cpu'); initial = method.state(bank['TARGET_FULL'][0])
    schedule = s3.aux.Schedule(source.tasks, source.counts, s3.s2.c0.BATCH_SIZES[setting])
    best = {}; target_plan = []; auxiliary_plans = {n: [] for n in method.ENGINES}
    for epoch, policy in enumerate(policies):
        h = read(out/f'inner_epoch_{epoch+1:03d}.json'); f = read(out/f'refit_epoch_{epoch+1:03d}.json')
        full = s3.s2.epoch_plan(tasks, counts, s3.s2.c0.BATCH_SIZES[setting], epoch)
        fit = method.fit_plan(full, split)
        if smoke:
            full, fit = full[:2], fit[:2]
        require(h['epoch'] == f['epoch'] == epoch+1 and h['full_plan_sha256'] == digest(full)
                and h['policy'] == policy['steps'] and len(h['losses']) == len(f['losses']) == len(full)
                and f['policy_sha256'] == digest(policy), 'epoch schedule/policy binding')
        require(len(policy['steps']) == len(full), 'policy step count')
        for i, p in enumerate(policy['steps']):
            ai = schedule.next(); target_plan.append(full[i])
            allowed = method.source_mask(source, ai, training, split, setting)
            require(set(p) == {'target', 'full_target', 'auxiliary', 'allowed', 'weights'}
                    and p['target'] == fit[i] and p['full_target'] == full[i] and p['auxiliary'] == ai
                    and p['allowed'] == allowed and set(p['weights']) == set(method.ENGINES), 'trusted inner/full/source schedule')
            for name, weights in p['weights'].items():
                require(type(weights) is list and len(weights) == len(allowed)
                        and all(type(w) in (int, float) and math.isfinite(w) and 0 <= w <= 1 for w in weights)
                        and math.fsum(weights) <= 1.000001 and all(ok or w == 0 for ok, w in zip(allowed, weights)), 'valid masked policy')
                if name in ('TARGET_FULL', 'TARGET_LAST', 'TARGET_FROZEN', 'FORK_TARGET'):
                    require(not any(weights), 'protected target received auxiliary weight')
                if name in ('JOINT_FULL', 'FORK_JOINT'):
                    uniform = [1./sum(allowed) if ok else 0. for ok in allowed] if any(allowed) else [0.]*len(allowed)
                    require(weights == uniform, 'uniform control')
                if any(weights): auxiliary_plans[name].append(ai)
            check_losses(h['losses'][i], p['weights']); check_losses(f['losses'][i], p['weights'])
        require(f['full_target_steps'] == len(target_plan), 'full target update count')
        require(set(h['optimizer_steps']) == set(f['optimizer_steps']) == set(method.ENGINES), 'Adam engine scope')
        for name, (model, opt) in bank.items():
            steps = {n: count for n, count, p in s5.expected_steps(model, opt, target_plan, auxiliary_plans[name]).values()}
            require(h['optimizer_steps'][name] == f['optimizer_steps'][name] == steps, 'actual Adam update coverage')
        choices = h['choices']; require(set(choices) == set(method.METHODS), 'all inner candidates')
        for name, value in choices.items():
            score = s3.s2.metrics(value['rows'], meta)['macro_rmse']
            if 'options' in value:
                require([o['alpha'] for o in value['options']] == list(method.GRID), 'frozen merge grid')
                for option in value['options']:
                    require(option['score'] == s3.s2.metrics(option['rows'], meta), 'independent grid metric')
                chosen = min(value['options'], key=lambda v: (v['score']['macro_rmse'], v['alpha']))
                require(value['rows'] == chosen['rows'], 'inner merge selection')
                if name == 'FORKMERGE_P':
                    require(policy['fork_alpha'] == value['fork_alpha'] == chosen['alpha']
                            and value['recipe'] == dict(engine='FORK_TARGET'), 'fork merge schedule')
                else:
                    require(value['recipe'] == dict(merge=name, alpha=chosen['alpha']), 'merge recipe')
            elif name in (*method.CONTROLS, *method.GRADIENT_METHODS):
                require(value['recipe'] == dict(engine=name), 'engine recipe')
            if name not in best or score < best[name]['inner_macro_rmse']:
                best[name] = dict(epoch=epoch+1, inner_macro_rmse=score, recipe=value['recipe'])
        target = min(method.CONTROLS[:3], key=lambda n: (metric(choices[n]['rows'])['macro_rmse'], method.CONTROLS.index(n)))
        require(choices['TARGET_ADAPTIVE'] == dict(rows=choices[target]['rows'], recipe=dict(engine=target)), 'adaptive plasticity rule')
        coefficients = method.gate_coefficients(choices[target]['rows'], choices['JOINT_FULL']['rows'])
        require(choices['PROTECTED_GATE'] == dict(rows=method.blend_rows(choices[target]['rows'], choices['JOINT_FULL']['rows'], coefficients),
                recipe=dict(engine=target, second='JOINT_FULL', coefficients=coefficients)), 'gate train-only rule')
    require(selection == dict(task=TASK, job=job, identity=identity, inner_split_sha256=digest(split),
                             policies_sha256=digest(policies), selected=best, outer_labels_used=False), 'independent frozen selection')
    all_aux = [p['auxiliary'] for policy in policies for p in policy['steps']]
    coverage = dict(visited=s3.aux.coverage(source, all_aux), accepted={})
    for name in method.ENGINES:
        selected = [dict(p['auxiliary'], indices=[i for i, w in zip(p['auxiliary']['indices'], p['weights'][name]) if w > 0])
                    for policy in policies for p in policy['steps'] if any(p['weights'][name])]
        coverage['accepted'][name] = s3.aux.coverage(source, selected)
    require(read(out/'coverage.json') == coverage and (smoke or all(v['updates'] > 0 for v in coverage['visited'].values())), 'task eligibility/acceptance coverage')
    require(r['target_updates_per_engine'] == r['inner_updates_per_engine'] == len(target_plan)
            and r['model_training_runs'] == 2*len(method.ENGINES)
            and r['model_epochs'] == 2*len(method.ENGINES)*spec['max_epochs'], 'training matrix counts')
    parameters = {}
    for name, (model, opt) in bank.items():
        updated = s5.expected_steps(model, opt, target_plan, auxiliary_plans[name])
        parameters[name] = dict(total=sum(p.numel() for p in model.parameters()),
                               requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
                               updated=sum(p.numel() for _, _, p in updated.values()))
    require(r['parameters'] == parameters, 'configured/updated parameter counts')
    s3.paired_initial(read(out/'initial_validation.json'), reference['initial'])
    model = bank['TARGET_FULL'][0]; trainer.model = model
    indices = {t: list(range(len(d))) for t, d in s3.s2.datasets_for(factory, trainer, setting)['validation'].items()}
    require(set(r['results']) == set(method.METHODS), 'complete outer results')
    for name, result in r['results'].items():
        path = out/(name+'.pt'); require(result['checkpoint_sha256'] == sha(path), 'checkpoint bytes')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(set(payload) == {'identity', 'selection_sha256', 'method', 'chosen', 'models', 'scalers', 'epoch', 'paired_control', 'control_engine'}
                and payload['identity'] == identity and payload['method'] == name and payload['chosen'] == result['chosen'] == best[name]
                and payload['epoch'] == best[name]['epoch'] and payload['selection_sha256'] == r['selection_sha256']
                and payload['scalers'] == expected['full_scalers'], 'checkpoint identity/selection')
        recipe = best[name]['recipe']; control = recipe['engine'] if name == 'PROTECTED_GATE' else 'TARGET_FULL'
        require(payload['control_engine'] == result['control_engine'] == control and
                set(payload['models']) == ({'main', 'second'} if name == 'PROTECTED_GATE' else {'main'}), 'inference branches')
        predictions = []
        for key, state in {**payload['models'], 'control': payload['paired_control']}.items():
            require(set(state) == set(initial), 'checkpoint tensor keys')
            for k, v in state.items():
                require(isinstance(v, torch.Tensor) and v.dtype == initial[k].dtype and v.shape == initial[k].shape
                        and torch.isfinite(v).all().item(), 'checkpoint tensor '+k)
                if k in dict(model.named_buffers()):
                    require(torch.equal(v, initial[k]), 'immutable buffer')
            model.load_state_dict(state, strict=True)
            predictions.append(evaluate(factory, trainer, setting, validation, indices, expected['full_scalers'], 'cpu', split='validation'))
        paired = predictions.pop()
        replay = predictions[0] if len(predictions) == 1 else method.blend_rows(*predictions, recipe['coefficients'])
        for suffix, computed, metric_key, hash_key in (('_validation.json', replay, 'selected', 'prediction_sha256'),
                                                       ('_paired_control.json', paired, 'paired_control', 'control_prediction_sha256')):
            rows = read(out/(name+suffix)); require(sha(out/(name+suffix)) == result[hash_key], 'prediction bytes')
            require(result[metric_key] == s3.s2.metrics(rows, validation), 'independent outer metric')
            compare_replay(computed, rows, spec)
        if name == 'PROTECTED_GATE':
            require(all(torch.equal(payload['models']['main'][k], payload['paired_control'][k]) for k in initial), 'gate independent target anchor')
    require(r['test_evaluated'] is False and r['scientific_acceptance'] == 'PENDING_REVIEW', 'scientific scope')
    return r


def compare(root, receipts, references):
    require({r['job']['id'] for r in receipts} == {j['id'] for j in jobs()} and len(receipts) == 6, 'six fixed cases')
    by_id = {r['job']['id']: r for r in receipts}; records = []; lr_choices = {}
    for setting in s3.s2.c0.SETTINGS:
        selections = [read(root/j['id']/'frozen_selection.json') for j in jobs() if j['setting'] == setting]
        chosen = choose_lrs(selections); lr_choices[setting] = chosen
        old = {n: references[f'{setting}_{n}_s42']['selected'] for n in s3.s2.METHODS}
        strong_name = min(old, key=lambda n: old[n]['macro_rmse']); strong = old[strong_name]
        control = by_id[chosen['TARGET_ADAPTIVE']]['results']['TARGET_ADAPTIVE']['selected']
        for name in method.METHODS:
            case = by_id[chosen[name]]; value = case['results'][name]; score = value['selected']
            records.append(dict(setting=setting, method=name, job=case['job'], chosen=value['chosen'], selected=score,
                 best_077_method=strong_name, best_077=strong, delta_vs_best_077=score['macro_rmse']-strong['macro_rmse'],
                 relative_vs_best_077=score['macro_rmse']/strong['macro_rmse']-1,
                 delta_vs_adaptive_target=score['macro_rmse']-control['macro_rmse'],
                 delta_vs_exact_paired_control=score['macro_rmse']-value['paired_control']['macro_rmse'],
                 paired_control=value['paired_control'],
                 endpoint_deltas_vs_best_077={t: v['rmse']-strong['endpoints'][t]['rmse'] for t, v in score['endpoints'].items()}))
    ranking = []
    for name in method.CANDIDATES:
        rows = [r for r in records if r['method'] == name]
        ranking.append(dict(method=name, worst_scene_relative=max(r['relative_vs_best_077'] for r in rows),
            mean_scene_relative=math.fsum(r['relative_vs_best_077'] for r in rows)/3,
            all_scene_point_signal=all(r['delta_vs_best_077'] < 0 and r['delta_vs_adaptive_target'] < 0
                                      and r['delta_vs_exact_paired_control'] < 0 for r in rows)))
    ranking.sort(key=lambda r: (r['worst_scene_relative'], r['mean_scene_relative'], r['method']))
    return dict(lr_choices=lr_choices, comparisons=records, candidate_ranking=ranking,
                all_scene_candidate_signals=[r['method'] for r in ranking if r['all_scene_point_signal']],
                unified_superiority_confirmed=False, scope='SINGLE_SEED_VALIDATION_SCREEN_NOT_FINAL_CLAIM')


def gate_check(root, commit, role):
    require(read(root/'gate.json') == dict(task=TASK, commit=commit, role=role, tests=list(TESTS), junit_sha256=sha(root/'tests.xml')), 'S6 gate identity')
    require(read(root/'command.json')['exit_code'] == 0, 'S6 gate exit')
    cases = ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases) == TEST_COUNT and all(not any(c.findall(t) for t in ('failure', 'error', 'skipped')) for c in cases)
            and {c.attrib['classname'] for c in cases} == {'tests.'+Path(t).stem for t in TESTS}, 'S6 complete gate')


def code_gate(root, commit, role):
    require(role in ('wsl', 'server'), 'S6 gate role'); s3.s2.c0.check_code(REPO, commit)
    root.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-m', 'pytest', *TESTS, '-q', '--basetemp', str(root/'pytest_tmp'), '--junitxml', str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as log:
        p = subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json', dict(argv=argv, exit_code=p.returncode)); require(p.returncode == 0, 'S6 tests failed')
    write(root/'gate.json', dict(task=TASK, commit=commit, role=role, tests=list(TESTS), junit_sha256=sha(root/'tests.xml')))
    gate_check(root, commit, role)


def sequence():
    return [('smoke', 'smoke_'+s, 'smoke_'+s) for s, _ in SMOKE]+[('train', j['id'], j['id']) for j in jobs()]


def claim_value(root, commit):
    return dict(task=TASK, commit=commit, output=str(root.resolve()), spec=SPEC, jobs=jobs(), smoke=[list(x) for x in SMOKE])


def check_launch(root, commit):
    claim = claim_value(root, commit); require(read(REPO/REGISTRY/'attempt.json') == claim, 'S6 one-shot claim')
    launch = read(root/'launch.json')
    require(launch['claim'] == claim and launch['task'] == TASK and launch['commit'] == commit, 'S6 launch')
    return launch


def worker_command(mode, job, root, commit, split, source_lock):
    return [sys.executable, str(Path(__file__).resolve()), '_worker', '--mode', mode, '--job', job, '--commit', commit,
            '--output', str(root), '--split-manifest', str(split), '--source-lock', str(source_lock)]


def verify(root, commit, split, source_lock):
    check_launch(root, commit)
    for role in ('wsl', 'server'):
        gate_check(root/(role+'_evidence'), commit, role)
    references = s3.references(root/'reference_077'); receipts = []; pids = []
    for job in [*('smoke_'+s for s, _ in SMOKE), *(j['id'] for j in jobs())]:
        require(read(root/(job+'.command.json'))['exit_code'] == 0, 'training worker exit')
        s5.call_worker(worker_command('verify', job, root, commit, split, source_lock), root/('verify_'+job),
                       dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        v = read(root/job/'verification.json'); r = read(root/job/'receipt.json')
        require(v == dict(task=TASK, commit=commit, job=job, content_status='PASS', raw_prediction_replayed=True,
                         new_optimizer_updates=0, device='cpu', pid=v['pid'], receipt_sha256=sha(root/job/'receipt.json'))
                and type(v['pid']) is int, 'independent CPU verification')
        pids.append(v['pid'])
        if not job.startswith('smoke_'): receipts.append(r)
    return dict(task=TASK, commit=commit, content_status='PASS', scientific_acceptance='PENDING_REVIEW',
                model_training_runs=sum(r['model_training_runs'] for r in receipts),
                model_epochs=sum(r['model_epochs'] for r in receipts), verification_pids=pids,
                **compare(root, receipts, references))


def run(a):
    from p1d4_batch import free_gpus
    s3.s2.c0.check_code(REPO, a.commit)
    for role in ('wsl', 'server'): gate_check(getattr(a, role+'_evidence'), a.commit, role)
    s3.references(a.reference_root); root = a.output
    require(not (REPO/REGISTRY/'attempt.json').exists(), 'S6 attempt consumed')
    require(a.gpus and len(set(a.gpus)) == len(a.gpus) and set(a.gpus) <= {0, 1, 2, 3}, 'GPU allowlist')
    available = free_gpus(a.gpus); require(bool(available), 'no idle authorized GPU')
    root.mkdir(parents=True, exist_ok=False); (REPO/REGISTRY).mkdir(parents=True, exist_ok=True)
    claim = claim_value(root, a.commit); write(REPO/REGISTRY/'attempt.json', claim)
    uuids = {str(g): s3.s2.c0.gpu_uuid(g) for g in available}
    write(root/'launch.json', dict(task=TASK, commit=a.commit, claim=claim, gpus=available, gpu_uuids=uuids))
    try:
        for role in ('wsl', 'server'):
            shutil.copytree(getattr(a, role+'_evidence'), root/(role+'_evidence'), ignore=shutil.ignore_patterns('pytest_tmp'))
        s3.copy_references(a.reference_root, root/'reference_077')
        def execute(item, gpu):
            mode, key, name = item; require(free_gpus([gpu]) == [gpu], 'GPU occupied')
            write(root/(name+'.assignment.json'), dict(task=TASK, commit=a.commit, mode=mode, job=key, gpu=gpu, gpu_uuid=uuids[str(gpu)]))
            s5.call_worker(worker_command(mode, key, root, a.commit, a.split_manifest, a.source_lock), root/name,
                           dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[str(gpu)], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        s5.dispatch([x for x in sequence() if x[0] == 'smoke'], available, execute)
        s5.dispatch([x for x in sequence() if x[0] == 'train'], available, execute)
        write(root/'verification.json', verify(root, a.commit, a.split_manifest, a.source_lock))
    except BaseException as exc:
        write(root/'failed.json', dict(task=TASK, error_type=type(exc).__name__, reason=str(exc))); raise


def worker(a):
    launch = check_launch(a.output, a.commit); out = a.output/a.job
    smoke_pair = next(((s, lr) for s, lr in SMOKE if a.job == 'smoke_'+s), None)
    job = next(j for j in jobs() if (j['setting'], j['lr']) == smoke_pair) if smoke_pair else next(j for j in jobs() if j['id'] == a.job)
    is_smoke = smoke_pair is not None; spec = dict(SPEC, max_epochs=1) if is_smoke else SPEC
    if a.mode == 'verify':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only verification'); torch.set_num_threads(1)
        require(not (out/'verification.json').exists(), 'verification already exists')
        f, t, source = s3.context(job['setting'], a.split_manifest, a.source_lock, 'cpu')
        ref = s3.references(a.output/'reference_077')[f'{job["setting"]}_B1_1E4_s42']
        verify_case(f, t, source, job, out, a.commit, ref, spec, smoke=is_smoke)
        write(out/'verification.json', dict(task=TASK, commit=a.commit, job=a.job, pid=os.getpid(), device='cpu',
              content_status='PASS', raw_prediction_replayed=True, new_optimizer_updates=0, receipt_sha256=sha(out/'receipt.json')))
        return
    require((a.mode == 'smoke') == is_smoke and a.mode in ('smoke', 'train'), 'worker mode')
    assignment = read(a.output/(a.job+'.assignment.json'))
    require(assignment == dict(task=TASK, commit=a.commit, mode=a.mode, job=a.job, gpu=assignment['gpu'],
            gpu_uuid=launch['gpu_uuids'][str(assignment['gpu'])]) and assignment['gpu'] in launch['gpus'], 'GPU assignment identity')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == assignment['gpu_uuid'] and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'single GPU binding')
    if a.mode == 'train':
        for s, _ in SMOKE:
            r = read(a.output/('smoke_'+s)/'receipt.json')
            require(r['task'] == TASK and r['commit'] == a.commit and r['smoke'] is True
                    and set(r['results']) == set(method.METHODS)
                    and read(a.output/('smoke_'+s+'.command.json'))['exit_code'] == 0, 'real smoke incomplete')
    f, t, source = s3.context(job['setting'], a.split_manifest, a.source_lock, 'cuda:0')
    out.mkdir(exist_ok=False)
    write(out/'started.json', dict(task=TASK, commit=a.commit, mode=a.mode, job=a.job, pid=os.getpid()))
    ref = s3.references(a.output/'reference_077')[f'{job["setting"]}_B1_1E4_s42']
    train_case(f, t, source, job, out, a.commit, 'cuda:0', ref, spec, smoke=is_smoke)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('code-gate', 'run', '_worker', 'verify', 'package'))
    p.add_argument('--commit', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--role', choices=('wsl', 'server')); p.add_argument('--gpus', type=int, nargs='+', choices=range(4))
    p.add_argument('--mode', choices=('smoke', 'train', 'verify')); p.add_argument('--job', choices=[x[1] for x in sequence()])
    for name in ('split-manifest', 'source-lock', 'reference-root', 'wsl-evidence', 'server-evidence'):
        p.add_argument('--'+name, type=Path)
    a = p.parse_args(); a.output = a.output.resolve()
    if a.action == 'code-gate': code_gate(a.output, a.commit, a.role)
    elif a.action == 'package':
        if (a.output/'verification.json').exists():
            v = read(a.output/'verification.json')
            require(v['task'] == TASK and v['commit'] == a.commit and v['content_status'] == 'PASS', 'verified package')
        else:
            require(read(a.output/'failed.json')['task'] == TASK, 'partial failure package requires failure record')
        print(s3.s2.package(a.output, a.commit))
    else:
        s3.s2.c0.check_code(REPO, a.commit)
        require(a.split_manifest is not None and a.source_lock is not None, 'asset paths')
        if a.action == 'run':
            require(all(getattr(a, k) is not None for k in ('gpus', 'reference_root', 'wsl_evidence', 'server_evidence')), 'run arguments'); run(a)
        elif a.action == 'verify': print(verify(a.output, a.commit, a.split_manifest, a.source_lock))
        else: worker(a)


if __name__ == '__main__':
    main()
