"""V9-S6R1: replay the locked S6 refits and compare two selection protocols."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import torch
import v9_s6_screen as s6

s3, s5, method = s6.s3, s6.s5, s6.method
read, write, sha, digest, require = s6.read, s6.write, s6.sha, s6.digest, s6.require
REPO = Path(__file__).resolve().parent
TASK = 'V9_S6R1_SELECTION_PROTOCOL_REPLAY_20261003'
REGISTRY = '.tmp/v9s6r1_attempt_20261003'
SOURCE_COMMIT = '8b8f7bea14f36fb073763db6a0d85405edda3077'
SOURCE_ZIP_SHA256 = '60cff001fe9a42780ebf4bcda5375739a83f1b20f4c788118a4ef5f6786c866a'
SOURCE_MANIFEST_SHA256 = 'd5517dde656213c539eb856f1391fe8495eb15460e6b5f3a01ffd38a3fd023fb'
SOURCE_FILE_COUNT = 1171
TOX_066_THRESHOLD = 1.023615319
SPEC = dict(s6.SPEC, schema='s6r1_locked_refit_replay_v1', source_commit=SOURCE_COMMIT,
            source_manifest_sha256=SOURCE_MANIFEST_SHA256, calibration_updates=0,
            recipe='final_S6_recipe_fixed_at_all_refit_epochs',
            selection='report_frozen_internal_and_development_validation_separately',
            development_tie='lower_LR_then_earlier_epoch_then_target_profile_order',
            original_state_replay='exact_tensors_same_training_backend',
            optimizer_log_replay='exact_original_refit_logs', extra_smoke_updates=0)
TESTS = ('tests/test_v9_s6r1_screen.py', *s6.TESTS)
TEST_COUNT = 352  # 311 existing regression cases plus 41 selection-replay cases.
jobs = s6.jobs


def manifest_bytes(root):
    """The original server directory has no manifest; its sibling ZIP does."""
    path = root/'checksums.sha256'
    if path.is_file():
        require(not path.is_symlink(), 'source manifest symlink')
        raw = path.read_bytes()
    else:
        archive = root.with_suffix('.zip')
        require(archive.is_file() and not archive.is_symlink() and sha(archive) == SOURCE_ZIP_SHA256,
                'locked 083 source archive missing or changed')
        with zipfile.ZipFile(archive) as z:
            require(z.namelist().count('checksums.sha256') == 1, 'source manifest member')
            raw = z.read('checksums.sha256')
    require(hashlib.sha256(raw).hexdigest() == SOURCE_MANIFEST_SHA256, 'trusted 083 manifest hash')
    return raw


def parse_manifest(raw):
    files = {}; folded = set()
    for line in raw.decode('utf-8').splitlines():
        parts = line.split('  ', 1)
        require(len(parts) == 2 and re.fullmatch('[0-9a-f]{64}', parts[0]) is not None, 'source checksum syntax')
        checksum, name = parts; p = PurePosixPath(name)
        require(name and p.as_posix() == name and not p.is_absolute() and '..' not in p.parts
                and ':' not in name and '\\' not in name and name.casefold() not in folded, 'unsafe/duplicate source member')
        folded.add(name.casefold()); files[name] = checksum
    require(len(files) == SOURCE_FILE_COUNT, 'complete 083 manifest')
    return files


def verify_source(root, *, job=None, manifest=None):
    require(root.is_dir() and not root.is_symlink(), '083 source directory')
    raw = manifest_bytes(root) if manifest is None else manifest
    require(hashlib.sha256(raw).hexdigest() == SOURCE_MANIFEST_SHA256, 'trusted 083 manifest hash')
    files = parse_manifest(raw)
    names = list(files) if job is None else [n for n in files if n.startswith(job+'/') or '/' not in n]
    require(names and (job is None or job in {j['id'] for j in jobs()}), 'source case allowlist')
    for name in names:
        path = root/name
        require(path.resolve().is_relative_to(root.resolve()) and not any(p.is_symlink() for p in [path, *path.parents]),
                'source link/path escape')
        require(path.is_file() and sha(path) == files[name], '083 source changed: '+name)
    if job is None:
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        require(actual in (set(files), set(files) | {'checksums.sha256'}), '083 source member set')
        require(not any(p.is_symlink() for p in root.rglob('*')), 'source directory symlink')
    v = read(root/'verification.json')
    require(v['task'] == s6.TASK and v['commit'] == SOURCE_COMMIT and v['content_status'] == 'PASS'
            and v['model_training_runs'] == 132 and v['model_epochs'] == 5280, 'accepted S6 source identity')
    return dict(task=s6.TASK, commit=SOURCE_COMMIT, manifest_sha256=SOURCE_MANIFEST_SHA256,
                source_zip_sha256=SOURCE_ZIP_SHA256, files_checked=len(names), root=str(root.resolve()))


@contextmanager
def evaluation_rng(device):
    """Diagnostic forwards cannot consume training RNG, including third-party loaders."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0] if str(device).startswith('cuda') else []):
            yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def exact_state(actual, expected):
    require(set(actual) == set(expected), 'original state keys')
    require(all(isinstance(v, torch.Tensor) and v.dtype == expected[k].dtype and v.shape == expected[k].shape
                and torch.equal(v, expected[k]) for k, v in actual.items()), 'original S6 tensor replay differs')


def context_info(factory, trainer, source, job, old, commit, spec):
    require(job in jobs(), 'fixed refit job')
    setting = job['setting']; training = s3.s2.observations(factory, trainer, setting, 'train')
    validation = s3.s2.observations(factory, trainer, setting, 'validation')
    old_spec = dict(s6.SPEC, max_epochs=spec['max_epochs'])
    previous_identity = s3.identity_for(trainer, source, job, training, validation, SOURCE_COMMIT, old_spec, task_id=s6.TASK)
    previous = read(old/'receipt.json'); selection = read(old/'frozen_selection.json'); policies = read(old/'frozen_policies.json')
    require(previous['task'] == s6.TASK and previous['commit'] == SOURCE_COMMIT and previous['job'] == job
            and previous['identity'] == read(old/'identity.json') == previous_identity and previous['smoke'] is False,
            'original refit source identity')
    expected = dict(train_observations=training, validation_observations=validation, source_train_observations=source.rows,
                    full_scalers=s6.scalers_for(trainer, setting),
                    population_check=s3.population_check(factory, source, setting, training, validation))
    for name, value in expected.items():
        require(read(old/(name+'.json')) == value, 'trusted original '+name)
    require(selection['task'] == s6.TASK and selection['job'] == job and selection['identity'] == previous_identity
            and selection['outer_labels_used'] is False and set(selection['selected']) == set(method.METHODS)
            and selection['policies_sha256'] == digest(policies) and len(policies) == spec['max_epochs'], 'frozen calibration identity')
    require(previous['selection_sha256'] == sha(old/'frozen_selection.json')
            and previous['policy_sha256'] == sha(old/'frozen_policies.json'), 'frozen calibration bytes')
    for name, chosen in selection['selected'].items():
        require(chosen == previous['results'][name]['chosen'] and type(chosen['epoch']) is int
                and 1 <= chosen['epoch'] <= spec['max_epochs'], 'original chosen recipe/epoch')
        require(chosen['recipe'] == read(old/f'inner_epoch_{chosen["epoch"]:03d}.json')['choices'][name]['recipe'],
                'final inner recipe only')
    identity = s3.identity_for(trainer, source, job, training, validation, commit, spec, task_id=TASK)
    binding = dict(commit=SOURCE_COMMIT, receipt_sha256=sha(old/'receipt.json'),
                   selection_sha256=sha(old/'frozen_selection.json'), policy_sha256=sha(old/'frozen_policies.json'),
                   manifest_sha256=SOURCE_MANIFEST_SHA256)
    return identity, binding, previous, selection, policies, expected


def validate_policy(policy, epoch, tasks, counts, source, training, split, setting, schedule):
    full = s3.s2.epoch_plan(tasks, counts, s3.s2.c0.BATCH_SIZES[setting], epoch)
    fit = method.fit_plan(full, split)
    require(set(policy) == {'steps', 'fork_alpha'} and policy['fork_alpha'] in method.GRID
            and len(policy['steps']) == len(full), 'frozen epoch policy')
    for index, p in enumerate(policy['steps']):
        ai = schedule.next(); allowed = method.source_mask(source, ai, training, split, setting)
        require(set(p) == {'target', 'full_target', 'auxiliary', 'allowed', 'weights'}
                and p['full_target'] == full[index] and p['target'] == fit[index]
                and p['auxiliary'] == ai and p['allowed'] == allowed and set(p['weights']) == set(method.ENGINES),
                'original full/auxiliary schedule')
        for name, weights in p['weights'].items():
            require(type(weights) is list and len(weights) == len(allowed)
                    and all(type(w) in (int, float) and math.isfinite(w) and 0 <= w <= 1 for w in weights)
                    and math.fsum(weights) <= 1.000001 and all(ok or w == 0 for ok, w in zip(allowed, weights)), 'frozen masked weights')
            if name in ('TARGET_FULL', 'TARGET_LAST', 'TARGET_FROZEN', 'FORK_TARGET'):
                require(not any(weights), 'protected target auxiliary weight')
            if name in ('JOINT_FULL', 'FORK_JOINT'):
                require(weights == [1./sum(allowed) if ok else 0. for ok in allowed], 'uniform source policy')
    return full


def expected_optimizer_steps(bank, target_plan, auxiliary):
    return {name: {n: count for n, count, _ in s5.expected_steps(model, opt, target_plan, auxiliary[name]).values()}
            for name, (model, opt) in bank.items()}


def prediction_evaluator(factory, trainer, job, model, validation, scalers, device):
    indices = {t: list(range(len(d))) for t, d in s3.s2.datasets_for(factory, trainer, job['setting'])['validation'].items()}
    cache = {}
    def evaluate(state):
        key = state_hash(state)
        if key not in cache:
            prior = trainer.model; modes = [m.training for m in model.modules()]
            try:
                with evaluation_rng(device):
                    model.load_state_dict(state, strict=True); trainer.model = model
                    cache[key] = s6.evaluate(factory, trainer, job['setting'], validation, indices, scalers, device, split='validation')
            finally:
                trainer.model = prior
                for module, mode in zip(model.modules(), modes): module.training = mode
        return cache[key]
    return evaluate


def prediction_set(name, chosen, states, initial, parameter_names, evaluate):
    models = s6.infer_recipe(chosen['recipe'], states, initial, parameter_names)
    rows = [evaluate(value) for value in models.values()]
    predictions = rows[0] if len(rows) == 1 else method.blend_rows(*rows, chosen['recipe']['coefficients'])
    control = chosen['recipe']['engine'] if name == 'PROTECTED_GATE' else 'TARGET_FULL'
    return models, control, predictions, evaluate(states[control])


def save_checkpoint(path, value):
    """Only the current run's derived best checkpoint may be atomically replaced."""
    pending = path.with_suffix('.pending.pt')
    require(not pending.exists(), 'pending checkpoint already exists')
    with pending.open('xb') as stream: torch.save(value, stream)
    os.replace(pending, path)


def check_original_payload(payload, old, result, rows, paired, spec):
    require(sha(old/(payload['method']+'.pt')) == result['checkpoint_sha256'], 'original checkpoint bytes')
    previous = torch.load(old/(payload['method']+'.pt'), map_location='cpu', weights_only=True)
    require(previous['method'] == payload['method'] and previous['chosen'] == payload['frozen_chosen']
            and previous['epoch'] == payload['epoch'] and previous['scalers'] == payload['scalers']
            and previous['control_engine'] == payload['control_engine'] and set(previous['models']) == set(payload['models']),
            'original checkpoint schema/selection')
    for name, state in payload['models'].items(): exact_state(state, previous['models'][name])
    exact_state(payload['paired_control'], previous['paired_control'])
    for suffix, recorded in (('_validation.json', rows), ('_paired_control.json', paired)):
        s6.compare_replay(recorded, read(old/(payload['method']+suffix)), spec)


def refit_case(factory, trainer, source, job, old, out, commit, device, reference, spec=SPEC):
    started = time.monotonic()
    identity, binding, previous, selection, policies, expected = context_info(factory, trainer, source, job, old, commit, spec)
    s5.pairing(identity, reference)
    environment = dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda, device=str(device),
                       gpu_name=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None)
    require(all(environment[k] == previous['environment'][k] for k in ('torch', 'cuda', 'device', 'gpu_name')), 'original training backend changed')
    out.mkdir(exist_ok=False); (out/'original').mkdir(); (out/'development').mkdir(); (out/'source_evidence').mkdir()
    for name in ('receipt', 'identity', 'frozen_selection', 'frozen_policies', 'inner_split', 'inner_scalers',
                 'full_scalers', 'coverage', *expected):
        src = old/(name+'.json'); dst = out/'source_evidence'/src.name
        if not dst.exists(): shutil.copyfile(src, dst)
    write(out/'identity.json', identity); write(out/'source_binding.json', binding)
    base = deepcopy(trainer.model).cpu()
    evaluator_model = method.build(base, source, 'TARGET_FULL', job['lr'], device)[0]
    s3.seed_everything(42, deterministic_algorithms=True)
    bank = s6.engines(base, source, job['lr'], device)
    initial = method.state(bank['TARGET_FULL'][0]); tasks, counts = s3.s2.c0.tasks_and_counts(job['setting'])
    parameter_names = {name: True for name, _ in bank['TARGET_FULL'][0].named_parameters()}; parameter_names['target_tasks'] = tasks
    evaluate = prediction_evaluator(factory, trainer, job, evaluator_model, expected['validation_observations'], expected['full_scalers'], device)
    initial_rows = evaluate(initial); s3.paired_initial(initial_rows, reference['initial']); write(out/'initial_validation.json', initial_rows)
    split = method.partition(expected['train_observations'], spec['inner_fraction'])
    schedule = s3.aux.Schedule(source.tasks, source.counts, s3.s2.c0.BATCH_SIZES[job['setting']])
    steps = 0; history = []; original = {}; best = {}; target_plan = []; auxiliary = {n: [] for n in method.ENGINES}
    for epoch, policy in enumerate(policies, 1):
        full = validate_policy(policy, epoch-1, tasks, counts, source, expected['train_observations'], split, job['setting'], schedule)
        recorded = read(old/f'refit_epoch_{epoch:03d}.json'); logs = []
        for index, p in enumerate(policy['steps']):
            steps += 1; item = p['full_target']; ai = p['auxiliary']; target_plan.append(item)
            batch = s3.s2.batch_for(factory, trainer, job['setting'], 'train', item['task'], item['indices'], device)
            log = {}
            for name, (model, opt) in bank.items():
                if any(p['weights'][name]): auxiliary[name].append(ai)
                log[name] = method.update(model, opt, batch, item['task'], expected['full_scalers'][item['task']],
                                          source, ai, p['weights'][name], device, steps)
            s6.check_losses(log, p['weights'])
            require(log == recorded['losses'][index], 'original refit loss/gradient trace differs')
            logs.append(log)
            if steps == 2:
                write(out/'startup_check.json', dict(task=TASK, commit=commit, job=job, target_batches=2,
                      model_updates=2*len(method.ENGINES), included_in_formal_run=True, original_logs_matched=True))
        states = {name: method.state(model) for name, (model, _) in bank.items()}
        merged = method.interpolate(states['FORK_TARGET'], states['FORK_JOINT'], policy['fork_alpha'])
        for name in ('FORK_TARGET', 'FORK_JOINT'):
            bank[name][0].load_state_dict(merged, strict=True); states[name] = merged
        optimizer_steps = {n: s3.optimizer_steps(m, opt) for n, (m, opt) in bank.items()}
        require(optimizer_steps == recorded['optimizer_steps'] == expected_optimizer_steps(bank, target_plan, auxiliary), 'paired optimizer steps')
        require(recorded['epoch'] == epoch and recorded['full_target_steps'] == steps and recorded['policy_sha256'] == digest(policy), 'original epoch binding')
        evaluate = prediction_evaluator(factory, trainer, job, evaluator_model, expected['validation_observations'], expected['full_scalers'], device)
        values = {}; all_rows = {}; controls = {}; pairs = {}
        for name in method.METHODS:
            chosen = selection['selected'][name]
            models, control, rows, paired = prediction_set(name, chosen, states, initial, parameter_names, evaluate)
            all_rows[name] = rows; controls[control] = paired; pairs[name] = control
            value = dict(epoch=epoch, selected=s6.metric(rows), paired_control=s6.metric(paired), control_engine=control)
            values[name] = value
            payload = dict(task=TASK, identity=identity, source_binding=binding, method=name, frozen_chosen=chosen,
                           epoch=epoch, models=models, scalers=expected['full_scalers'], paired_control=states[control], control_engine=control)
            if epoch == chosen['epoch']:
                check_original_payload(payload, old, previous['results'][name], rows, paired, spec)
                save_checkpoint(out/'original'/(name+'.pt'), payload)
                original[name] = dict(value, checkpoint_sha256=sha(out/'original'/(name+'.pt')), exact_original_state=True)
            if name not in best or value['selected']['macro_rmse'] < best[name]['selected']['macro_rmse']:
                save_checkpoint(out/'development'/(name+'.pt'), payload)
                best[name] = dict(value, checkpoint_sha256=sha(out/'development'/(name+'.pt')))
        prediction_path = out/f'validation_epoch_{epoch:03d}.json'
        write(prediction_path, dict(methods=all_rows, controls=controls, paired_engines=pairs))
        entry = dict(epoch=epoch, policy_sha256=digest(policy), full_plan_sha256=digest(full),
                     losses=logs, optimizer_steps=optimizer_steps, target_steps=steps,
                     original_trace_exact=True, predictions_sha256=sha(prediction_path), results=values)
        write(out/f'epoch_{epoch:03d}.json', entry); history.append(entry)
        print(f'{job["id"]} locked refit epoch={epoch}/{len(policies)}', flush=True)
    require(set(original) == set(best) == set(method.METHODS), 'complete both selection channels')
    result = dict(task=TASK, commit=commit, job=job, identity=identity, source_binding=binding,
                  original=original, development=best, model_training_runs=len(method.ENGINES),
                  model_epochs=len(method.ENGINES)*len(policies), target_updates_per_engine=steps,
                  total_updates=steps*len(method.ENGINES), calibration_updates=0, extra_smoke_updates=0,
                  test_evaluated=False, scientific_acceptance='PENDING_REVIEW',
                  history_sha256=digest(history), environment=environment, elapsed_seconds=time.monotonic()-started)
    write(out/'receipt.json', result); return result


def verify_case(factory, trainer, source, job, old, out, commit, reference, spec=SPEC):
    identity, binding, previous, selection, policies, expected = context_info(factory, trainer, source, job, old, commit, spec)
    s5.pairing(identity, reference); result = read(out/'receipt.json')
    require(result['task'] == TASK and result['commit'] == commit and result['job'] == job
            and result['identity'] == read(out/'identity.json') == identity
            and result['source_binding'] == read(out/'source_binding.json') == binding, 'new receipt identity')
    for p in (out/'source_evidence').iterdir():
        require(p.is_file() and p.read_bytes() == (old/p.name).read_bytes(), 'source evidence copy')
    required = {'receipt', 'identity', 'frozen_selection', 'frozen_policies', 'inner_split', 'inner_scalers', 'coverage', *expected}
    require({p.stem for p in (out/'source_evidence').iterdir()} == required, 'source evidence members')
    tasks, counts = s3.s2.c0.tasks_and_counts(job['setting']); training = expected['train_observations']; validation = expected['validation_observations']
    split = method.partition(training, spec['inner_fraction'])
    schedule = s3.aux.Schedule(source.tasks, source.counts, s3.s2.c0.BATCH_SIZES[job['setting']])
    bank = s6.engines(trainer.model, source, job['lr'], 'cpu'); initial = method.state(bank['TARGET_FULL'][0])
    evaluator = bank['TARGET_FULL'][0]; target_plan = []; auxiliary = {n: [] for n in method.ENGINES}
    history = []; best = {}; original = {}; steps = 0
    require(read(out/'startup_check.json') == dict(task=TASK, commit=commit, job=job, target_batches=2,
            model_updates=2*len(method.ENGINES), included_in_formal_run=True, original_logs_matched=True), 'included startup check')
    for epoch, policy in enumerate(policies, 1):
        full = validate_policy(policy, epoch-1, tasks, counts, source, training, split, job['setting'], schedule)
        recorded = read(old/f'refit_epoch_{epoch:03d}.json'); h = read(out/f'epoch_{epoch:03d}.json')
        target_plan.extend(full); steps += len(full)
        for p in policy['steps']:
            for name in method.ENGINES:
                if any(p['weights'][name]): auxiliary[name].append(p['auxiliary'])
        require(h['epoch'] == epoch and h['target_steps'] == steps and h['policy_sha256'] == digest(policy)
                and h['full_plan_sha256'] == digest(full) and h['losses'] == recorded['losses']
                and h['optimizer_steps'] == recorded['optimizer_steps'] == expected_optimizer_steps(bank, target_plan, auxiliary)
                and h['original_trace_exact'] is True, 'frozen refit trace/update verification')
        for p, log in zip(policy['steps'], h['losses']): s6.check_losses(log, p['weights'])
        path = out/f'validation_epoch_{epoch:03d}.json'; predictions = read(path)
        require(h['predictions_sha256'] == sha(path) and set(predictions) == {'methods', 'controls', 'paired_engines'}
                and set(predictions['methods']) == set(predictions['paired_engines']) == set(h['results']) == set(method.METHODS),
                'all per-epoch predictions')
        expected_controls = set()
        for name, value in h['results'].items():
            recipe = selection['selected'][name]['recipe']; control = recipe['engine'] if name == 'PROTECTED_GATE' else 'TARGET_FULL'
            expected_controls.add(control)
            require(predictions['paired_engines'][name] == control and value == dict(epoch=epoch,
                    selected=s3.s2.metrics(predictions['methods'][name], validation),
                    paired_control=s3.s2.metrics(predictions['controls'][control], validation), control_engine=control), 'independent epoch metrics/control')
            if name not in best or value['selected']['macro_rmse'] < best[name]['selected']['macro_rmse']: best[name] = value
            if epoch == selection['selected'][name]['epoch']: original[name] = value
        require(set(predictions['controls']) == expected_controls, 'paired control scope')
        history.append(h)
    require(result['history_sha256'] == digest(history), 'complete history identity')
    s3.paired_initial(read(out/'initial_validation.json'), reference['initial'])
    for channel, selected in (('original', original), ('development', best)):
        require(set(result[channel]) == set(method.METHODS), 'complete selected methods')
        for name, value in selected.items():
            path = out/channel/(name+'.pt'); saved = result[channel][name]
            require(saved == dict(value, checkpoint_sha256=sha(path), **({'exact_original_state': True} if channel == 'original' else {})),
                    'independent selected epoch/checkpoint')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            require(set(payload) == {'task', 'identity', 'source_binding', 'method', 'frozen_chosen', 'epoch', 'models', 'scalers', 'paired_control', 'control_engine'}
                    and payload['task'] == TASK and payload['identity'] == identity and payload['source_binding'] == binding
                    and payload['method'] == name and payload['epoch'] == value['epoch'] and payload['frozen_chosen'] == selection['selected'][name]
                    and payload['scalers'] == expected['full_scalers'] and payload['control_engine'] == value['control_engine'], 'selected payload identity')
            require(set(payload['models']) == ({'main', 'second'} if name == 'PROTECTED_GATE' else {'main'}), 'selected model branches')
            for state in [*payload['models'].values(), payload['paired_control']]:
                require(set(state) == set(initial), 'selected tensor names')
                for key, tensor in state.items():
                    require(isinstance(tensor, torch.Tensor) and tensor.dtype == initial[key].dtype
                            and tensor.shape == initial[key].shape and torch.isfinite(tensor).all().item(), 'selected finite tensor shape')
                    if key in dict(evaluator.named_buffers()): require(torch.equal(tensor, initial[key]), 'immutable buffer')
            evaluate = prediction_evaluator(factory, trainer, job, evaluator, validation, expected['full_scalers'], 'cpu')
            parts = [evaluate(state) for state in payload['models'].values()]
            rows = parts[0] if len(parts) == 1 else method.blend_rows(*parts, payload['frozen_chosen']['recipe']['coefficients'])
            paired = evaluate(payload['paired_control'])
            predictions = read(out/f'validation_epoch_{value["epoch"]:03d}.json')
            s6.compare_replay(rows, predictions['methods'][name], spec)
            s6.compare_replay(paired, predictions['controls'][value['control_engine']], spec)
            if name == 'PROTECTED_GATE': exact_state(payload['models']['main'], payload['paired_control'])
            if channel == 'original': check_original_payload(payload, old, previous['results'][name], rows, paired, spec)
    require(result['model_training_runs'] == len(method.ENGINES) and result['model_epochs'] == len(method.ENGINES)*spec['max_epochs']
            and result['target_updates_per_engine'] == steps and result['total_updates'] == steps*len(method.ENGINES)
            and result['calibration_updates'] == result['extra_smoke_updates'] == 0 and result['test_evaluated'] is False
            and result['scientific_acceptance'] == 'PENDING_REVIEW', 'fixed refit scope')
    return result


def compare(receipts, old_root, references):
    require(len(receipts) == 6 and {r['job']['id'] for r in receipts} == {j['id'] for j in jobs()}, 'six refit cases')
    records = []; target_best = {}; choices = {}
    for setting in s3.s2.c0.SETTINGS:
        cases = [r for r in receipts if r['job']['setting'] == setting]
        old_selections = [read(old_root/r['job']['id']/'frozen_selection.json') for r in cases]
        original_lrs = s6.choose_lrs(old_selections)
        strong_name, strong = min(((n, v) for n, v in references.items() if n.startswith(setting+'_')), key=lambda kv: kv[1]['selected']['macro_rmse'])
        target_options = [(r, n) for r in cases for n in method.CONTROLS[:3]]
        target_case, target_name = min(target_options, key=lambda p: (p[0]['development'][p[1]]['selected']['macro_rmse'],
            p[0]['job']['lr'], p[0]['development'][p[1]]['epoch'], method.CONTROLS.index(p[1])))
        target_best[setting] = dict(job=target_case['job'], method=target_name, result=target_case['development'][target_name])
        choices[setting] = {}
        for name in method.METHODS:
            case = min(cases, key=lambda r: (r['development'][name]['selected']['macro_rmse'], r['job']['lr'], r['development'][name]['epoch']))
            original_case = next(r for r in cases if r['job']['id'] == original_lrs[name])
            dev = case['development'][name]; original = original_case['original'][name]
            score = dev['selected']['macro_rmse']; reference = strong['selected']['macro_rmse']
            row = dict(setting=setting, method=name, development_job=case['job'], development=dev,
                       original_job=original_case['job'], original=original,
                       delta_development_minus_original=score-original['selected']['macro_rmse'],
                       best_077_method=strong_name, best_077=strong['selected'],
                       delta_vs_best_077=score-reference, relative_vs_best_077=score/reference-1,
                       delta_vs_exact_paired_control=score-dev['paired_control']['macro_rmse'],
                       delta_vs_target_best_dev=score-target_best[setting]['result']['selected']['macro_rmse'],
                       tox_066_threshold=TOX_066_THRESHOLD if setting == 'ToxAcute' else None,
                       below_tox_066=score < TOX_066_THRESHOLD if setting == 'ToxAcute' else None,
                       endpoint_deltas_vs_best_077={t: v['rmse']-strong['selected']['endpoints'][t]['rmse'] for t, v in dev['selected']['endpoints'].items()})
            records.append(row); choices[setting][name] = dict(original=original_case['job']['id'], development=case['job']['id'])
    ranking = []
    for name in method.CANDIDATES:
        rows = [r for r in records if r['method'] == name]
        ranking.append(dict(method=name, worst_scene_relative=max(r['relative_vs_best_077'] for r in rows),
            mean_scene_relative=math.fsum(r['relative_vs_best_077'] for r in rows)/3,
            all_scene_point_signal=all(r['delta_vs_best_077'] < 0 and r['delta_vs_exact_paired_control'] < 0
                and r['delta_vs_target_best_dev'] < 0 and r['below_tox_066'] is not False for r in rows)))
    ranking.sort(key=lambda r: (r['worst_scene_relative'], r['mean_scene_relative'], r['method']))
    return dict(comparisons=records, lr_choices=choices, target_best_dev=target_best, candidate_ranking=ranking,
                all_scene_candidate_signals=[r['method'] for r in ranking if r['all_scene_point_signal']],
                unified_superiority_confirmed=False, scope='POST_083_DEVELOPMENT_SELECTION_DIAGNOSTIC_NOT_INDEPENDENT_CONFIRMATION')


def gate_check(root, commit, role):
    require(read(root/'gate.json') == dict(task=TASK, commit=commit, role=role, tests=list(TESTS), junit_sha256=sha(root/'tests.xml')), 'S6R1 gate identity')
    require(read(root/'command.json')['exit_code'] == 0, 'S6R1 gate exit')
    cases = ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases) == TEST_COUNT and all(not any(c.findall(t) for t in ('failure', 'error', 'skipped')) for c in cases)
            and {c.attrib['classname'] for c in cases} == {'tests.'+Path(t).stem for t in TESTS}, 'S6R1 complete gate')


def code_gate(root, commit, role):
    require(role in ('wsl', 'server'), 'explicit gate role'); s3.s2.c0.check_code(REPO, commit)
    root.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-m', 'pytest', *TESTS, '-q', '--basetemp', str(root/'pytest_tmp'), '--junitxml', str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as log:
        p = subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json', dict(argv=argv, exit_code=p.returncode)); require(p.returncode == 0, 'S6R1 code tests')
    write(root/'gate.json', dict(task=TASK, commit=commit, role=role, tests=list(TESTS), junit_sha256=sha(root/'tests.xml')))
    gate_check(root, commit, role)


def claim_value(root, old, commit):
    return dict(task=TASK, commit=commit, output=str(root.resolve()), source_root=str(old.resolve()), spec=SPEC, jobs=jobs())


def check_launch(root, old, commit):
    claim = claim_value(root, old, commit); require(read(REPO/REGISTRY/'attempt.json') == claim, 'S6R1 one-shot claim')
    launch = read(root/'launch.json')
    require(launch['claim'] == claim and launch['task'] == TASK and launch['commit'] == commit, 'S6R1 launch identity')
    return launch


def worker_command(mode, job, root, old, commit, split, source_lock):
    return [sys.executable, str(Path(__file__).resolve()), '_worker', '--mode', mode, '--job', job, '--output', str(root),
            '--source-root', str(old), '--commit', commit, '--split-manifest', str(split), '--source-lock', str(source_lock)]


def verify(root, old, commit, split, source_lock):
    check_launch(root, old, commit)
    for role in ('wsl', 'server'): gate_check(root/(role+'_evidence'), commit, role)
    verify_source(old, manifest=(root/'source_manifest.sha256').read_bytes())
    references = s3.references(root/'reference_077'); receipts = []; pids = []
    for job in jobs():
        name = job['id']; require(read(root/(name+'.command.json'))['exit_code'] == 0, 'refit worker exit')
        s5.call_worker(worker_command('verify', name, root, old, commit, split, source_lock), root/('verify_'+name),
                       dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        v = read(root/name/'verification.json'); r = read(root/name/'receipt.json')
        require(v == dict(task=TASK, commit=commit, job=name, pid=v['pid'], device='cpu', content_status='PASS',
                selected_raw_prediction_replayed=True, new_optimizer_updates=0, receipt_sha256=sha(root/name/'receipt.json'))
                and type(v['pid']) is int, 'independent CPU verification')
        receipts.append(r); pids.append(v['pid'])
    require(len(set(pids)) == 6 and sum(r['total_updates'] for r in receipts) == 26400, 'six independent cases/update count')
    return dict(task=TASK, commit=commit, content_status='PASS', scientific_acceptance='PENDING_REVIEW',
                model_training_runs=66, model_epochs=2640, total_updates=26400, calibration_updates=0, extra_smoke_updates=0,
                verification_pids=pids, source_manifest_sha256=SOURCE_MANIFEST_SHA256, **compare(receipts, old, references))


def run(a):
    from p1d4_batch import free_gpus
    s3.s2.c0.check_code(REPO, a.commit)
    for role in ('wsl', 'server'): gate_check(getattr(a, role+'_evidence'), a.commit, role)
    source_snapshot = verify_source(a.source_root); s3.references(a.source_root/'reference_077')
    require(not a.output.exists() and not (REPO/REGISTRY/'attempt.json').exists(), 'S6R1 output/attempt already exists')
    require(not a.output.resolve().is_relative_to(a.source_root.resolve())
            and not a.source_root.resolve().is_relative_to(a.output.resolve()), 'new output overlaps immutable 083')
    available = free_gpus(a.gpus); require(bool(available), 'no idle authorized GPU')
    a.output.mkdir(parents=True, exist_ok=False); (REPO/REGISTRY).mkdir(parents=True, exist_ok=True)
    claim = claim_value(a.output, a.source_root, a.commit); write(REPO/REGISTRY/'attempt.json', claim)
    uuids = {str(g): s3.s2.c0.gpu_uuid(g) for g in available}
    write(a.output/'launch.json', dict(task=TASK, commit=a.commit, claim=claim, gpus=available, gpu_uuids=uuids))
    try:
        write(a.output/'source_snapshot.json', source_snapshot)
        with (a.output/'source_manifest.sha256').open('xb') as stream: stream.write(manifest_bytes(a.source_root))
        for role in ('wsl', 'server'):
            shutil.copytree(getattr(a, role+'_evidence'), a.output/(role+'_evidence'), ignore=shutil.ignore_patterns('pytest_tmp'))
        s3.copy_references(a.source_root/'reference_077', a.output/'reference_077')
        def execute(job, gpu):
            require(free_gpus([gpu]) == [gpu], 'GPU occupied'); name = job['id']
            write(a.output/(name+'.assignment.json'), dict(task=TASK, commit=a.commit, job=name, gpu=gpu, gpu_uuid=uuids[str(gpu)]))
            s5.call_worker(worker_command('refit', name, a.output, a.source_root, a.commit, a.split_manifest, a.source_lock), a.output/name,
                           dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[str(gpu)], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        s5.dispatch(jobs(), available, execute)
        write(a.output/'verification.json', verify(a.output, a.source_root, a.commit, a.split_manifest, a.source_lock))
    except BaseException as exc:
        write(a.output/'failed.json', dict(task=TASK, commit=a.commit, error_type=type(exc).__name__, reason=str(exc))); raise


def worker(a):
    launch = check_launch(a.output, a.source_root, a.commit); job = next(j for j in jobs() if j['id'] == a.job)
    verify_source(a.source_root, job=a.job, manifest=(a.output/'source_manifest.sha256').read_bytes())
    out = a.output/a.job; old = a.source_root/a.job
    if a.mode == 'verify':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU verification only'); torch.set_num_threads(1)
        require(not (out/'verification.json').exists(), 'verification already exists')
        f, t, source = s3.context(job['setting'], a.split_manifest, a.source_lock, 'cpu')
        ref = s3.references(a.output/'reference_077')[f'{job["setting"]}_B1_1E4_s42']
        verify_case(f, t, source, job, old, out, a.commit, ref)
        write(out/'verification.json', dict(task=TASK, commit=a.commit, job=a.job, pid=os.getpid(), device='cpu', content_status='PASS',
              selected_raw_prediction_replayed=True, new_optimizer_updates=0, receipt_sha256=sha(out/'receipt.json')))
    else:
        require(a.mode == 'refit', 'refit only')
        assignment = read(a.output/(a.job+'.assignment.json'))
        require(assignment == dict(task=TASK, commit=a.commit, job=a.job, gpu=assignment['gpu'],
                gpu_uuid=launch['gpu_uuids'][str(assignment['gpu'])]) and assignment['gpu'] in launch['gpus'], 'GPU assignment identity')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == assignment['gpu_uuid'] and torch.cuda.is_available()
                and torch.cuda.device_count() == 1, 'single GPU binding')
        f, t, source = s3.context(job['setting'], a.split_manifest, a.source_lock, 'cuda:0')
        ref = s3.references(a.output/'reference_077')[f'{job["setting"]}_B1_1E4_s42']
        refit_case(f, t, source, job, old, out, a.commit, 'cuda:0', ref)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('code-gate', 'source-check', 'run', '_worker', 'verify', 'package'))
    p.add_argument('--commit', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--role', choices=('wsl', 'server')); p.add_argument('--mode', choices=('refit', 'verify'))
    p.add_argument('--job', choices=[j['id'] for j in jobs()]); p.add_argument('--gpus', type=int, nargs='+', choices=range(4))
    for name in ('source-root', 'split-manifest', 'source-lock', 'wsl-evidence', 'server-evidence'): p.add_argument('--'+name, type=Path)
    a = p.parse_args(); a.output = a.output.resolve()
    if a.action == 'code-gate': code_gate(a.output, a.commit, a.role)
    elif a.action == 'package':
        require(read(a.output/'launch.json')['commit'] == a.commit, 'package identity')
        if (a.output/'verification.json').exists():
            v = read(a.output/'verification.json'); require(v['task'] == TASK and v['commit'] == a.commit and v['content_status'] == 'PASS', 'verified package')
        else:
            v = read(a.output/'failed.json'); require(v['task'] == TASK and v['commit'] == a.commit, 'failed partial package')
        print(s3.s2.package(a.output, a.commit))
    else:
        s3.s2.c0.check_code(REPO, a.commit); require(a.source_root is not None, '083 source root required')
        a.source_root = a.source_root.resolve()
        if a.action == 'source-check': print(verify_source(a.source_root))
        else:
            require(a.split_manifest is not None and a.source_lock is not None, 'original asset paths required')
            if a.action == 'run':
                require(a.gpus is not None and a.wsl_evidence is not None and a.server_evidence is not None, 'run inputs'); run(a)
            elif a.action == 'verify': print(verify(a.output, a.source_root, a.commit, a.split_manifest, a.source_lock))
            else: worker(a)


if __name__ == '__main__': main()
