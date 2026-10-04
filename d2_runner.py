"""Bounded D2-S1 training and independent validation replay, never test inference."""
from collections import Counter
from pathlib import Path
import hashlib
import math
import time

import torch

import d2_control as core
from reproducibility import seed_everything, state_dict_sha256
from v9_cost_probe import read, write, sha, digest, require
from v9_s2_screen import metrics

TASK = 'D2_S1_INITIALIZATION_JOINT_CONTROL_20261004'


def evaluate(model, data, device, batch_size):
    model.eval()
    rows = []
    with torch.no_grad():
        for task, ds in data.datasets['validation'].items():
            for start in range(0, len(ds), batch_size):
                indices = list(range(start, min(start+batch_size, len(ds))))
                batch = data.batch('target', 'validation', task, indices, device)
                raw = model(batch, task).double()
                scaler = data.scalers['target'][task]
                predictions = (raw*scaler['std']+scaler['mean']).cpu().tolist()
                rows.extend(dict(data.by_task['validation'][task][i], prediction=float(p))
                            for i, p in zip(indices, predictions))
    return rows, metrics(rows, data.rows['validation'])


def initial_identity(model, data, job, commit, spec):
    return dict(task=TASK, job=job, commit=commit, spec=spec, data=data.identity,
                initial_encoder=state_dict_sha256(model.encoder),
                initial_heads=state_dict_sha256(model.decoders),
                total_steps=core.update_count(data.source_counts, spec),
                validation_steps=core.evaluation_steps(data.source_counts, spec))


def cpu_state(model):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    require(all(torch.isfinite(v).all() for v in state.values()), 'finite checkpoint tensors')
    return state


def save_checkpoint(path, model, opt, identity, step):
    # Only this run's own explicitly named checkpoint is replaced.
    temp = path.with_suffix('.writing')
    require(not temp.exists(), 'incomplete checkpoint exists')
    with temp.open('xb') as f:
        torch.save(dict(identity=identity, step=step, model_state=cpu_state(model),
                        optimizer_state=opt.state_dict()), f)
    temp.replace(path)


def exposure_add(counts, row):
    counts[row['task']] += len(row['indices'])


def fresh_exposure(data):
    return {role: Counter({t: 0 for t in tasks}) for role, tasks in
            [('target', data.target_tasks), ('source', data.source_tasks)]}


def train_one(data, job, output, commit, device, spec=core.SPEC):
    core.check_job(job)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    seed_everything(job['seed'], deterministic_algorithms=True)
    model = data.model(job, device)
    opt = core.optimizer(model, job['encoder_lr'], spec)
    from rdkit import rdBase
    write(root/'environment.json', dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
          rdkit_version=rdBase.rdkitVersion, device=device,
          device_name=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else 'cpu',
          deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    identity = initial_identity(model, data, job, commit, spec)
    write(root/'identity.json', identity)
    write(root/'target_train.json', data.rows['train'])
    write(root/'target_validation.json', data.rows['validation'])
    initial_encoder = identity['initial_encoder']
    initial_heads = identity['initial_heads']
    history, audits = [], []
    source_epoch_losses = []
    total_exposure = fresh_exposure(data)
    trace = hashlib.sha256()
    checkpoints = set(identity['validation_steps'])
    best = float('inf')
    losses = dict(target=[], source=[])
    epoch_losses = dict(target=[], source=[])
    started = time.monotonic()
    for item in core.schedule(data.target_counts, data.source_counts, job['seed'], spec):
        step = item['step']
        trace.update(digest(item).encode('ascii'))
        record = core.update(model, opt, data, item, job['seed'], device, spec, audit=step <= 2)
        for role, value in record['losses'].items():
            losses[role].append(value)
            epoch_losses[role].append(value)
            exposure_add(total_exposure[role], item[role])
        if step <= 2:
            audits.append(dict(step=step, batch=item, **record))
        if step == 2:
            require(state_dict_sha256(model.decoders) != initial_heads, 'real smoke: heads unchanged')
            require((state_dict_sha256(model.encoder) == initial_encoder) == (job['condition'] == 'FJ'), 'real smoke: freeze/plasticity')
            for a in audits:
                g = a['gradient_l1']
                require(g['target'] > 0 and (g['encoder'] > 0) == (job['condition'] != 'FJ')
                        and (g['source'] > 0) == model.joint, 'real smoke: nonzero gradient roles')
            write(root/'first_two_updates.json', dict(status='PASS', updates=audits,
                  scope='FIRST_TWO_UPDATES_OF_SAME_TRAINING_TRAJECTORY', discarded_training=False))
        if item['source_pass_end']:
            source_epoch_losses.append(dict(source_clock_pass=item['source_pass'], step=step,
                mean_batch_mse={r: math.fsum(v)/len(v) if v else None for r, v in epoch_losses.items()},
                cumulative_exposure={r: dict(v) for r, v in total_exposure.items()}))
            epoch_losses = dict(target=[], source=[])
        if step in checkpoints:
            rows, scores = evaluate(model, data, device, spec['target_batch_size'])
            name = f'validation_{step:07d}.json'
            write(root/name, rows)
            h = dict(step=step, source_clock_pass=item['source_pass'], metrics=scores,
                     predictions=name, predictions_sha256=sha(root/name), schedule_sha256=trace.hexdigest(),
                     cumulative_exposure={r: dict(v) for r, v in total_exposure.items()},
                     mean_batch_mse={r: math.fsum(v)/len(v) if v else None for r, v in losses.items()})
            history.append(h)
            write(root/f'validation_{step:07d}.receipt.json', h)
            losses = dict(target=[], source=[])
            if scores['macro_rmse'] < best:
                best = scores['macro_rmse']
                save_checkpoint(root/'best.pt', model, opt, identity, step)
            print(f'{job["id"]} step={step}/{identity["total_steps"]} source_pass={item["source_pass"]} validation_macro_rmse={scores["macro_rmse"]:.7f}', flush=True)
    require([r['step'] for r in history] == identity['validation_steps'], 'complete training checkpoints')
    require(len(source_epoch_losses) == spec['source_passes'], 'complete source clock')
    for t, n in data.source_counts.items():
        require(total_exposure['source'][t] == (n*spec['source_passes'] if model.joint else 0), 'source full coverage')
    save_checkpoint(root/'last.pt', model, opt, identity, identity['total_steps'])
    selected = core.select(history)
    write(root/'history.json', history)
    write(root/'source_pass_history.json', source_epoch_losses)
    result = dict(job=job, identity=identity, selected=selected, completed_updates=identity['total_steps'],
                  source_passes_completed=spec['source_passes'] if model.joint else 0,
                  exposure={r: dict(v) for r, v in total_exposure.items()},
                  schedule_sha256=trace.hexdigest(), best_sha256=sha(root/'best.pt'), last_sha256=sha(root/'last.pt'),
                  final_encoder=state_dict_sha256(model.encoder), elapsed_seconds=time.monotonic()-started,
                  best_at_final_checkpoint=selected['step'] == identity['total_steps'],
                  scientific_status='PENDING_REVIEW', test_accessed=False, calibration_accessed=False)
    write(root/'training.json', result)
    return result


def assert_close(actual, expected, tolerance, message):
    require(type(actual) is type(expected), message+' type')
    if isinstance(actual, dict):
        require(actual.keys() == expected.keys(), message+' fields')
        for key in actual:
            assert_close(actual[key], expected[key], tolerance, message+'.'+key)
    elif isinstance(actual, list):
        require(len(actual) == len(expected), message+' length')
        for a, b in zip(actual, expected):
            assert_close(a, b, tolerance, message)
    elif isinstance(actual, float):
        require(math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected) <= tolerance, message)
    else:
        require(actual == expected, message)


def check_checkpoint(path, expected_sha, data, job, identity, expected_step, device, spec):
    require(sha(path) == expected_sha, 'checkpoint SHA')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(digest(payload['identity']) == digest(identity) and type(payload['step']) is int
            and payload['step'] == expected_step, 'checkpoint trusted identity/step')
    model = data.model(job, device)
    initial = cpu_state(model)
    model.load_state_dict(payload['model_state'], strict=True)
    require(all(torch.isfinite(v).all() for v in payload['model_state'].values()), 'checkpoint finite tensors')
    if job['condition'] == 'FJ':
        require(state_dict_sha256(model.encoder) == identity['initial_encoder'], 'frozen encoder changed')
    else:
        require(state_dict_sha256(model.encoder) != identity['initial_encoder'], 'plastic encoder never changed')
    if not model.joint:
        for t in data.source_tasks:
            require(all(torch.equal(v, initial['decoders.'+t+'.'+k])
                        for k, v in model.decoders[t].state_dict().items()), 'target-only source head changed')
    opt = core.optimizer(model, job['encoder_lr'], spec)
    expected_groups = [{k: v for k, v in g.items() if k != 'params'} for g in opt.state_dict()['param_groups']]
    stored_groups = [{k: v for k, v in g.items() if k != 'params'} for g in payload['optimizer_state']['param_groups']]
    require(digest(stored_groups) == digest(expected_groups), 'optimizer full configuration')
    opt.load_state_dict(payload['optimizer_state'])
    expected_tasks = Counter()
    for item in core.schedule(data.target_counts, data.source_counts, job['seed'], spec):
        if item['step'] > expected_step:
            break
        expected_tasks[item['target']['task']] += 1
        if model.joint:
            expected_tasks[item['source']['task']] += 1
    for name, p in model.named_parameters():
        state = opt.state.get(p)
        if name.startswith('encoder.'):
            inactive = '.direct_bond_embeddings.' in name and data.args.edge_bias_mode == 'path'
            inactive |= '.path_bond_embeddings.' in name and data.args.edge_bias_mode == 'direct'
            count = 0 if job['condition'] == 'FJ' or inactive else expected_step
        else:
            count = expected_tasks[name.split('.')[1]]
        if count == 0:
            require(state is None and torch.equal(p.detach().cpu(), initial[name]), 'inactive parameter/state changed '+name)
        else:
            require(state is not None and float(state['step']) == count, 'actual optimizer steps '+name)
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and all(torch.isfinite(v).all() for v in state.values()), 'optimizer state finite/schema')
            require(state['exp_avg'].shape == p.shape and state['exp_avg_sq'].shape == p.shape and
                    (state['exp_avg_sq'] >= 0).all(), 'optimizer moments')
    expected_lrs = ([job['encoder_lr']] if job['condition'] != 'FJ' else [])+[spec['head_lr']]
    require([g['lr'] for g in opt.param_groups] == expected_lrs and
            all(g['weight_decay'] == spec['weight_decay'] for g in opt.param_groups), 'optimizer configuration')
    require(sha(path) == expected_sha, 'checkpoint changed during verification')
    return model


def verify_one(data, job, output, commit, device='cpu', spec=core.SPEC):
    core.check_job(job)
    root = Path(output)
    model = data.model(job, device)
    identity = initial_identity(model, data, job, commit, spec)
    require(digest(read(root/'identity.json')) == digest(identity), 'independent trusted run identity')
    require(read(root/'target_train.json') == data.rows['train'] and
            read(root/'target_validation.json') == data.rows['validation'], 'independent target observations')
    result = read(root/'training.json')
    require(digest(result['identity']) == digest(identity) and digest(result['job']) == digest(job), 'result identity')
    history = read(root/'history.json')
    require([r['step'] for r in history] == identity['validation_steps'], 'complete validation schedule')
    exposures = fresh_exposure(data)
    trace = hashlib.sha256()
    at_step = {r['step']: r for r in history}
    epoch_rows = read(root/'source_pass_history.json')
    require(len(epoch_rows) == spec['source_passes'], 'source pass history completeness')
    for item in core.schedule(data.target_counts, data.source_counts, job['seed'], spec):
        trace.update(digest(item).encode('ascii'))
        for role in ('target', 'source') if model.joint else ('target',):
            exposure_add(exposures[role], item[role])
        if item['source_pass_end']:
            ep = epoch_rows[item['source_pass']-1]
            require(ep['source_clock_pass'] == item['source_pass'] and ep['step'] == item['step'] and
                    ep['cumulative_exposure'] == exposures, 'full source pass exposure')
        if item['step'] in at_step:
            row = at_step[item['step']]
            expected_name = f'validation_{item["step"]:07d}.json'
            require(row['predictions'] == expected_name and sha(root/expected_name) == row['predictions_sha256'], 'prediction bytes/path')
            require(row == read(root/f'validation_{item["step"]:07d}.receipt.json'), 'checkpoint history receipt')
            require(row['cumulative_exposure'] == exposures and row['schedule_sha256'] == trace.hexdigest(), 'independent exposure/schedule')
            require(row['metrics'] == metrics(read(root/expected_name), data.rows['validation']), 'independently recomputed metrics')
    selected = core.select(history)
    require(result['selected'] == selected and result['completed_updates'] == identity['total_steps'] and
            result['exposure'] == exposures and result['schedule_sha256'] == trace.hexdigest(), 'result selection/coverage')
    require(result['source_passes_completed'] == (spec['source_passes'] if model.joint else 0) and
            result['test_accessed'] is False and result['calibration_accessed'] is False, 'training scope')
    smoke = read(root/'first_two_updates.json')
    first_two = []
    for item in core.schedule(data.target_counts, data.source_counts, job['seed'], spec):
        first_two.append(item)
        if len(first_two) == 2:
            break
    require(smoke['status'] == 'PASS' and smoke['discarded_training'] is False and
            [r['batch'] for r in smoke['updates']] == first_two, 'real smoke stream')
    for row in smoke['updates']:
        g = row['gradient_l1']
        require(g['target'] > 0 and (g['encoder'] > 0) == (job['condition'] != 'FJ') and
                (g['source'] > 0) == model.joint and all(math.isfinite(v) for v in g.values()), 'smoke gradient paths')
    best_model = check_checkpoint(root/'best.pt', result['best_sha256'], data, job, identity, selected['step'], device, spec)
    replay, replay_metrics = evaluate(best_model, data, device, spec['target_batch_size'])
    assert_close(replay, read(root/selected['predictions']), 2e-4, 'fresh-process validation replay')
    assert_close(replay_metrics, selected['metrics'], 2e-4, 'fresh-process metric replay')
    last_model = check_checkpoint(root/'last.pt', result['last_sha256'], data, job, identity, identity['total_steps'], device, spec)
    require(state_dict_sha256(last_model.encoder) == result['final_encoder'], 'final encoder digest')
    return dict(status='CONTENT_PASS', job=job, identity_sha256=digest(identity),
                training_sha256=sha(root/'training.json'), best_sha256=result['best_sha256'],
                last_sha256=result['last_sha256'], selected=selected, replay_max_abs_tolerance=2e-4,
                verification_device=device, scientific_acceptance=False, test_accessed=False)
