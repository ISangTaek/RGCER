"""Ten candidate methods plus one control across three fixed validation scenes."""
from pathlib import Path
import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import xml.etree.ElementTree as ET

import torch
import v9_s3_screen as s3
import v9_s3r1_screen as r1
import v9_method_portfolio as method

REPO = Path(__file__).resolve().parent
TASK = 'V9_S5_TEN_METHOD_SCREEN_20261002'
REGISTRY = '.tmp/v9s5_attempt_20261002'
SPEC = dict(s3.SPEC, method_schema='v9_s5_v1', warmup_epochs=0, full_encoder_lr=.001, adapter_lr=.001, architecture=method.SPEC,
            prediction_replay_rtol=.001, prediction_replay_atol=.00001)
SMOKE = tuple(('ToxAcute' if i % 2 == 0 else 'A', m) for i, m in enumerate(method.METHODS))
TESTS = ('tests/test_v9_method_portfolio.py', 'tests/test_v9_s5_screen.py', *r1.TESTS)
TEST_COUNT = 277
read, write, sha, digest, require = s3.read, s3.write, s3.sha, s3.digest, s3.require


def jobs():
    return [dict(id=f'{s}_{m}_s42', setting=s, method=m, encoder_lr=.001, seed=42)
            for m in method.METHODS for s in s3.s2.c0.SETTINGS]


def pairing(identity, reference):
    old = reference['identity']
    require(identity['initial_encoder'] == old['initial_encoder'] and identity['initial_target_heads'] == old['initial_heads']
            and all(identity[k] == old[k] for k in ('contract_sha256', 'train_sha256', 'validation_sha256')), '077 pairing')


def observations(factory, trainer, source, job, commit, spec):
    setting = job['setting']
    training = s3.s2.observations(factory, trainer, setting, 'train')
    validation = s3.s2.observations(factory, trainer, setting, 'validation')
    populations = s3.population_check(factory, source, setting, training, validation)
    identity = s3.identity_for(trainer, source, job, training, validation, commit, spec, task_id=TASK)
    _, counts = s3.s2.c0.tasks_and_counts(setting)
    require({t: len(d) for t, d in s3.s2.datasets_for(factory, trainer, setting)['train'].items()} == counts, 'target counts')
    return identity, dict(train_observations=training, validation_observations=validation,
                          source_train_observations=source.rows, population_check=populations)


def train_one(factory, trainer, source, job, out, commit, device, reference, spec=SPEC):
    require(job in jobs(), 'S5 fixed job')
    setting = job['setting']; tasks, counts = s3.s2.c0.tasks_and_counts(setting)
    identity, data = observations(factory, trainer, source, job, commit, spec)
    pairing(identity, reference)
    write(out/'identity.json', identity)
    for name, value in data.items():
        write(out/(name+'.json'), value)
    s3.seed_everything(42, deterministic_algorithms=True)
    trainer.model, opt = method.build(trainer.model, source, job['method'], job['encoder_lr'], device)
    model = trainer.model; validation = data['validation_observations']
    rows, _ = s3.s2.evaluate(factory, trainer, setting, None, validation, device, 0)
    s3.paired_initial(rows, reference['initial']); write(out/'initial_validation.json', rows)
    schedule = s3.aux.Schedule(source.tasks, source.counts, s3.s2.c0.BATCH_SIZES[setting])
    history = []; all_aux = []; updates = 0; best = math.inf; start = time.monotonic()
    for epoch in range(spec['max_epochs']):
        model.train()
        plan = s3.s2.epoch_plan(tasks, counts, s3.s2.c0.BATCH_SIZES[setting], epoch)
        auxiliary = []; losses = []
        for item in plan:
            a = schedule.next(); auxiliary.append(a); all_aux.append(a)
            batch = s3.s2.batch_for(factory, trainer, setting, 'train', item['task'], item['indices'], device)
            losses.append(method.step(trainer, setting, batch, item['task'], epoch, source, a, opt, device))
            updates += 1
        rows, score = s3.s2.evaluate(factory, trainer, setting, None, validation, device, epoch)
        h = dict(epoch=epoch+1, validation=score, updates=len(plan), total_updates=updates,
                 schedule_sha256=digest(plan), auxiliary_schedule=auxiliary, losses=losses,
                 phase='FIXED_PORTFOLIO', optimizer_steps=s3.optimizer_steps(model, opt))
        history.append(h)
        write(out/f'validation_epoch_{epoch+1:03d}.json', rows); write(out/f'epoch_{epoch+1:03d}.json', h)
        if score['macro_rmse'] < best:
            best = score['macro_rmse']; best_epoch = epoch+1
            with (out/'best.pending.pt').open('xb') as stream:
                torch.save(dict(identity=identity, epoch=best_epoch,
                                model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                optimizer_state=opt.state_dict(), algorithm_state=method.algorithm_state(opt)), stream)
            os.replace(out/'best.pending.pt', out/'best.pt')
        print(f'{job["id"]} epoch={epoch+1} macro_rmse={score["macro_rmse"]:.8f} best_epoch={best_epoch}', flush=True)
    require(s3.s2.select(history, spec)[0] == best_epoch, 'selection')
    coverage = s3.aux.coverage(source, all_aux)
    require(all(r['updates'] > 0 for r in coverage.values()), 'unvisited auxiliary task')
    write(out/'auxiliary_coverage.json', coverage)
    payload = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model_state'], strict=True)
    rows, score = s3.s2.evaluate(factory, trainer, setting, None, validation, device, best_epoch-1)
    require(rows == read(out/f'validation_epoch_{best_epoch:03d}.json'), 'selected checkpoint exact replay')
    write(out/'selected_validation.json', rows)
    result = dict(task=TASK, job=job, commit=commit, identity=identity, history=history, best_epoch=best_epoch,
                  updates=updates, selected=score, checkpoint_sha256=sha(out/'best.pt'),
                  selected_state_sha256=s3.state_dict_sha256(model), parameter_counts=method.parameter_counts(model),
                  checkpoint_validation_replay=True, test_evaluated=False, scientific_acceptance='PENDING_REVIEW',
                  auxiliary_batches=len(all_aux), target_batches=updates, coverage=coverage, elapsed_seconds=time.monotonic()-start,
                  environment=dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda, device=str(device),
                                   gpu_name=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None))
    write(out/'receipt.json', result)
    return result


def expected_steps(model, opt, target_plan, auxiliary_plan):
    counts = {t: sum(r['task'] == t for r in target_plan+auxiliary_plan) for t in model.task_name}
    names = {id(p): n for n, p in model.named_parameters()}; expected = {}
    for i, p in enumerate(p for g in opt.param_groups for p in g['params']):
        name = names[id(p)]
        count = counts[name.split('.')[1]] if name.startswith(('decoders.', 'private.')) else len(target_plan)
        mode = model.encoder.backbone.edge_bias_mode
        if (mode == 'path' and '.direct_bond_embeddings.' in '.'+name) or (mode == 'direct' and '.path_bond_embeddings.' in '.'+name):
            count = 0
        if count:
            expected[i] = (name, count, p)
    return expected


def check_optimizer(model, opt, actual, recorded, target_plan, auxiliary_plan):
    require(set(actual) == {'state', 'param_groups'} and actual['param_groups'] == opt.state_dict()['param_groups'], 'optimizer groups')
    expected = expected_steps(model, opt, target_plan, auxiliary_plan)
    require(set(actual['state']) == set(expected) and recorded == {n: count for n, count, p in expected.values()}, 'optimizer coverage/steps')
    for i, (name, count, p) in expected.items():
        v = actual['state'][i]
        require(set(v) == {'step', 'exp_avg', 'exp_avg_sq'} and float(v['step']) == count
                and all(torch.isfinite(x).all().item() for x in v.values())
                and all(v[k].shape == p.shape and v[k].dtype == p.dtype for k in ('exp_avg', 'exp_avg_sq')), 'Adam tensor '+name)


def check_losses(losses, count, block_names, kind):
    require(len(losses) == count, 'loss batch count')
    for loss in losses:
        require(set(loss) == {'target', 'auxiliary', 'regularization', 'gradient_norms', 'rule', 'diagnostics'}, 'loss fields')
        require(set(loss['gradient_norms']) == {'encoder', 'target', 'source'}, 'gradient scopes')
        require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in
                    [loss['target'], loss['auxiliary'], *loss['gradient_norms'].values()]), 'finite losses/norms')
        require(math.isfinite(loss['regularization']) and loss['regularization'] >= -1e-6, 'regularization finite')
        if kind not in ('TPRS', 'TPO_FT'):
            require(loss['rule'] == kind and set(loss['diagnostics']) == {'target_norm', 'auxiliary_norm', 'dot_before', 'dot_target_after', 'dot_auxiliary_after'}
                    and all(type(v) in (int, float) and math.isfinite(v) for v in loss['diagnostics'].values()), 'method diagnostics')
            continue
        require(loss['rule'] == 'TARGET_PRIORITY' and set(loss['diagnostics']) == block_names, 'target-priority scopes')
        for b in loss['diagnostics'].values():
            require(set(b) == {'target_norm', 'auxiliary_norm', 'dot_before', 'dot_after', 'auxiliary_norm_after',
                               'projection_coefficient', 'auxiliary_scale', 'tolerance'}
                    and all(type(v) in (int, float) and math.isfinite(v) for v in b.values()), 'block diagnostics')
            tn, an = b['target_norm'], b['auxiliary_norm']
            tol = 1e-6*max(tn*an, 1e-12)
            coefficient = min(b['dot_before']/tn**2, 0.) if tn > 0 else 0.
            require(tn >= 0 and an >= 0 and 0 <= b['auxiliary_scale'] <= 1
                    and math.isclose(b['tolerance'], tol, rel_tol=1e-6, abs_tol=1e-18)
                    and math.isclose(b['projection_coefficient'], coefficient, rel_tol=1e-6, abs_tol=1e-12)
                    and abs(b['dot_before']) <= tn*an+tol and b['dot_after'] >= -tol
                    and 0 <= b['auxiliary_norm_after'] <= tn*(1+1e-6)+1e-12, 'recorded gradient constraints')


def verify_job(factory, trainer, source, job, out, commit, reference, spec=SPEC):
    r = read(out/'receipt.json'); setting = job['setting']; tasks, counts = s3.s2.c0.tasks_and_counts(setting)
    identity, data = observations(factory, trainer, source, job, commit, spec); pairing(identity, reference)
    require(r['identity'] == read(out/'identity.json') == identity and r['task'] == TASK and r['commit'] == commit and r['job'] == job, 'receipt identity')
    for name, value in data.items():
        require(read(out/(name+'.json')) == value, 'trusted '+name)
    s3.seed_everything(42, deterministic_algorithms=True)
    trainer.model, opt = method.build(trainer.model, source, job['method'], job['encoder_lr'], 'cpu')
    model = trainer.model; validation = data['validation_observations']
    require(r['parameter_counts'] == method.parameter_counts(model), 'parameter counts')
    initial_rows, _ = s3.s2.evaluate(factory, trainer, setting, None, validation, 'cpu', 0)
    s3.paired_initial(read(out/'initial_validation.json'), reference['initial'])
    s3.paired_initial(initial_rows, read(out/'initial_validation.json'))
    schedule = s3.aux.Schedule(source.tasks, source.counts, s3.s2.c0.BATCH_SIZES[setting]); tp = []; ap = []
    history = r['history']; best, _ = s3.s2.select(history, spec)
    blocks = {method.priority.block_name('encoder.'+n) for n, _ in model.encoder.named_parameters()}
    for epoch, h in enumerate(history):
        plan = s3.s2.epoch_plan(tasks, counts, s3.s2.c0.BATCH_SIZES[setting], epoch)
        aq = [schedule.next() for _ in plan]; tp += plan; ap += aq
        require(h == read(out/f'epoch_{epoch+1:03d}.json') and h['schedule_sha256'] == digest(plan) and h['auxiliary_schedule'] == aq
                and h['updates'] == len(plan) and h['total_updates'] == len(tp) and h['phase'] == 'FIXED_PORTFOLIO', 'paired schedule/phase')
        require(h['optimizer_steps'] == {n: c for n, c, p in expected_steps(model, opt, tp, ap).values()}, 'epoch optimizer steps')
        require(h['validation'] == s3.s2.metrics(read(out/f'validation_epoch_{epoch+1:03d}.json'), validation), 'independent epoch metrics')
        check_losses(h['losses'], len(plan), blocks, job['method'])
    require(r['best_epoch'] == best and r['updates'] == r['target_batches'] == r['auxiliary_batches'] == len(tp), 'selection/update count')
    require(r['coverage'] == read(out/'auxiliary_coverage.json') == s3.aux.coverage(source, ap)
            and all(x['updates'] > 0 for x in r['coverage'].values()), 'coverage')
    rows = read(out/'selected_validation.json')
    require(rows == read(out/f'validation_epoch_{best:03d}.json') and r['selected'] == s3.s2.metrics(rows, validation), 'selected metrics')
    require(sha(out/'best.pt') == r['checkpoint_sha256'], 'checkpoint bytes')
    payload = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    require(set(payload) == {'identity', 'epoch', 'model_state', 'optimizer_state', 'algorithm_state'} and payload['identity'] == identity and payload['epoch'] == best, 'checkpoint binding')
    initial = model.state_dict(); actual = payload['model_state']
    require(set(actual) == set(initial), 'state keys')
    for name, v in initial.items():
        x = actual[name]
        require(isinstance(x, torch.Tensor) and x.shape == v.shape and x.dtype == v.dtype and torch.isfinite(x).all().item(), 'tensor '+name)
        if name in dict(model.named_buffers()) or (job['method'] not in method.FULL and name.startswith('encoder.')):
            require(torch.equal(x.cpu(), v.cpu()), 'protected buffer '+name)
    algorithm = payload['algorithm_state']
    if job['method'] == 'DB_MTL':
        active = {name: p for name, _, p in expected_steps(model, opt, tp, ap).values() if name.startswith('encoder.')}
        require(set(algorithm) == set(active), 'DB EMA coverage')
        for name, values in algorithm.items():
            require(type(values) is list and len(values) == 2 and all(isinstance(x, torch.Tensor) and
                    x.shape == active[name].shape and x.dtype == active[name].dtype and torch.isfinite(x).all().item() for x in values), 'DB EMA tensor')
    else:
        require(algorithm == {}, 'unexpected algorithm state')
    n = len(tp)//len(history)*best
    check_optimizer(model, opt, payload['optimizer_state'], history[best-1]['optimizer_steps'], tp[:n], ap[:n])
    model.load_state_dict(actual, strict=True)
    require(s3.state_dict_sha256(model) == r['selected_state_sha256'], 'selected state')
    replay, _ = s3.s2.evaluate(factory, trainer, setting, None, validation, 'cpu', best-1)
    require(len(replay) == len(rows), 'CPU replay population')
    for x, y in zip(replay, rows):
        require(set(x) == set(y) and all(x[k] == v for k, v in y.items() if k != 'prediction') and
                math.isclose(x['prediction'], y['prediction'], rel_tol=spec['prediction_replay_rtol'], abs_tol=spec['prediction_replay_atol']), 'CPU raw-data prediction replay')
    require(r['checkpoint_validation_replay'] is True and r['test_evaluated'] is False and r['scientific_acceptance'] == 'PENDING_REVIEW', 'scope')
    return r


def compare(results, reference):
    require(len(results) == len(jobs()) and {r['job']['id'] for r in results} == {j['id'] for j in jobs()}, '33 fixed results')
    values = {r['job']['id']: r for r in results}; records = []
    for r in results:
        j = r['job']; setting = j['setting']; score = r['selected']['macro_rmse']
        old = {m: reference[f'{setting}_{m}_s42']['selected'] for m in s3.s2.METHODS}
        best_method = min(old, key=lambda m: old[m]['macro_rmse']); strong = old[best_method]
        control = values[f'{setting}_TPO_FT_s42']['selected']
        records.append(dict(job=j, selected=r['selected'], best_epoch=r['best_epoch'], best_077_method=best_method,
            best_077=strong['macro_rmse'], delta_vs_best_077=score-strong['macro_rmse'],
            relative_vs_best_077=score/strong['macro_rmse']-1, delta_vs_paired_control=score-control['macro_rmse'],
            endpoint_deltas_vs_best_077={t: v['rmse']-strong['endpoints'][t]['rmse'] for t, v in r['selected']['endpoints'].items()},
            endpoint_deltas_vs_control={t: v['rmse']-control['endpoints'][t]['rmse'] for t, v in r['selected']['endpoints'].items()}))
    signals = []; architecture_signals = []
    for m in method.METHODS:
        subset = [r for r in records if r['job']['method'] == m]
        if all(r['delta_vs_best_077'] < 0 for r in subset):
            signals.append(m)
            if m == 'TPRS' and all(r['delta_vs_paired_control'] < 0 for r in subset):
                architecture_signals.append(m)
    return dict(comparisons=records, all_scene_candidate_signals=signals,
                private_architecture_candidate_signals=architecture_signals, unified_superiority_confirmed=False,
                scope='EXPLORATORY_SINGLE_SEED_VALIDATION_SCREEN')


def smoke(factory, trainer, source, setting, model_method, out, commit, device):
    require((setting, model_method) in SMOKE, 'smoke scope')
    s3.seed_everything(42, deterministic_algorithms=True)
    trainer.model, opt = method.build(trainer.model, source, model_method, .001, device)
    model = trainer.model; model.train(); before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tasks, counts = s3.s2.c0.tasks_and_counts(setting)
    plan = s3.s2.epoch_plan(tasks, counts, s3.s2.c0.BATCH_SIZES[setting], 0)
    schedule = s3.aux.Schedule(source.tasks, source.counts, s3.s2.c0.BATCH_SIZES[setting]); rows = []
    for item in plan[:2]:
        a = schedule.next(); batch = s3.s2.batch_for(factory, trainer, setting, 'train', item['task'], item['indices'], device)
        rows.append(dict(target=item, auxiliary=a, diagnostics=method.step(trainer, setting, batch, item['task'], 0, source, a, opt, device)))
    require(len(rows) == 2 and any(not torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items() if not k.startswith('decoders.')), 'smoke shared update')
    private_changed = any(not torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items() if k.startswith('private.'))
    require(private_changed == (model_method == 'TPRS') and all(r['diagnostics']['gradient_norms']['source'] > 0 for r in rows), 'smoke private/source update')
    write(out/'receipt.json', dict(task=TASK, commit=commit, setting=setting, method=model_method, source=source.identity,
          updates=2, rows=rows, private_changed=private_changed, scope='DISCARDED_REAL_SMOKE', content_status='PASS'))


def gate_check(root, commit, role):
    require(read(root/'gate.json') == dict(task=TASK, commit=commit, role=role, tests=list(TESTS), junit_sha256=sha(root/'tests.xml')), 'S5 gate identity')
    require(read(root/'command.json')['exit_code'] == 0, 'S5 gate exit')
    cases = ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases) == TEST_COUNT and all(not any(c.findall(t) for t in ('failure', 'error', 'skipped')) for c in cases)
            and {c.attrib['classname'] for c in cases} == {'tests.'+Path(t).stem for t in TESTS}, 'S5 complete gate')


def code_gate(root, commit, role):
    require(role in ('wsl', 'server'), 'S5 gate role'); s3.s2.c0.check_code(REPO, commit)
    root.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-m', 'pytest', *TESTS, '-q', '--basetemp', str(root/'pytest_tmp'), '--junitxml', str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as log:
        p = subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json', dict(argv=argv, exit_code=p.returncode)); require(p.returncode == 0, 'S5 tests failed')
    write(root/'gate.json', dict(task=TASK, commit=commit, role=role, tests=list(TESTS), junit_sha256=sha(root/'tests.xml')))
    gate_check(root, commit, role)


def claim_value(root, commit):
    return dict(task=TASK, commit=commit, output=str(root.resolve()), spec=SPEC, jobs=jobs(), smoke=[list(x) for x in SMOKE])


def check_launch(root, commit):
    claim = claim_value(root, commit)
    require(read(REPO/REGISTRY/'attempt.json') == claim, 'S5 one-shot claim')
    launch = read(root/'launch.json')
    require(launch['claim'] == claim and launch['task'] == TASK and launch['commit'] == commit, 'S5 launch')
    return launch


def sequence():
    return [('smoke', 'smoke_'+s+'_'+m, 'smoke_'+s+'_'+m) for s, m in SMOKE]+[('train', j['id'], j['id']) for j in jobs()]


def worker_command(mode, job, root, commit, split, source_lock):
    return [sys.executable, str(Path(__file__).resolve()), '_worker', '--mode', mode, '--job', job, '--commit', commit,
            '--output', str(root), '--split-manifest', str(split), '--source-lock', str(source_lock)]


def call_worker(argv, prefix, env):
    with prefix.with_suffix('.log').open('xb') as log:
        p = subprocess.run(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    write(prefix.with_suffix('.command.json'), dict(argv=argv, exit_code=p.returncode))
    require(p.returncode == 0, 'S5 worker failed; preserve and stop')


def verify(root, commit, split, source_lock):
    check_launch(root, commit)
    for role in ('wsl', 'server'):
        gate_check(root/(role+'_evidence'), commit, role)
    old = s3.references(root/'reference_077'); results = []; pids = []
    for setting, model_method in SMOKE:
        name = 'smoke_'+setting+'_'+model_method
        r = read(root/name/'receipt.json')
        require(read(root/(name+'.command.json'))['exit_code'] == 0 and r['task'] == TASK and r['commit'] == commit
                and r['setting'] == setting and r['method'] == model_method and r['updates'] == 2 and r['content_status'] == 'PASS'
                and r['private_changed'] == (model_method == 'TPRS') and len(r['rows']) == 2, 'real smoke receipt')
    # Deliberately no factory/context in this parent process, even across settings.
    for job in jobs():
        require(read(root/(job['id']+'.command.json'))['exit_code'] == 0, 'training worker exit')
        argv = worker_command('verify', job['id'], root, commit, split, source_lock)
        call_worker(argv, root/('verify_'+job['id']), dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        v = read(root/job['id']/'verification.json'); r = read(root/job['id']/'receipt.json')
        require(v['task'] == TASK and v['commit'] == commit and v['job'] == job and v['content_status'] == 'PASS'
                and v['receipt_sha256'] == sha(root/job['id']/'receipt.json') and v['raw_prediction_replayed'] is True
                and v['device'] == 'cpu' and v['new_optimizer_updates'] == 0 and type(v['pid']) is int, 'independent CPU verification')
        pids.append(v['pid']); results.append(r)
    require(sum(r['updates'] for r in results) == 13200, 'S5 total updates')
    return dict(task=TASK, commit=commit, content_status='PASS', scientific_acceptance='PENDING_REVIEW',
                total_epochs=1320, total_updates=13200, smoke_updates=22, verification_pids=pids, **compare(results, old))


def dispatch(items, gpus, execute):
    """At most one job per GPU; stop scheduling after failure, drain live jobs."""
    require(bool(gpus) and len(set(gpus)) == len(gpus) and set(gpus) <= {0, 1, 2, 3}, 'dispatch GPU scope')
    pending = iter(items); active = {}; failures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        def submit(gpu):
            item = next(pending, None)
            if item is not None:
                active[pool.submit(execute, item, gpu)] = gpu
        for gpu in gpus:
            submit(gpu)
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            idle = []
            for future in completed:
                idle.append(active.pop(future))
                try:
                    future.result()
                except BaseException as exc:
                    failures.append(exc)
            if not failures:
                for gpu in sorted(idle):
                    submit(gpu)
    if failures:
        raise RuntimeError('portfolio worker failure; no new jobs scheduled, in-flight jobs preserved') from failures[0]


def run(a):
    from p1d4_batch import free_gpus
    s3.s2.c0.check_code(REPO, a.commit)
    for role in ('wsl', 'server'):
        gate_check(getattr(a, role+'_evidence'), a.commit, role)
    s3.references(a.reference_root); root = a.output
    require(not (REPO/REGISTRY/'attempt.json').exists(), 'S5 attempt consumed')
    require(a.gpus and len(set(a.gpus)) == len(a.gpus) and set(a.gpus) <= {0, 1, 2, 3}, 'GPU allowlist')
    available = free_gpus(a.gpus)
    require(bool(available), 'no idle authorized GPU')
    root.mkdir(parents=True, exist_ok=False); (REPO/REGISTRY).mkdir(parents=True, exist_ok=True)
    claim = claim_value(root, a.commit); write(REPO/REGISTRY/'attempt.json', claim)
    uuids = {str(g): s3.s2.c0.gpu_uuid(g) for g in available}
    write(root/'launch.json', dict(task=TASK, commit=a.commit, claim=claim, gpus=available, gpu_uuids=uuids))
    try:
        for role in ('wsl', 'server'):
            shutil.copytree(getattr(a, role+'_evidence'), root/(role+'_evidence'), ignore=shutil.ignore_patterns('pytest_tmp'))
        s3.copy_references(a.reference_root, root/'reference_077')
        def execute(item, gpu):
            mode, key, name = item
            require(free_gpus([gpu]) == [gpu], 'GPU occupied')
            write(root/(name+'.assignment.json'), dict(task=TASK, commit=a.commit, mode=mode, job=key, gpu=gpu, gpu_uuid=uuids[str(gpu)]))
            argv = worker_command(mode, key, root, a.commit, a.split_manifest, a.source_lock)
            call_worker(argv, root/name, dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[str(gpu)], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                        CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        dispatch([x for x in sequence() if x[0] == 'smoke'], available, execute)
        dispatch([x for x in sequence() if x[0] == 'train'], available, execute)
        write(root/'verification.json', verify(root, a.commit, a.split_manifest, a.source_lock))
    except BaseException as exc:
        write(root/'failed.json', dict(task=TASK, error_type=type(exc).__name__, reason=str(exc))); raise


def worker(a):
    launch = check_launch(a.output, a.commit)
    if a.mode == 'verify':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only verification')
        torch.set_num_threads(1)
        job = next(j for j in jobs() if j['id'] == a.job)
        out = a.output/a.job
        require(not (out/'verification.json').exists(), 'verification already exists')
        f, t, source = s3.context(job['setting'], a.split_manifest, a.source_lock, 'cpu')
        ref = s3.references(a.output/'reference_077')[f'{job["setting"]}_B1_1E4_s42']
        verify_job(f, t, source, job, out, a.commit, ref)
        write(out/'verification.json', dict(task=TASK, commit=a.commit, job=job, pid=os.getpid(), device='cpu',
              content_status='PASS', raw_prediction_replayed=True, new_optimizer_updates=0, receipt_sha256=sha(out/'receipt.json')))
        return
    item = next((x for x in sequence() if x[:2] == (a.mode, a.job)), None)
    require(item is not None, 'fixed worker scope')
    assignment = read(a.output/(item[2]+'.assignment.json'))
    require(assignment == dict(task=TASK, commit=a.commit, mode=a.mode, job=a.job, gpu=assignment['gpu'],
            gpu_uuid=launch['gpu_uuids'][str(assignment['gpu'])]) and assignment['gpu'] in launch['gpus'], 'GPU assignment identity')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == assignment['gpu_uuid'] and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'single GPU binding')
    if a.mode == 'train':
        for _, _, name in [x for x in sequence() if x[0] == 'smoke']:
            r = read(a.output/name/'receipt.json')
            require(r['task'] == TASK and r['commit'] == a.commit and r['content_status'] == 'PASS'
                    and read(a.output/(name+'.command.json'))['exit_code'] == 0, 'real smoke incomplete')
    smoke_pair = next(((s, m) for s, m in SMOKE if a.job == 'smoke_'+s+'_'+m), None)
    setting = smoke_pair[0] if a.mode == 'smoke' else next(j for j in jobs() if j['id'] == a.job)['setting']
    f, t, source = s3.context(setting, a.split_manifest, a.source_lock, 'cuda:0')
    s3.population_check(f, source, setting, s3.s2.observations(f, t, setting, 'train'), s3.s2.observations(f, t, setting, 'validation'))
    out = a.output/item[2]; out.mkdir(exist_ok=False)
    write(out/'started.json', dict(task=TASK, commit=a.commit, mode=a.mode, job=a.job, pid=os.getpid()))
    if a.mode == 'smoke':
        smoke(f, t, source, setting, smoke_pair[1], out, a.commit, 'cuda:0')
    else:
        ref = s3.references(a.output/'reference_077')[f'{setting}_B1_1E4_s42']
        train_one(f, t, source, next(j for j in jobs() if j['id'] == a.job), out, a.commit, 'cuda:0', ref)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('code-gate', 'run', '_worker', 'verify', 'package'))
    p.add_argument('--commit', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--role', choices=('wsl', 'server')); p.add_argument('--gpus', type=int, nargs='+', choices=range(4))
    p.add_argument('--mode', choices=('smoke', 'train', 'verify')); p.add_argument('--job', choices=[x[1] for x in sequence()])
    for name in ('split-manifest', 'source-lock', 'reference-root', 'wsl-evidence', 'server-evidence'):
        p.add_argument('--'+name, type=Path)
    a = p.parse_args(); a.output = a.output.resolve()
    if a.action == 'code-gate':
        code_gate(a.output, a.commit, a.role)
    elif a.action == 'package':
        print(s3.s2.package(a.output, a.commit))
    else:
        s3.s2.c0.check_code(REPO, a.commit)
        require(a.split_manifest is not None and a.source_lock is not None, 'asset paths')
        if a.action == 'run':
            require(all(getattr(a, k) is not None for k in ('gpus', 'reference_root', 'wsl_evidence', 'server_evidence')), 'run arguments')
            run(a)
        elif a.action == 'verify':
            print(verify(a.output, a.commit, a.split_manifest, a.source_lock))
        else:
            worker(a)


if __name__ == '__main__':
    main()
