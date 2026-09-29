"""V9-C0: one-shot baseline cost calibration, never candidate selection.

Six jobs, seed 42, real train batches (Tox 64 / PubChem 32), 12 updates each. This
calibrates frozen/full Graphormer paths only; it does not implement V9's three
proposed architectures or justify their GPU-hour budget. Historical factories
validate source/data identity. No validation evaluation, test, checkpoints,
source training, retry, resume, or formal-run API is invoked.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parent
TASK = 'V9_C0_COST_CALIBRATION_20260929'
REGISTRY = '.tmp/v9_c0_attempts_20260929'
LOCK_SHA = 'd3502bba88feeaffa884f28dd1fbaa866aed465b111210ec1afee49841bd5d61'
SETTINGS = ('ToxAcute', 'A', 'B')
MODES = ('frozen', 'full')
UPDATES = 12
WARMUP = 2
BATCH_SIZES = {'ToxAcute': 64, 'A': 32, 'B': 32}
WALL_SECONDS = 1800
TEST_FILES = ('tests/test_v9_cost_probe.py', 'tests/test_p1d4_identity.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     allow_nan=False, separators=(',', ':')).encode('utf8')
    return hashlib.sha256(raw).hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def bad(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=bad)


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def jobs():
    return [dict(id=f'{s}_{m}_s42', setting=s, mode=m, seed=42)
            for s in SETTINGS for m in MODES]


def check_job(job):
    require(type(job) is dict and any(digest(job) == digest(j) for j in jobs()),
            'job outside six-job matrix')


def contract(setting):
    from p1d4_identity import LOCK, contract_for
    require(sha(LOCK) == LOCK_SHA, 'accepted identity lock changed')
    return contract_for(setting, 42)


def tasks_and_counts(setting):
    c = contract(setting)
    if setting == 'ToxAcute':
        from p1d_tox import TASKS
        return list(TASKS), c['counts']['train']
    scaler = c['scaler']['scaler']
    require(scaler['task_names'] == c['task_names'], 'scaler task order')
    return c['task_names'], dict(zip(scaler['task_names'], scaler['counts']))


def schedule(tasks, counts, batch_size):
    """Round-robin strata cover every endpoint; not a performance epoch.

    Within each task use disjoint chunks of a fixed permutation, cycling only
    when that task's train records run out. Small final batches remain small.
    """
    import numpy as np
    require(type(batch_size) is int and batch_size in (32, 64), 'accepted batch size')
    require(type(tasks) is list and len(tasks) in (3, 5) and len(set(tasks)) == len(tasks),
            'task set')
    require(set(tasks) == set(counts) and all(type(n) is int and n > 0 for n in counts.values()),
            'positive train counts')
    chunks = {}
    for i, task in enumerate(tasks):
        perm = np.random.default_rng(np.random.SeedSequence([42, i])).permutation(counts[task]).tolist()
        chunks[task] = [perm[k:k+batch_size] for k in range(0, len(perm), batch_size)]
    used = dict.fromkeys(tasks, 0)
    rows = []
    for step in range(UPDATES):
        task = tasks[step % len(tasks)]
        indices = chunks[task][used[task] % len(chunks[task])]
        used[task] += 1
        rows.append(dict(step=step, task=task, indices=indices, measured=step >= WARMUP))
    return rows


def summarize(rows, tasks, counts, batch_size):
    require(len(rows) == UPDATES, 'twelve completed updates required')
    expected = schedule(tasks, counts, batch_size)
    fields = {'step', 'task', 'indices', 'measured', 'n', 'sample_ids_sha256',
              'max_nodes', 'real_nodes', 'batch_seconds', 'update_seconds', 'loss'}
    for row, exp in zip(rows, expected):
        require(type(row) is dict and set(row) == fields, 'step fields')
        require(digest({k: row[k] for k in exp}) == digest(exp), 'batch schedule changed')
        require(type(row['n']) is int and row['n'] == len(exp['indices']), 'batch size')
        require(type(row['max_nodes']) is int and row['max_nodes'] > 0
                and type(row['real_nodes']) is int and 0 < row['real_nodes'] <= row['max_nodes'] * row['n'],
                'node counts')
        require(type(row['sample_ids_sha256']) is str and len(row['sample_ids_sha256']) == 64
                and set(row['sample_ids_sha256']) <= set('0123456789abcdef'), 'sample digest')
        for key in ('batch_seconds', 'update_seconds', 'loss'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]), 'finite measurement')
        require(row['batch_seconds'] >= 0 and row['update_seconds'] > 0, 'positive timing')
    per_task = {}
    for task in tasks:
        measured = [r for r in rows if r['task'] == task and r['measured']]
        require(measured, 'missing measured endpoint')
        times = [r['batch_seconds'] + r['update_seconds'] for r in measured]
        per_task[task] = dict(measured_batches=len(times), mean_step_seconds=statistics.mean(times),
                              max_observed_step_seconds=max(times),
                              formal_steps_per_epoch=math.ceil(counts[task] / batch_size))
    estimate = math.fsum(v['mean_step_seconds'] * v['formal_steps_per_epoch'] for v in per_task.values())
    observed = max(r['batch_seconds'] + r['update_seconds'] for r in rows)
    return dict(per_task=per_task, train_epoch_seconds_rough=estimate,
                train_only_40_epoch_hours_rough=40 * estimate / 3600,
                observed_max_step_envelope_40_epoch_hours=40 * observed * sum(
                    v['formal_steps_per_epoch'] for v in per_task.values()) / 3600,
                excludes=['validation', 'checkpoint_io', 'candidate_overhead', 'covariance_or_affinity',
                          'source_training', 'independent_confirmation'],
                is_upper_bound=False, authorizes_formal_training=False)


def configure_model(model, mode):
    import torch
    require(mode in MODES, 'cost mode')
    require(all(n.startswith(('encoder.backbone.', 'decoders.')) for n, _ in model.named_parameters()),
            'unreviewed model parameters')
    for p in model.encoder.parameters():
        p.requires_grad_(mode == 'full')
        p.grad = None
    for p in model.decoders.parameters():
        p.requires_grad_(True)
    # Dropout remains in train mode, matching the historical requires_grad-only
    # freeze path. This is not SRGT's deterministic source branch.
    model.train()
    groups = [dict(params=list(model.decoders.parameters()), lr=.001)]
    if mode == 'full':
        groups.insert(0, dict(params=list(model.encoder.parameters()), lr=.0001))
    return torch.optim.AdamW(groups, weight_decay=1e-5)


def checked_step(model, optimizer, loss, mode):
    import torch
    require(torch.isfinite(loss).all().item(), 'nonfinite loss')
    loss.backward()
    if mode == 'frozen':
        require(all(p.grad is None for p in model.encoder.parameters()), 'frozen gradient leak')
    else:
        require(any(p.grad is not None for p in model.encoder.parameters()), 'missing backbone gradients')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                  1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def claim(repo, root, commit):
    registry = Path(repo) / REGISTRY
    registry.mkdir(parents=True, exist_ok=True)
    # A different output folder or a crashed process cannot reset the budget.
    write(registry / 'attempt.json', dict(task=TASK, commit=commit, root=str(Path(root).resolve()),
                                        max_updates=72, max_wall_seconds=WALL_SECONDS))


def check_code(repo, commit):
    from p1d4_batch import check_commit
    return check_commit(repo, commit)


def check_gate(path, commit, role):
    require(role in ('wsl', 'server'), 'code gate role')
    root = Path(path)
    receipt = read(root / 'gate.json')
    require(receipt == dict(task=TASK, role=role, commit=commit, tests=list(TEST_FILES),
                            junit_sha256=sha(root / 'tests.xml'), real_data_optimizer_updates=0), 'code gate receipt identity')
    cases = ET.parse(root / 'tests.xml').findall('.//testcase')
    require(cases and all(not any(c.findall(t) for t in ('failure', 'error', 'skipped')) for c in cases),
            'code tests failure/error/skip')
    require({c.attrib['classname'] for c in cases} == {'tests.' + Path(t).stem for t in TEST_FILES},
            'code test files incomplete')
    return receipt


def code_gate(root, commit, role):
    require(role in ('wsl', 'server'), 'explicit code gate role required')
    check_code(REPO, commit)
    root.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-m', 'pytest', *TEST_FILES, '-q',
            '--basetemp', str(root / 'pytest_tmp'), '--junitxml', str(root / 'tests.xml')]
    with (root / 'tests.log').open('xb') as log:
        result = subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    write(root / 'command.json', dict(argv=argv, exit_code=result.returncode))
    require(result.returncode == 0, 'code gate failed')
    write(root / 'gate.json', dict(task=TASK, role=role, commit=commit, tests=list(TEST_FILES),
                                 junit_sha256=sha(root / 'tests.xml'), real_data_optimizer_updates=0))
    check_gate(root, commit, role)


def worker(root, job, commit, split, source):
    check_job(job)
    check_code(REPO, commit)
    launch = read(root / 'launch.json')
    require(launch['commit'] == commit and launch['task'] == TASK and launch['jobs'] == jobs(), 'launch identity')
    require(read(REPO / REGISTRY / 'attempt.json')['root'] == str(root.resolve()), 'attempt root')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == launch['gpu_uuid'], 'GPU binding')
    out = root / job['id']
    out.mkdir(exist_ok=False)
    write(out / 'started.json', dict(job=job, commit=commit))
    import torch
    from p1d4_runtime import factory_for
    from reproducibility import state_dict_sha256
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one visible CUDA GPU required')
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    factory = factory_for(REPO, setting=job['setting'], seed=42, split_manifest=split,
                          source_lock=source, device='cuda:0')
    trainer = (factory.make_trainer() if job['setting'] == 'ToxAcute'
               else factory.make_trainer(original=True, device='cuda:0'))
    tasks, counts = tasks_and_counts(job['setting'])
    datasets = factory.datasets['train'] if job['setting'] == 'ToxAcute' else trainer.datasets['train']
    require({t: len(datasets[t]) for t in tasks} == counts, 'trusted train population')
    model = trainer.model
    initial_encoder = state_dict_sha256(model.encoder)
    optimizer = configure_model(model, job['mode'])
    trainer.optimizer = optimizer
    preparation_seconds = time.perf_counter() - start
    rows = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for item in schedule(tasks, counts, BATCH_SIZES[job['setting']]):
        task, indices = item['task'], item['indices']
        expected_ids = [str(datasets[task].get_sample_id(i)) for i in indices]
        tick = time.perf_counter()
        if job['setting'] == 'ToxAcute':
            batch = factory.collator([datasets[task][i] for i in indices]).to('cuda:0')
        else:
            batch = trainer._batch('train', task, indices)
        torch.cuda.synchronize()
        batch_seconds = time.perf_counter() - tick
        require(not batch.is_empty and batch.y.numel() == len(indices)
                and [str(x) for x in batch.sample_id] == expected_ids, 'dropped/wrong train batch')
        # The watermark is written before every possible optimizer update.
        write(out / f'update_{item["step"]:02d}.intent.json', dict(step=item['step'], max_total=item['step']+1))
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if job['setting'] == 'ToxAcute':
            loss, _ = trainer._training_step(batch, task, 0)
        else:
            from loss import QuantileRegressionLoss
            scaler = trainer.scalers[task]
            loss = QuantileRegressionLoss().compute_loss(model(batch, task_name=task)[task],
                (batch.y.reshape(-1, 1) - scaler['mean']) / scaler['std'])
        loss_value = checked_step(model, optimizer, loss, job['mode'])
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - tick
        row = dict(item, n=len(indices), sample_ids_sha256=digest(expected_ids), max_nodes=int(batch.x.shape[1]),
                   real_nodes=int(batch.node_mask.sum()), batch_seconds=batch_seconds,
                   update_seconds=update_seconds, loss=loss_value)
        write(out / f'step_{item["step"]:02d}.json', row)
        rows.append(row)
    require(all(torch.isfinite(v).all().item() for v in model.state_dict().values()), 'nonfinite final model')
    final_encoder = state_dict_sha256(model.encoder)
    require((initial_encoder == final_encoder) == (job['mode'] == 'frozen'), 'backbone mutation scope')
    receipt = dict(schema='v9_cost_job_v1', task=TASK, job=job, commit=commit,
        contract_sha256=digest(contract(job['setting'])), counts=counts, rows=rows,
        summary=summarize(rows, tasks, counts, BATCH_SIZES[job['setting']]), preparation_seconds=preparation_seconds,
        initial_encoder=initial_encoder, final_encoder=final_encoder,
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        environment=dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                         device_name=torch.cuda.get_device_name(0), visible_device=launch['gpu_uuid']),
        scope='TRAIN_COST_ONLY_NO_CANDIDATES', acceptance='PENDING_REVIEW',
        holdout_predictions_accessed=False, validation_evaluated=False, scientific_pass=False)
    write(out / 'receipt.json', receipt)


def verify(root, commit):
    launch = read(root / 'launch.json')
    require(launch['task'] == TASK and launch['commit'] == commit and launch['jobs'] == jobs(), 'suite identity')
    result = {}
    for job in jobs():
        out = root / job['id']
        r = read(out / 'receipt.json')
        tasks, counts = tasks_and_counts(job['setting'])
        require(r['schema'] == 'v9_cost_job_v1' and r['task'] == TASK and r['job'] == job
                and r['commit'] == commit and r['contract_sha256'] == digest(contract(job['setting'])), 'receipt identity')
        require(r['counts'] == counts and r['holdout_predictions_accessed'] is False
                and r['validation_evaluated'] is False and r['scientific_pass'] is False
                and r['scope'] == 'TRAIN_COST_ONLY_NO_CANDIDATES' and r['acceptance'] == 'PENDING_REVIEW', 'receipt scope')
        require(sorted(p.name for p in out.glob('step_*.json')) == [f'step_{i:02d}.json' for i in range(UPDATES)], 'step file matrix')
        rows = [read(out / f'step_{i:02d}.json') for i in range(UPDATES)]
        require(digest(rows) == digest(r['rows']), 'step evidence changed')
        for i in range(UPDATES):
            require(read(out / f'update_{i:02d}.intent.json') == dict(step=i, max_total=i+1), 'update watermark')
        summary = summarize(rows, tasks, counts, BATCH_SIZES[job['setting']])
        require(digest(summary) == digest(r['summary']), 'summary not independently reproduced')
        require((r['initial_encoder'] == r['final_encoder']) == (job['mode'] == 'frozen'), 'backbone digest scope')
        require(type(r['peak_allocated_bytes']) is int and 0 < r['peak_allocated_bytes'] <= r['peak_reserved_bytes'], 'GPU memory')
        result[job['id']] = dict(receipt_sha256=sha(out / 'receipt.json'), summary=summary)
    return dict(task=TASK, commit=commit, completed_jobs=6, optimizer_updates=72, results=result,
                content_status='PASS', scientific_acceptance='NOT_ASSESSED', formal_training_authorized=False)


def gpu_uuid(index):
    value = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=uuid',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    require(value.startswith('GPU-') and '\n' not in value and len(value) > 12, 'physical GPU UUID')
    return value


def run(root, commit, gpu, split, source, wsl, server):
    from p1d4_batch import free_gpus
    import shutil
    check_code(REPO, commit)
    check_gate(wsl, commit, 'wsl')
    check_gate(server, commit, 'server')
    require(free_gpus([gpu]) == [gpu], 'selected GPU busy')
    contract('ToxAcute')
    root.mkdir(parents=True, exist_ok=False)
    claim(REPO, root, commit)
    uuid = gpu_uuid(gpu)
    write(root / 'launch.json', dict(task=TASK, commit=commit, jobs=jobs(), gpu=gpu, gpu_uuid=uuid,
                                   max_optimizer_updates=72, max_wall_seconds=WALL_SECONDS))
    shutil.copytree(wsl, root / 'wsl_evidence', ignore=shutil.ignore_patterns('pytest_tmp'))
    shutil.copytree(server, root / 'server_evidence', ignore=shutil.ignore_patterns('pytest_tmp'))
    deadline = time.monotonic() + WALL_SECONDS
    try:
        for job in jobs():
            require(free_gpus([gpu]) == [gpu], 'GPU occupied before next job')
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'suite wall limit reached')
            argv = [sys.executable, str(Path(__file__).resolve()), '_worker', '--output', str(root),
                    '--commit', commit, '--job', job['id'], '--split-manifest', str(split), '--source-lock', str(source)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root / (job['id'] + '.log')).open('xb') as log:
                child = subprocess.run(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            write(root / (job['id'] + '.command.json'), dict(argv=argv, exit_code=child.returncode))
            require(child.returncode == 0, 'job failed; no retry or later job')
        value = verify(root, commit)
        value['elapsed_wall_seconds'] = WALL_SECONDS - (deadline - time.monotonic())
        write(root / 'verification.json', value)
    except BaseException as exc:
        write(root / 'failed.json', dict(task=TASK, error_type=type(exc).__name__, reason=str(exc),
                                        formal_training_authorized=False))
        raise


def package(root):
    """Include partial failures as well as successful probes; no model weights."""
    import zipfile
    require(root.is_dir() and (root / 'launch.json').is_file(), 'probe output required')
    archive = root.with_name(root.name + '.zip')
    sidecar = Path(str(archive) + '.sha256')
    require(not archive.exists() and not sidecar.exists(), 'existing package')
    files = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'symlink in evidence')
        if not path.is_file():
            continue
        require(path.suffix in ('.json', '.log', '.xml'), 'unexpected evidence file type')
        files[path.relative_to(root).as_posix()] = sha(path)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(root / name, name)
        z.writestr('checksums.sha256', ''.join(f'{value}  {name}\n' for name, value in files.items()))
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'package CRC')
        require(set(z.namelist()) == set(files) | {'checksums.sha256'}, 'package members')
        require(all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in files.items()), 'package content hash')
    with sidecar.open('x', encoding='utf8') as f:
        f.write(f'{sha(archive)}  {archive.name}\n')
    return dict(archive=str(archive), sha256=sha(archive), scientific_acceptance='NOT_ASSESSED')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['code-gate', 'run', 'verify', 'package', '_worker'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--source-lock', type=Path)
    p.add_argument('--wsl-evidence', type=Path)
    p.add_argument('--server-evidence', type=Path)
    p.add_argument('--role', choices=['wsl', 'server'])
    p.add_argument('--gpu', type=int, choices=range(4))
    p.add_argument('--job', choices=[j['id'] for j in jobs()])
    a = p.parse_args()
    root = a.output.resolve()
    if a.action == 'code-gate':
        code_gate(root, a.commit, a.role)
    elif a.action == 'verify':
        print(json.dumps(verify(root, a.commit), ensure_ascii=False, indent=2))
    elif a.action == 'package':
        require(read(root / 'launch.json')['commit'] == a.commit, 'package commit identity')
        print(json.dumps(package(root), ensure_ascii=False, indent=2))
    else:
        require(a.split_manifest is not None and a.source_lock is not None, 'explicit asset paths required')
        if a.action == 'run':
            require(a.gpu is not None and a.wsl_evidence is not None and a.server_evidence is not None,
                    'GPU and separate WSL/server evidence required')
            run(root, a.commit, a.gpu, a.split_manifest.resolve(), a.source_lock.resolve(),
                a.wsl_evidence.resolve(), a.server_evidence.resolve())
        else:
            require(a.job is not None, 'worker job required')
            worker(root, next(j for j in jobs() if j['id'] == a.job), a.commit,
                   a.split_manifest.resolve(), a.source_lock.resolve())


if __name__ == '__main__':
    main()
