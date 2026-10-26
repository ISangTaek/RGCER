"""Resolve and execute only the approved F01 18-run batch.

The release carries concrete views/configurations. The existing accepted graph
cache is reused with registry/per-record checks, without a whole-database scan.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .artifacts import load_arrays, load_manifest, read_json, write_json
from .cache import GraphCache
from .chemistry import digest
from .config import git_identity, resolve_run
from .evaluation import sealed_predict, score_prediction
from .formal_data import PROTOCOL, read_tsv
from .models import build_model, parameter_counts
from .preparation import write_tsv
from .training import ModelInputs, run_training, seed_everything


def resolve_release(matrix, prepared, destination, commit):
    prepared, destination = Path(prepared), Path(destination)
    rows = read_tsv(matrix)
    expected = {(m, b, s) for m in ('M3', 'N1', 'N1S') for b in ('FULL', 'S21_P10') for s in (42,43,44)}
    if len(rows) != 18 or {(r['method'], r['label_budget'], int(r['seed'])) for r in rows} != expected:
        raise ValueError('F01 must contain exactly the approved 18 unique units')
    groups = read_json(prepared/'endpoint_groups.json')
    destination.mkdir(parents=True, exist_ok=False)
    released = []
    for row in rows:
        if (row['protocol_id'], row['split'], row['outer_fold'], row['label_subset_seed']) != (PROTOCOL, 'S', '0', '1042'):
            raise ValueError('F01 protocol, fold or subset changed')
        relative_view = f'views/S0/{row["label_budget"]}'
        manifest = load_manifest(prepared/relative_view)
        run = resolve_run(dict(run_id=row['run_id'], method_id=row['method'],
            config_id='C0' if row['method'] == 'M3' else 'C2', init_seed=int(row['seed']),
            fold='S0', phase='P3', state='READY', code_commit=commit, protocol_id=PROTOCOL,
            batch='F01', experiment=row['experiment'], label_budget=row['label_budget'],
            label_subset_seed=1042, view_relative_path=relative_view), view_manifest=manifest)
        run['config']['selection_tasks'] = None
        if row['method'] in ('N1', 'N1S') and row['label_budget'] == 'S21_P10':
            run['config']['auxiliary_selection_tasks'] = groups['S21']
        run['config_sha256'] = digest(run['config'])
        run['run_sha256'] = digest({k:v for k,v in run.items() if k != 'run_sha256'})
        write_json(destination/'runs'/f'{run["run_id"]}.json', run)
        released.append({**row, 'status':'READY', 'split_materialized':True, 'code_commit':commit,
                         'view_id':manifest['view_id'], 'run_sha256':run['run_sha256']})
    write_tsv(destination/'run_matrix.tsv', released, list(released[0]))
    write_json(destination/'release.json', dict(protocol_id=PROTOCOL, batch='F01', state='READY',
        code_commit=commit, run_ids=[r['run_id'] for r in rows], new_fits=18,
        checkpoint_selection='all59 validation macro-RMSE; auxiliary S21 never selects main results',
        graph_cache_verification='accepted registry + consumed-payload checks; no whole database rehash',
        endpoint_groups=groups))
    return released


def preflight(release, cache_root, output, device):
    release = Path(release); meta = read_json(release/'release.json')
    if git_identity() != meta['code_commit']:
        raise ValueError('Checkout differs from released commit')
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Server preflight requires an available CUDA device')
    results = []
    runs = [read_json(release/'runs'/f'{name}.json') for name in meta['run_ids']]
    for method in ('M3', 'N1', 'N1S'):
        run = next(r for r in runs if (r['method_id'], r['label_budget'], r['init_seed']) == (method, 'S21_P10', 42))
        root = release/run['view_relative_path']; manifest = load_manifest(root)
        train = load_arrays(root, 'train.npz', manifest); stats = load_arrays(root, 'statistics.npz', manifest)
        cache = GraphCache(cache_root, source_sha256=manifest['source_sha256'], verify_database=False)
        if cache.manifest['cache_id'] != manifest['cache_id']:
            raise ValueError('Accepted graph cache identity differs')
        seed_everything(42)
        model = build_model(method, manifest['task_names'], run['config'], adjacency=stats['adjacency'],
                            counts=stats['count'], device=device, seed=42).eval()
        indices = np.flatnonzero(np.isfinite(train['labels']).any(1))[:16].tolist()
        with torch.no_grad():
            prediction = model(ModelInputs(cache, train, method, device).batch(indices))
        if prediction.shape != (len(indices), 59) or not torch.isfinite(prediction).all():
            raise ValueError('Nonfinite or misaligned 59-endpoint forward')
        results.append(dict(method=method, input_records=len(indices), output_shape=list(prediction.shape),
            parameters=parameter_counts(model), finite=True))
        del model
    report = dict(status='SERVER_REAL_CACHE_FORWARD_PASS', code_commit=git_identity(),
        device=device, gpu=torch.cuda.get_device_name(torch.device(device)), torch=str(torch.__version__),
        optimizer_steps=0, results=results)
    write_json(output, report)
    return report


def execute_run(release, run_id, cache_root, output, device, resume=False):
    release, output = Path(release), Path(output)
    meta = read_json(release/'release.json')
    if run_id not in meta['run_ids'] or git_identity() != meta['code_commit']:
        raise ValueError('Run or checkout is outside the released F01 batch')
    run_file = release/'runs'/f'{run_id}.json'; run = read_json(run_file)
    view = release/run['view_relative_path']; run_root = output/run_id
    training = run_root/'training'; run_root.mkdir(parents=True, exist_ok=resume)
    summary_file = training/'summary.json'
    summary = read_json(summary_file) if summary_file.is_file() else None
    if summary is None or summary['terminal'] != 'COMPLETE':
        run_training(run_file, view, cache_root, training, device=device,
                     resume=resume and (training/'last.pt').exists(), verify_database=False)
        summary = read_json(summary_file)
    if summary['terminal'] != 'COMPLETE' or summary['identity']['run_sha256'] != run['run_sha256']:
        raise ValueError('Training did not complete under the released run identity')
    for role in ('validation', 'test'):
        predictions = run_root/role
        # A partial prediction directory is retained; recovery requires inspecting
        # that directory, not deleting or overwriting it automatically.
        if not (predictions/'manifest.json').is_file():
            sealed_predict(training/'best.pt', view, cache_root, predictions, role=role,
                           device=device, verify_database=False)
        score_path = run_root/f'{role}_metrics.json'
        if not score_path.exists():
            score_prediction(predictions, view, score_path, expected_checkpoint_sha256=summary['best_sha256'])
    write_json(run_root/'complete.json', dict(run_id=run_id, status='COMPLETE', code_commit=git_identity(),
        main_checkpoint='training/best.pt', best_epoch=summary['best_epoch'],
        auxiliary_checkpoint='training/best_s21.pt' if 'auxiliary_selection' in summary else None,
        device=device, new_fit=1))


def batch(release, cache_root, output, gpus, resume=False):
    """One independent subprocess per selected, user-checked free physical GPU."""
    release, output = Path(release).resolve(), Path(output).resolve()
    if not gpus or len(set(gpus)) != len(gpus) or set(gpus)-{0,1,2,3}:
        raise ValueError('Choose distinct authorized physical GPUs 0-3')
    meta = read_json(release/'release.json'); output.mkdir(parents=True, exist_ok=resume)
    pending = list(meta['run_ids']); active = {}; completed = []; failed = []
    logs = output/'logs'; logs.mkdir(exist_ok=True)
    try:
        while pending or active:
            for gpu in gpus:
                if failed or not pending or gpu in active:
                    continue
                run_id = pending.pop(0)
                if resume and (output/run_id/'complete.json').exists():
                    done = read_json(output/run_id/'complete.json')
                    if done['code_commit'] != meta['code_commit']:
                        raise ValueError('Completed run belongs to another commit')
                    completed.append(run_id); continue
                command = [sys.executable, '-B', '-m', 'pgt_v2.formal_batch', '--threads', '1', 'run',
                    '--release', str(release), '--cache', str(Path(cache_root).resolve()), '--output', str(output),
                    '--run-id', run_id, '--device', 'cuda:0'] + (['--resume'] if resume else [])
                log = (logs/f'{run_id}.log').open('a' if resume else 'x', encoding='utf-8')
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(gpu), 'PYTHONUNBUFFERED':'1',
                       'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (process, log, run_id)
                print(f'START GPU{gpu} {run_id}', flush=True)
            for gpu, (process, log, run_id) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close(); del active[gpu]
                    (completed if code == 0 else failed).append(run_id)
                    print(f'EXIT GPU{gpu} {run_id} code={code}', flush=True)
            write_json(output/'batch_status.json', dict(code_commit=meta['code_commit'], completed=completed,
                failed=failed, pending=pending, running=[v[2] for v in active.values()]))
            if failed and not active:
                break
            if pending or active:
                time.sleep(2)
    finally:
        # Do not kill a running fit on scheduler interruption. Report live PIDs;
        # the operator must check them before resuming the same output directory.
        for gpu, (process, log, run_id) in active.items():
            print(f'RUNNING_ON_EXIT GPU{gpu} pid={process.pid} {run_id}', flush=True)
            log.close()
    if failed:
        raise RuntimeError('F01 run failed; remaining units not started: '+', '.join(failed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1)
    commands = parser.add_subparsers(dest='command', required=True)
    resolve = commands.add_parser('resolve')
    for key in ('matrix', 'prepared', 'output', 'commit'):
        resolve.add_argument('--'+key, required=True)
    check = commands.add_parser('preflight')
    single = commands.add_parser('run'); multi = commands.add_parser('batch')
    for cmd in (check, single, multi):
        for key in ('release', 'cache', 'output'):
            cmd.add_argument('--'+key, required=True)
    for cmd in (check, single):
        cmd.add_argument('--device', default='cuda:0')
    single.add_argument('--run-id', required=True)
    for cmd in (single, multi):
        cmd.add_argument('--resume', action='store_true')
    multi.add_argument('--gpus', nargs='+', type=int, required=True)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    if args.command == 'resolve':
        resolve_release(args.matrix, args.prepared, args.output, args.commit)
    elif args.command == 'preflight':
        preflight(args.release, args.cache, args.output, args.device)
    elif args.command == 'run':
        execute_run(args.release, args.run_id, args.cache, args.output, args.device, args.resume)
    else:
        batch(args.release, args.cache, args.output, args.gpus, args.resume)
