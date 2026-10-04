"""D2-S1 CLI: separate WSL/server gates, 27 jobs, independent replay and package."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile

from v9_cost_probe import read, write, sha, digest, require
import d2_control as core
from d2_runner import TASK

REPO = Path(__file__).resolve().parent
TESTS = ('tests/test_d2_control.py', 'tests/test_d2_data.py', 'tests/test_d2_runner.py',
         'tests/test_d2_screen.py', 'tests/test_v9_s3_screen.py', 'tests/test_p1d4_identity.py')
REGISTRY = '.tmp/d2_s1_attempt_20261004'


def check_code(commit):
    from p1d4_batch import check_commit
    return check_commit(REPO, commit)


def gate_check(root, commit, role):
    root = Path(root)
    r = read(root/'gate.json')
    require(r['task'] == TASK and r['commit'] == commit and r['role'] == role and
            r['tests'] == list(TESTS) and r['junit_sha256'] == sha(root/'tests.xml') and
            r['exit_code'] == 0 and r['status'] == 'PASS', 'code gate identity')
    cases = ET.parse(root/'tests.xml').getroot().findall('.//testcase')
    require(cases and len(cases) == r['test_count'] and
            {c.attrib['classname'] for c in cases} == {'tests.'+Path(t).stem for t in TESTS}, 'complete code gate')
    require(all(not any(c.find(k) is not None for k in ('failure', 'error', 'skipped')) for c in cases), 'no failed/skipped tests')
    # Collection itself is recomputed from this commit. A tiny fabricated report
    # with the right class names is not sufficient.
    require(len(cases) == EXPECTED_TEST_COUNT, 'frozen code gate case count')
    require(digest(sorted(c.attrib['classname']+'::'+c.attrib['name'] for c in cases)) == EXPECTED_TEST_IDS,
            'frozen code gate case identities')
    return r


# Frozen JUnit case inventory; both hosts must match, with zero skips.
EXPECTED_TEST_COUNT = 142
EXPECTED_TEST_IDS = 'c5358403710e65cbda2ac40d85bde84dddbe6ab134b9e18635cf4b8c55a89597'


def code_gate(root, commit, role):
    check_code(commit)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *TESTS,
            '--junitxml='+str(root/'tests.xml'), '--basetemp='+str(root/'pytest_tmp')]
    with (root/'tests.log').open('x', encoding='utf8') as log:
        result = subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    cases = ET.parse(root/'tests.xml').getroot().findall('.//testcase')
    r = dict(task=TASK, role=role, commit=commit, hostname=socket.gethostname(), cwd=str(REPO),
             python=sys.executable, python_version=sys.version, argv=argv, exit_code=result.returncode,
             tests=list(TESTS), test_count=len(cases), junit_sha256=sha(root/'tests.xml'),
             status='PASS' if result.returncode == 0 else 'FAIL')
    write(root/'gate.json', r)
    gate_check(root, commit, role)
    return r


def job_for(name):
    found = [j for j in core.jobs() if j['id'] == name]
    require(len(found) == 1, 'unknown D2-S1 job')
    return found[0]


def worker(args, verify=False):
    check_code(args.commit)
    job = job_for(args.job)
    from d2_data import load
    from d2_runner import train_one, verify_one
    if not verify:
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one real CUDA GPU required')
    data = load(REPO, job['setting'], job['seed'], job['condition'], args.split_manifest, args.source_lock)
    if verify:
        result = verify_one(data, job, args.output, args.commit, 'cpu')
        write(Path(args.output)/'verification.json', result)
    else:
        train_one(data, job, args.output, args.commit, 'cuda:0')


def execute_child(job, root, commit, gpu, split_manifest, source_lock):
    from v9_cost_probe import gpu_uuid
    uuid = gpu_uuid(gpu)
    output = root/'runs'/job['id']
    argv = [sys.executable, '-u', str(REPO/'d2_screen.py'), 'worker', '--commit', commit,
            '--job', job['id'], '--output', str(output), '--split-manifest', str(split_manifest),
            '--source-lock', str(source_lock)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    write(root/'commands'/(job['id']+'.json'), dict(argv=argv, commit=commit, job=job,
          hostname=socket.gethostname(), physical_gpu=gpu, gpu_uuid=uuid, cwd=str(REPO)))
    with (root/'logs'/(job['id']+'.log')).open('x', encoding='utf8') as log:
        run = subprocess.run(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    write(root/'commands'/(job['id']+'.exit.json'), dict(exit_code=run.returncode))
    require(run.returncode == 0, 'training child failed: '+job['id'])
    verify_argv = argv.copy()
    verify_argv[3] = 'verify-worker'
    env['CUDA_VISIBLE_DEVICES'] = ''
    with (root/'logs'/(job['id']+'.verify.log')).open('x', encoding='utf8') as log:
        replay = subprocess.run(verify_argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    write(root/'commands'/(job['id']+'.verify.json'), dict(argv=verify_argv, exit_code=replay.returncode,
                                                        device='cpu', fresh_process=True))
    require(replay.returncode == 0, 'independent replay failed: '+job['id'])
    checked = read(output/'verification.json')
    require(checked['status'] == 'CONTENT_PASS' and checked['job'] == job and
            checked['training_sha256'] == sha(output/'training.json'), 'child verification identity')
    return read(output/'training.json')


def pairing_check(results):
    for scene in core.SETTINGS:
        group = [r for r in results if r['job']['setting'] == scene]
        require(len(group) == 9, 'nine scene controls')
        require(len({r['identity']['initial_heads'] for r in group}) == 1, 'all-condition identical fresh heads')
        require(len({r['schedule_sha256'] for r in group}) == 1 and
                len({digest(r['exposure']['target']) for r in group}) == 1, 'matched target exposure and schedule')
        for labels in [('FJ', 'PT', 'PJ'), ('RT', 'RJ')]:
            subset = [r for r in group if r['job']['condition'] in labels]
            require(len({r['identity']['initial_encoder'] for r in subset}) == 1, 'paired encoder initialization')
        require(len({r['identity']['initial_encoder'] for r in group}) == 2, 'distinct source/random initialization')


def preflight_inputs(split_manifest, source_lock):
    """All three asset/population checks finish before any GPU job starts."""
    from d2_data import load
    rows = []
    expected = {'ToxAcute': (3, 262, 39, 56, 78063), 'A': (5, 407, 133, 104, 75872),
                'B': (5, 261, 83, 56, 78063)}
    for setting in core.SETTINGS:
        data = load(REPO, setting, 42, 'FJ', split_manifest, source_lock)
        actual = (len(data.target_tasks), sum(data.target_counts.values()), len(data.rows['validation']),
                  len(data.source_tasks), sum(data.source_counts.values()))
        require(actual == expected[setting], 'frozen complete population: '+setting)
        rows.append(dict(setting=setting, identity=data.identity,
                         updates_per_trajectory=core.update_count(data.source_counts),
                         validation_steps=core.evaluation_steps(data.source_counts)))
    return rows


def run(args):
    from p1d4_batch import free_gpus
    check_code(args.commit)
    gate_check(args.wsl_gate, args.commit, 'wsl')
    gate_check(args.server_gate, args.commit, 'server')
    allowed = [int(v) for v in args.gpus.split(',')]
    available = free_gpus(allowed)
    require(available, 'no allowed idle GPU; no work started')
    root = Path(args.output).resolve()
    require(not root.exists(), 'output exists')
    require(not (REPO/REGISTRY).exists(), 'D2-S1 attempt already claimed; no automatic retry')
    preflight = preflight_inputs(args.split_manifest, args.source_lock)
    registry = REPO/REGISTRY
    registry.mkdir(parents=True, exist_ok=False)
    write(registry/'claim.json', dict(task=TASK, output=str(root), commit=args.commit, utc=datetime.now(timezone.utc).isoformat()))
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('runs', 'commands', 'logs', 'gates'):
        (root/folder).mkdir()
    write(root/'preflight.json', preflight)
    for role, path in [('wsl', args.wsl_gate), ('server', args.server_gate)]:
        (root/'gates'/role).mkdir()
        for name in ('gate.json', 'tests.xml', 'tests.log'):
            shutil.copyfile(Path(path)/name, root/'gates'/role/name)
    write(root/'plan.json', dict(task=TASK, commit=args.commit, jobs=core.jobs(), spec=core.SPEC,
                               gpu_slots=allowed, maximum_concurrency=4, test_access=False,
                               source_lock_sha256=sha(args.source_lock), split_sha256=sha(args.split_manifest)))
    queue, active, results, failures = list(core.jobs()), {}, [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        while queue or active:
            if not failures and queue:
                used = {v[0] for v in active.values()}
                for gpu in [g for g in free_gpus(allowed) if g not in used]:
                    if not queue:
                        break
                    j = queue.pop(0)
                    f = pool.submit(execute_child, j, root, args.commit, gpu, args.split_manifest, args.source_lock)
                    active[f] = (gpu, j)
                if queue and not active:
                    failures.append(dict(error='No idle permitted GPU; remaining queue stopped without waiting indefinitely'))
            for future in list(active):
                if future.done():
                    _, j = active.pop(future)
                    try:
                        results.append(future.result())
                    except Exception:
                        failures.append(dict(job=j, error=traceback.format_exc()))
            if failures:
                queue = []  # Running children finish; do not discard completed facts.
            if active:
                time.sleep(2)
    if failures:
        write(root/'completion.json', dict(task=TASK, status='FAILED_PARTIAL', failures=failures,
                                          completed=[r['job']['id'] for r in results], scientific_acceptance=False))
        raise ValueError('D2-S1 partial failure; preserve and package output; no automatic retry')
    pairing_check(results)
    write(root/'paired_comparison.json', core.paired_report(results))
    write(root/'completion.json', dict(task=TASK, status='CONTENT_PASS_PENDING_SCIENTIFIC_REVIEW', jobs=len(results),
                                      scientific_acceptance=False, test_accessed=False, next_stage_unlocked=False))


def package(root, output):
    """Facts package only. Tensor checkpoints remain on server with SHA receipts."""
    root, output = Path(root).resolve(), Path(output).resolve()
    require(root.is_dir() and not output.exists() and not output.is_relative_to(root), 'new external ZIP path')
    allowed = {'.json', '.log', '.xml'}
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in allowed)
    require(files and all(not p.is_symlink() and p.resolve().is_relative_to(root) for p in files), 'package paths')
    manifest = {p.relative_to(root).as_posix(): sha(p) for p in files}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(root).as_posix())
        z.writestr('transport_manifest.json', json.dumps(dict(files=manifest, tensor_checkpoints_included=False), indent=2))
    with zipfile.ZipFile(output) as z:
        require(z.testzip() is None, 'ZIP CRC')
    return dict(zip=str(output), sha256=sha(output), files=len(files), checkpoints_retained_on_server=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('code-gate', 'worker', 'verify-worker', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--commit', required=True)
        p.add_argument('--output', type=Path, required=True)
        if name == 'code-gate':
            p.add_argument('--role', choices=('wsl', 'server'), required=True)
        else:
            p.add_argument('--split-manifest', type=Path, required=True)
            p.add_argument('--source-lock', type=Path, required=True)
            if name in ('worker', 'verify-worker'):
                p.add_argument('--job', choices=[j['id'] for j in core.jobs()], required=True)
            else:
                p.add_argument('--wsl-gate', type=Path, required=True)
                p.add_argument('--server-gate', type=Path, required=True)
                p.add_argument('--gpus', default='0,1,2,3')
    p = sub.add_parser('package')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'code-gate':
        code_gate(args.output, args.commit, args.role)
    elif args.command in ('worker', 'verify-worker'):
        worker(args, args.command == 'verify-worker')
    elif args.command == 'run':
        run(args)
    else:
        print(json.dumps(package(args.root, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
