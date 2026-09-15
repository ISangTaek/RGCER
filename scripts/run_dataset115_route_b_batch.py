"""Execute the frozen 15 Route B runs, one process per explicitly free GPU.

No shell commands, retries, GPU probing decisions, or metric-based selection.
On failure no more jobs start; active independent jobs finish and keep evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
MATRIX = [(m,s) for s in range(42,47) for m in ('B0','B1','RPT')]


def make_command(python, repo, a, method, seed, out):
    cmd=[python,str(repo/'scripts/train_dataset115_route_b.py')]
    for key in ('csv','split_manifest','tox_manifest','source_lock','expected_commit'):
        cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
    return cmd+['--method',method,'--seed',str(seed),'--output',str(out)]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('csv','split-manifest','tox-manifest','source-lock','expected-commit','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True,choices=range(4))
    a=p.parse_args(argv)
    if len(a.gpus)!=len(set(a.gpus)):p.error('duplicate GPU')
    output=Path(a.output).resolve()
    if output.exists():p.error('output exists; never overwrite or silently resume')
    for key in ('csv','split_manifest','tox_manifest','source_lock'):
        path=Path(getattr(a,key)).resolve()
        if not path.is_file():p.error('missing input: '+key)
        setattr(a,key,str(path))
    output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    lock=threading.Lock(); pending=list(MATRIX); rows=[]; stopped=False
    def utc():return datetime.now(timezone.utc).isoformat()
    def worker(gpu):
        nonlocal stopped
        while True:
            with lock:
                if stopped or not pending:return
                method,seed=pending.pop(0)
            run_id=f'{method}_seed{seed}';log=output/'command_logs'/run_id;log.mkdir(parents=True)
            cmd=make_command(sys.executable,repo,a,method,seed,output/'runs'/run_id)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(seed))
            record=dict(method=method,seed=seed,argv=cmd,cwd=str(repo),started_utc=utc(),
                cuda_visible_devices=str(gpu),cublas_workspace_config=':4096:8',hostname=__import__('socket').gethostname())
            error=None
            try:
                with (log/'stdout.txt').open('xb') as stdout,(log/'stderr.txt').open('xb') as stderr:
                    rc=subprocess.run(cmd,cwd=repo,env=env,stdout=stdout,stderr=stderr,check=False).returncode
            except Exception as exc:
                rc=-1;error=type(exc).__name__+': '+str(exc)
            record.update(finished_utc=utc(),exit_code=rc,launch_error=error)
            (log/'command.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf8')
            with lock:
                rows.append(dict(method=method,seed=seed,exit_code=rc,gpu=gpu))
                if rc!=0:stopped=True
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in a.gpus]
        for f in futures:f.result()
    summary=dict(task_id='S5B_ROUTEB_15_TRAIN_VALIDATION_20260915',commit=a.expected_commit,
        planned_matrix=[dict(method=m,seed=s) for m,s in MATRIX],results=rows,
        not_started=[dict(method=m,seed=s) for m,s in pending],automatic_retries=0,
        test_predictions_accessed=False,acceptance_status='PENDING_REVIEW')
    (output/'batch_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(summary,allow_nan=False))
    return 0 if len(rows)==15 and not stopped else 2


if __name__=='__main__':raise SystemExit(main())
