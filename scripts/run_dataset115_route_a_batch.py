"""Fixed 5-source then 15-target queue; one process per explicitly free GPU.

The per-run entry performs content verification before exit 0. No human review
between seeds, no retries or tuning, no source smoke reuse, no test access.
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
from dataset115_route_a_training import TASK_ID

SOURCES=[('source',None,s) for s in range(42,47)]
TARGETS=[('target',m,s) for s in range(42,47) for m in ('B0','B1','RPT')]
MATRIX=SOURCES+TARGETS


def name(role,method,seed):return f'{method if role=="target" else "source"}_seed{seed}'


def make_command(python,repo,a,job,root):
    role,method,seed=job
    cmd=[python,str(repo/'scripts/train_dataset115_route_a.py')]
    for k in ('csv','split_manifest','tox_manifest','expected_commit'):cmd+=['--'+k.replace('_','-'),str(getattr(a,k))]
    cmd+=['--role',role,'--seed',str(seed),'--output',str(root/'runs'/name(*job))]
    if role=='target':cmd+=['--method',method]
    if role=='target' and method!='B0':cmd+=['--source-output',str(root/'runs'/f'source_seed{seed}')]
    return cmd


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for k in ('csv','split-manifest','tox-manifest','expected-commit','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--gpus',nargs='+',type=int,choices=range(4),required=True)
    a=p.parse_args(argv)
    if len(set(a.gpus))!=len(a.gpus):p.error('duplicate GPU')
    out=Path(a.output).resolve()
    if out.exists():p.error('output exists: no overwrite/resume')
    for k in ('csv','split_manifest','tox_manifest'):
        path=Path(getattr(a,k)).resolve()
        if not path.is_file():p.error('missing input '+k)
        setattr(a,k,str(path))
    out.mkdir(parents=True,exist_ok=False);repo=Path(__file__).resolve().parents[1]
    lock=threading.Lock();rows=[];stopped=False
    def utc():return datetime.now(timezone.utc).isoformat()
    def phase(jobs):
        pending=list(jobs)
        def worker(gpu):
            nonlocal stopped
            while True:
                with lock:
                    if stopped or not pending:return
                    job=pending.pop(0)
                role,method,seed=job;runid=name(*job);log=out/'command_logs'/runid;log.mkdir(parents=True)
                cmd=make_command(sys.executable,repo,a,job,out)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(seed))
                record=dict(role=role,method=method,seed=seed,argv=cmd,cwd=str(repo),started_utc=utc(),
                    cuda_visible_devices=str(gpu),hostname=__import__('socket').gethostname())
                error=None
                try:
                    with (log/'stdout.txt').open('xb') as stdout,(log/'stderr.txt').open('xb') as stderr:
                        rc=subprocess.run(cmd,cwd=repo,env=env,stdout=stdout,stderr=stderr,check=False).returncode
                except Exception as exc:rc=-1;error=type(exc).__name__+': '+str(exc)
                record.update(finished_utc=utc(),exit_code=rc,launch_error=error)
                (log/'command.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf8')
                with lock:
                    rows.append(dict(role=role,method=method,seed=seed,exit_code=rc,gpu=gpu))
                    if rc!=0:stopped=True
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            for f in [pool.submit(worker,g) for g in a.gpus]:f.result()
    phase(SOURCES)
    if not stopped:phase(TARGETS)
    started={(r['role'],r['method'],r['seed']) for r in rows}
    def entry(j):return dict(role=j[0],method=j[1],seed=j[2])
    summary=dict(task_id=TASK_ID,commit=a.expected_commit,planned_matrix=[entry(j) for j in MATRIX],results=rows,
        not_started=[entry(j) for j in MATRIX if j not in started],automatic_retries=0,
        test_predictions_accessed=False,acceptance_status='PENDING_REVIEW')
    (out/'batch_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(summary,allow_nan=False))
    return 0 if len(rows)==20 and not stopped else 2


if __name__=='__main__':raise SystemExit(main())
