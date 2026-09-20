"""One-shot, fixed-budget P1D4 batch control. Pure scheduling plus process IO.

No arbitrary training commands or hyperparameters are accepted. Claims remain
after failure, and a new output path cannot obtain a second attempt.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,traceback

from p1d_optimization import require,select_arms
from p1d_schedule import screening_jobs,resolve_plan,job as frozen_job
from p1d4_identity import contract_for

TASK='P1D4_FORMAL_20260920'
REGISTRY='.tmp/p1d4_attempts_20260920'


def utc():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def read(path):
    def bad(x):raise ValueError('nonfinite JSON '+x)
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(Path(path).read_bytes(),parse_constant=bad,object_pairs_hook=pairs)


def check_commit(repo,expected):
    require(type(expected) is str and len(expected)==40 and set(expected)<=set('0123456789abcdef'),'full commit required')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    require(actual==expected,'checkout commit changed')
    status=subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=repo,text=True)
    require(not status.strip(),'working tree not clean')
    return actual


def claim(repo,job,root,commit):
    from p1d_tox_training import _same
    require(_same(job,frozen_job(job['setting'],job['arm'],job['seed'],job['phase'])),'job differs from frozen matrix')
    folder=Path(repo)/REGISTRY;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(job['run_id']+'.json')
    selection_sha=(hashlib.sha256((Path(root)/'selections.json').read_bytes()).hexdigest()
                   if job['phase']=='replication' else None)
    write(path,dict(task_id=TASK,job=job,root=str(Path(root).resolve()),commit=commit,started_utc=utc(),
                   selection_sha256=selection_sha))
    return path


def select_from_verified(results):
    selected={}
    for setting in ('ToxAcute','A','B'):
        histories={}
        for arm in ('B1_high','B1_low','HF_high','HF_low'):
            run=f'{setting}_{arm}_s42'
            value=results[run]
            require(value['job']==frozen_job(setting,arm,42,'screen'),'screen result identity')
            require(value['result']['validation_status']=='PASS','screen content not verified')
            histories[arm]=value['result']['history']
        c=contract_for(setting,42)
        tasks=list(c['counts']['validation']) if setting=='ToxAcute' else c['task_names']
        selected[setting]=select_arms(histories,setting=setting,seed=42,task_names=tasks)
    return selected


def execute_matrix(executor):
    """executor completes a whole phase, returning ONLY verified run receipts."""
    first=screening_jobs()
    results=executor(first)
    require(set(results)=={j['run_id'] for j in first},'incomplete screening phase')
    selections=select_from_verified(results)
    plan=resolve_plan(selections)
    rest=[j for j in plan['jobs'] if j['phase']=='replication']
    later=executor(rest)
    require(set(later)=={j['run_id'] for j in rest},'incomplete replication phase')
    for j in rest:
        r=later[j['run_id']]
        require(r['job']==j and r['result']['validation_status']=='PASS','replication identity/status')
    results.update(later)
    return dict(task_id=TASK,selections=selections,plan=plan,results=results,
                execution_status='FINISHED',validation_status='PASS',acceptance_status='PENDING_REVIEW')


def free_gpus(allowed):
    require(type(allowed) is list and 1<=len(allowed)<=4 and len(set(allowed))==len(allowed)
            and all(type(i) is int and i in range(4) for i in allowed),'GPU0-3 only; unique slots')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    mapping={row.split(',')[1].strip():int(row.split(',')[0]) for row in raw.splitlines() if row.strip()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    used=[row.split(',')[0].strip() for row in apps.splitlines() if row.strip()]
    require(all(uuid in mapping for uuid in used),'unknown compute GPU UUID')
    busy={mapping[uuid] for uuid in used}
    return [i for i in allowed if i in mapping.values() and i not in busy]


def run_batch(repo,root,commit,gpus,split_manifest,source_lock):
    repo=Path(repo).resolve();root=Path(root).resolve()
    check_commit(repo,commit)
    require(root.is_dir(),'batch launch directory missing')
    launch=read(root/'launch.json')
    require(launch['commit']==commit and launch['task_id']==TASK,'not the launched supervisor')
    require(read(repo/REGISTRY/'batch.json')['root']==str(root),'batch claim root')
    write(repo/REGISTRY/'supervisor.started.json',dict(pid=os.getpid(),root=str(root),utc=utc()))
    all_results={}
    def phase(jobs):
        # Freeze the selected matrix before any replication process is launched.
        if jobs[0]['phase']=='replication':
            selections=select_from_verified(all_results)
            write(root/'selections.json',selections)
            write(root/'resolved_plan.json',resolve_plan(selections))
        queue=list(jobs);active={};failed=[];results={}
        with ThreadPoolExecutor(max_workers=4) as pool:
            while queue or active:
                available=free_gpus(gpus)
                used={gpu for gpu,_ in active.values()}
                for gpu in [x for x in available if x not in used]:
                    if not queue:break
                    j=queue.pop(0)
                    claim(repo,j,root,commit)
                    future=pool.submit(run_child,repo,root,j,commit,gpu,split_manifest,source_lock)
                    active[future]=(gpu,j)
                for future in list(active):
                    if not future.done():continue
                    _,j=active.pop(future)
                    try:results[j['run_id']]=future.result()
                    except Exception as exc:failed.append(dict(run_id=j['run_id'],error=repr(exc)))
                if queue or active:time.sleep(2)
        if failed:
            write(root/(jobs[0]['phase']+'_failures.json'),failed)
            raise RuntimeError('phase failed; no dependent replication or automatic retries')
        all_results.update(results)
        return results
    try:
        result=execute_matrix(phase)
        shutil.copytree(repo/REGISTRY,root/'attempt_claims')
        write(root/'batch_summary.json',result)
        return 0
    except Exception:
        write(root/'batch_failure.json',dict(task_id=TASK,error=traceback.format_exc(),finished_utc=utc(),
              execution_status='FAILED',validation_status='FAIL',acceptance_status='PENDING_REVIEW'))
        return 1


def run_child(repo,root,j,commit,gpu,split_manifest,source_lock):
    check_commit(repo,commit)
    require(free_gpus([gpu])==[gpu],'assigned GPU became occupied; no retry')
    logdir=root/'logs'/j['run_id'];logdir.mkdir(parents=True)
    argv=[sys.executable,'-u','scripts/run_p1d4_batch.py','worker','--output',str(root),'--commit',commit,
          '--run-id',j['run_id'],'--split-manifest',str(split_manifest),'--source-lock',str(source_lock)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    start=utc()
    with (logdir/'stdout.log').open('xb') as out,(logdir/'stderr.log').open('xb') as err:
        process=subprocess.Popen(argv,cwd=repo,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL)
        write(logdir/'start.json',dict(argv=argv,pid=process.pid,commit=commit,gpu=gpu,started_utc=start))
        code=process.wait()
    write(logdir/'exit.json',dict(exit_code=code,started_utc=start,finished_utc=utc(),pid=process.pid))
    require(code==0,'worker failed '+j['run_id'])
    result=read(root/'receipts'/(j['run_id']+'.json'))
    require(result['job']==j and result['result']['validation_status']=='PASS','worker receipt')
    return result


def worker(repo,root,commit,run_id,split_manifest,source_lock):
    import torch
    from p1d4_runtime import execute_job
    check_commit(repo,commit)
    root=Path(root).resolve()
    require(type(run_id) is str and '/' not in run_id and '\\' not in run_id,'run id')
    entry=read(Path(repo)/REGISTRY/(run_id+'.json'))
    require(entry['root']==str(root) and entry['commit']==commit and entry['task_id']==TASK,'worker claim')
    j=entry['job'];require(j['run_id']==run_id,'worker identity')
    permitted=screening_jobs()
    if j['phase']=='replication':
        require(hashlib.sha256((root/'selections.json').read_bytes()).hexdigest()==entry['selection_sha256'],
                'selected plan changed after claim')
        permitted=resolve_plan(read(root/'selections.json'))['jobs']
    require(j in permitted,'worker outside approved matrix')
    write(Path(repo)/REGISTRY/(run_id+'.started.json'),dict(pid=os.getpid(),utc=utc()))
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one visible GPU required')
    torch.use_deterministic_algorithms(True)
    result=execute_job(repo,j,root/'runs'/run_id,split_manifest=split_manifest,source_lock=source_lock,device='cuda:0')
    result['peak_cuda_memory_allocated_bytes']=torch.cuda.max_memory_allocated()
    result['runtime']=dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                           gpu_name=torch.cuda.get_device_name(0),python=sys.version,
                           physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),commit=commit)
    (root/'receipts').mkdir(exist_ok=True)
    write(root/'receipts'/(run_id+'.json'),result)
    return 0
