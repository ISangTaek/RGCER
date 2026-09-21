"""Approved Human6 smoke/formal dispatcher and one result check. No test API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import human6 as h


def utc():
    return datetime.now(timezone.utc).isoformat()


def jobs(phase):
    h.require(phase in ('smoke','formal'), 'phase')
    return [(m,s) for m in h.METHODS for s in ((42,) if phase=='smoke' else range(42,47))]


def name(method, seed):
    return f'{method}_s{seed}'


def inputs(args, p):
    return h.load_views(ROOT/p['data']['csv_path'], args.split_manifest, p)


def verify(work, phase, views, p):
    """Recompute metrics from predictions joined to current permitted raw truth."""
    import torch
    expected = jobs(phase); epochs = 6 if phase=='smoke' else 40
    root = Path(work)/phase
    h.require(root.is_dir(), 'missing phase')
    actual = {x.name for x in root.iterdir() if x.is_dir()}
    h.require(actual == {name(m,s) for m,s in expected}, 'missing/extra run directory')
    summaries = []; heads = {}
    for method, seed in expected:
        d = root/name(method,seed)
        r = json.loads((d/'result.json').read_text())
        for n in ('best.pt','last.pt','resolved_config.json','history.json','best_validation.json'):
            h.require((d/n).stat().st_size == r['files'][n]['size_bytes'] and h.digest(d/n) == r['files'][n]['sha256'], 'result file SHA')
        ident = json.loads((d/'resolved_config.json').read_text())
        source = next(a for a in p['source_assets'] if a['seed']==seed)
        h.require(ident == r['identity'] and ident['method'] == method and ident['seed'] == seed
                  and ident['policy_sha'] == h.semantic_digest(p) and ident['tasks'] == list(h.TASKS)
                  and ident['source_sha'] == source['sha256'] and ident['architecture'] == p['architecture']
                  and ident['initial_encoder'] == source['encoder_state_sha256'], 'result run/source identity')
        h.require(ident['scaler'] == h.TaskScaler.fit(views['train'].labels,h.TASKS,allow_empty=False).to_dict(), 'train scaler')
        h.require(heads.setdefault(seed,ident['initial_heads']) == ident['initial_heads'], 'unpaired heads')
        history = json.loads((d/'history.json').read_text())
        h.require(len(history)==epochs and r['epochs']==epochs and r['updates']==epochs*9, 'epochs/budget')
        for e,row in enumerate(history):
            h.require(row['epoch']==e and row['total_steps']==(e+1)*9 and row['exposure']==p['counts']['train'], 'history exposure/updates')
            met = row['validation']
            h.require(set(met['endpoints'])==set(h.TASKS), 'history tasks')
            h.require(all(met['endpoints'][t]['n']==p['counts']['validation'][t] for t in h.TASKS), 'validation N')
            h.require(all(h.math.isfinite(met['endpoints'][t][k]) and met['endpoints'][t][k]>=0
                          for t in h.TASKS for k in ('rmse','mae')), 'history nonfinite')
            for k in ('rmse','mae'):
                h.require(h.math.isclose(met['macro_'+k],sum(met['endpoints'][t][k] for t in h.TASKS)/6,abs_tol=1e-12), 'macro arithmetic')
            opt = row['optimization']
            h.require(opt['frozen'] == (method=='FROZEN' or method=='HF_low' and e<5)
                      and opt['head_lr']==.001 and opt['backbone_lr']==(0. if method=='FROZEN' else .0001), 'optimization history')
        best = min(range(epochs),key=lambda e:history[e]['validation']['macro_rmse'])
        h.require(r['best_epoch']==best, 'best selection')
        rows = json.loads((d/'best_validation.json').read_text())
        calculated = h.metrics(rows,views['validation'])
        h.require(calculated==r['metrics']==history[best]['validation'], 'best metrics')
        b = torch.load(d/'best.pt',map_location='cpu',weights_only=True)
        last = torch.load(d/'last.pt',map_location='cpu',weights_only=True)
        h.require(b['identity']==last['identity']==ident and b['epoch']==last['best_epoch']==best, 'checkpoint identity')
        h.require(last['history']==history and last['best_rows']==rows and last['total_steps']==9*epochs, 'last history')
        h.require(h.state_digest(b['model_state'])==b['model_state_sha']==last['best_state_sha']
                  ==h.state_digest(last['best_state']), 'best tensor parity')
        h.require(h.state_digest(last['model_state'])==last['model_state_sha'], 'last tensors')
        if method=='FROZEN':
            for state in (b['model_state'],last['model_state']):
                h.require(h.state_digest({k[8:]:v for k,v in state.items() if k.startswith('encoder.')})==ident['initial_encoder'], 'frozen encoder drift')
        summaries.append(dict(run_id=name(method,seed),method=method,seed=seed,best_epoch=best,
                              run_directory=str(d.resolve()), updates=r['updates'],metrics=calculated,files=r['files']))
    report = dict(phase=phase,runs=summaries,updates=sum(r['updates'] for r in summaries),
                  policy_sha=h.semantic_digest(p),validation_status='PASS',acceptance_status='PENDING_CODEX_REVIEW',test_accessed=False)
    h.json_write(Path(work)/(phase+'_summary.json'),report)
    return report


def gpu_free(gpu):
    # Index names refer to physical nvidia-smi GPUs, not inherited CUDA remapping.
    result = subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],
                            text=True,capture_output=True,check=True)
    h.require(int(result.stdout.strip()) < 512, 'selected GPU is not idle')
    processes = subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],
                               text=True,capture_output=True,check=True)
    h.require(not processes.stdout.strip(), 'selected GPU has another compute process')


def batch(args,p):
    phase = args.command
    gpus = args.gpus
    h.require(gpus and len(set(gpus))==len(gpus) and all(g in range(4) for g in gpus), 'GPU0-3 only, no duplicates')
    work = Path(args.work).resolve()
    work.mkdir(parents=True,exist_ok=True)
    phase_root = work/phase
    h.require(not phase_root.exists(), 'phase already started: do not repeat; report partial instead')
    views = inputs(args,p)
    for seed in range(42,47):
        h.source_asset(p,seed,args.source_root,tensors=False)
    if phase=='formal':
        verify(work,'smoke',views,p)
    for gpu in gpus:
        gpu_free(gpu)
    phase_root.mkdir()
    q = queue.Queue()
    for item in jobs(phase): q.put(item)
    stop = threading.Event(); records = []; lock = threading.Lock()
    def worker(gpu):
        while not stop.is_set():
            try: method,seed = q.get_nowait()
            except queue.Empty: return
            if stop.is_set(): return
            log = work/'logs'/phase/name(method,seed)
            log.mkdir(parents=True,exist_ok=False)
            cmd = [sys.executable,str(Path(__file__).resolve()),'one','--phase',phase,'--method',method,
                   '--seed',str(seed),'--work',str(work),'--split-manifest',str(Path(args.split_manifest).resolve())]
            if args.source_root: cmd += ['--source-root',str(Path(args.source_root).resolve())]
            rec = dict(method=method,seed=seed,gpu=gpu,command=cmd,start=utc())
            h.json_write(log/'command.json',rec)
            try:
                gpu_free(gpu)
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
                with (log/'stdout.log').open('x') as out, (log/'stderr.log').open('x') as err:
                    proc = subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,check=False)
                rec['exit_code']=proc.returncode
            except Exception as exc:
                rec.update(exit_code=-1,error=str(exc))
            rec['end']=utc()
            if rec['exit_code'] != 0: stop.set()
            h.json_write(log/'command.json',rec)
            with lock: records.append(rec)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for f in [pool.submit(worker,g) for g in gpus]: f.result()
    completed={(r['method'],r['seed']) for r in records}
    h.json_write(work/(phase+'_commands.json'),dict(commands=records,stopped_on_failure=stop.is_set(),
        not_started=[name(m,s) for m,s in jobs(phase) if (m,s) not in completed]))
    h.require(not stop.is_set(), 'a run failed; completed jobs retained, no automatic retry')
    report = verify(work,phase,views,p)
    print(json.dumps(dict(phase=phase,runs=len(report['runs']),updates=report['updates'],validation='PASS')))


def package(work,p):
    work = Path(work).resolve()
    h.require((work/'formal_summary.json').is_file(), 'verify formal first')
    # Exclude large checkpoints, caches and all unrelated material. Exact files only.
    paths = [work/'formal_summary.json',work/'smoke_summary.json',work/'formal_commands.json',work/'smoke_commands.json']
    for phase in ('smoke','formal'):
        for m,s in jobs(phase):
            d = work/phase/name(m,s)
            paths += [d/n for n in ('resolved_config.json','history.json','best_validation.json','result.json')]
            log = work/'logs'/phase/name(m,s)
            paths += [log/n for n in ('command.json','stdout.log','stderr.log')]
    # These files are supplied by the card; never collect SSH configs or credentials.
    for n in ('wsl_tests.xml','wsl_commands.log','server_tests.xml','server_commands.log'):
        paths.append(work/'evidence'/n)
    for path in paths: h.require(path.is_file(), 'missing delivery file: '+str(path))
    checks = {path.relative_to(work).as_posix():h.digest(path) for path in paths}
    dest = work.parent/(work.name+'_review.zip')
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths: z.write(path,path.relative_to(work).as_posix())
        z.writestr('checksums.json',json.dumps(checks,indent=2))
        z.writestr('runtime_policy.json',json.dumps(p,indent=2))
        z.writestr('README.md','Human6 smoke + train/validation only. Large best/last checkpoints remain under the server WORK path.\n')
    with zipfile.ZipFile(dest) as z:
        h.require(z.testzip() is None, 'ZIP CRC')
        for member, sha in checks.items():
            import hashlib
            h.require(hashlib.sha256(z.read(member)).hexdigest()==sha,'ZIP checksum')
    with dest.with_suffix('.zip.sha256').open('x') as f: f.write(h.digest(dest)+'  '+dest.name+'\n')
    print(dest)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['one','smoke','formal','verify','package'])
    ap.add_argument('--work',required=True)
    ap.add_argument('--split-manifest',required=True)
    ap.add_argument('--source-root')
    ap.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    ap.add_argument('--phase',choices=['smoke','formal'],default='formal')
    ap.add_argument('--method',choices=h.METHODS)
    ap.add_argument('--seed',type=int)
    a=ap.parse_args(); p=h.policy()
    if a.command in ('smoke','formal'): batch(a,p)
    elif a.command=='one':
        h.require((a.method,a.seed) in jobs(a.phase),'run outside matrix')
        v=inputs(a,p); state,asset=h.source_asset(p,a.seed,a.source_root)
        t=h.Trainer(v,state,p['architecture'],method=a.method,seed=a.seed,device='cuda:0',
                    policy_sha=h.semantic_digest(p),source_sha=asset['sha256'])
        r=t.run(Path(a.work)/a.phase/name(a.method,a.seed),epochs=6 if a.phase=='smoke' else 40)
        print(json.dumps(dict(method=a.method,seed=a.seed,epochs=r['epochs'],updates=r['updates'])))
    elif a.command=='verify':
        print(json.dumps(verify(a.work,a.phase,inputs(a,p),p),ensure_ascii=False))
    else:
        verify(a.work,'smoke',inputs(a,p),p)
        verify(a.work,'formal',inputs(a,p),p)
        package(a.work,p)


if __name__=='__main__': main()
