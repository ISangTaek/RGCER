"""V11-S1: ten chemical multi-view methods and two capacity-matched controls.

72 new trajectories / 2,880 epochs / 60,480 optimizer updates. No new test
inference, pretraining, cost probe, source-feature fit, or discarded smoke.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import torch

import v10_s1_screen as s1
import v11_references as refs
import v11_features as chemical
import v10_s2_screen as s2
import v11_models as models

read, write, sha, digest, require = s1.read, s1.write, s1.sha, s1.digest, s1.require
cache, s3, orchestration = s1.cache, s1.s3, s1.orchestration
REPO = Path(__file__).resolve().parent
TASK = 'V11_S1_CHEMICAL_MULTIVIEW_SCREEN_20261004'
REGISTRY = REPO / '.tmp/v11_s1_attempt_20261004'
SETTINGS = s1.SETTINGS
SPEC = dict(s1.SPEC,new_methods=list(models.NEW),new_controls=list(models.CONTROLS),
            loss='MSE',candidate_count=10,source_cache_commit=refs.old.SOURCE_COMMIT,reference_commit=refs.SOURCE_COMMIT,
            new_trajectories=72,new_optimizer_updates=60480,model_configurations=models.CONFIGS,
            feature_spec=chemical.FEATURE_SPEC,tiny=False,
            ranking='max_min_relative_gain_then_mean_then_total_parameters_then_declared_order')
TESTS = s2.TESTS + ('tests/test_v11_features.py','tests/test_v11_models.py',
                    'tests/test_v11_screen.py','tests/test_v11_references.py')
TEST_COUNT = 256


def jobs():
    return [dict(id=f'{s}_{m}_c{c}_s42',setting=s,method=m,config=c,seed=42)
            for s in SETTINGS for m in models.NEW+models.CONTROLS for c in range(2)]


def make_model(data,name,spec):
    return models.build(name,data,tiny=spec['tiny'])


def identity(data, job, commit, spec):
    return dict(task=TASK,commit=commit,job=job,spec=spec,reference_lock_sha256=sha(refs.LOCK),
                source_cache_commit=data.value['commit'],cache_identity=data.value['identity'],
                cache_rows_sha256=digest(data.rows),metadata=data.value['metadata'],
                chemical_identity={k:chemical.tensor_sha(v) for k,v in data.chemical.items() if torch.is_tensor(v)},
                chemical_spec=data.chemical['spec'],
                source_tasks=data.sources,target_tasks=data.targets)


def episode(data, task, query, seed, spec, *, evaluation=False):
    return data.episode(task,query,cap=None if evaluation else spec['context_cap'],seed=seed)


def expected_diagnostics(e,name): return {}


def predict(data, name, model, spec):
    model.eval(); rows=[]; diagnostics={}
    with torch.no_grad():
        for task in data.targets:
            query=data.by[task,'validation']
            e=episode(data,task,query,42,spec,evaluation=True)
            prediction=model(e)
            require(prediction.shape==(len(query),1) and bool(torch.isfinite(prediction).all()), 'finite prediction')
            scaler=data.value['scalers'][task]
            prediction=prediction.double()*scaler['std']+scaler['mean']
            rows.extend(dict(data.rows[i],prediction=float(p)) for i,p in zip(query,prediction[:,0].cpu()))
            diagnostics[task]=models.diagnostics(model)
            require(diagnostics[task]==expected_diagnostics(e,name),'prediction diagnostics')
    return rows,s3.s2.metrics(rows,[r for r in data.rows if r['split']=='validation']),diagnostics


def coverage(data, histories):
    seen={t:Counter() for t in data.tasks}; episodes=Counter()
    for h in histories:
        for step in h['trace']:
            for e in step['episodes']:
                seen[e['task']].update(e['query']); episodes[e['task']]+=1
    counts={t:dict(query_observations=sum(c.values()),unique_query_observations=len(c),
                   population=len(data.by[t,'train']),updates=episodes[t]) for t,c in seen.items()}
    return seen,counts


def train_case(data, job, out, commit, spec=SPEC):
    require(job in jobs(),'fixed V11 training job'); out.mkdir(parents=True,exist_ok=False)
    s1.seed_everything(42,deterministic_algorithms=True)
    name=job['method']; model=make_model(data,name,spec)
    optimizer=models.optimizer(model,job['config'])
    ident=identity(data,job,commit,spec); history=[]; best=math.inf; updates=0; started=time.monotonic()
    total_planned=sum(len(cache.epoch_schedule(data,i,batch_size=spec['query_batch'])) for i in range(spec['epochs']))
    for epoch in range(1,spec['epochs']+1):
        trace=[]
        schedule=cache.epoch_schedule(data,epoch-1,batch_size=spec['query_batch'])
        for step,item in enumerate(schedule):
            model.train(); optimizer.zero_grad(set_to_none=True); records=[]
            for j,(task,query) in enumerate([item['target']]+item['source']):
                e=episode(data,task,query,4200000+epoch*10000+step*100+j,spec)
                loss=model.loss(e,data.y[query])
                require(loss.ndim==0 and bool(torch.isfinite(loss)),'finite training loss')
                weight=1. if j==0 else spec['source_loss_weight']/len(item['source'])
                (loss*weight).backward()
                diagnostic=models.diagnostics(model)
                require(diagnostic==expected_diagnostics(e,name),'training diagnostics')
                records.append(dict(task=task,query=query,context=[r.sample_id for r in e.base.context_rows],
                                    loss=float(loss.detach()),weight=weight,diagnostics=diagnostic))
            params=[p for p in model.parameters() if p.grad is not None]
            require(params and all(bool(torch.isfinite(p.grad).all()) for p in params),'finite gradients')
            norm=torch.nn.utils.clip_grad_norm_(params,spec['grad_clip'],error_if_nonfinite=True)
            models.schedule(optimizer,model,updates,total_planned)
            optimizer.step(); updates+=1
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'finite parameters')
            trace.append(dict(episodes=records,gradient_norm=float(norm)))
        rows,score,diagnostics=predict(data,name,model,spec)
        write(out/f'validation_{epoch:03d}.json',rows)
        h=dict(epoch=epoch,score=score,updates=len(schedule),trace=trace,validation_diagnostics=diagnostics)
        write(out/f'epoch_{epoch:03d}.json',h); history.append(h)
        if score['macro_rmse']<best:
            best=score['macro_rmse']; chosen=epoch
            payload=dict(identity=ident,epoch=epoch,updates=updates,optimizer=optimizer.state_dict(),
                         state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            with (out/'best.pending.pt').open('xb') as f: torch.save(payload,f)
            os.replace(out/'best.pending.pt',out/'best.pt')
        print(f'{job["id"]} epoch={epoch} macro_rmse={score["macro_rmse"]:.8f}',flush=True)
    seen,counts=coverage(data,history)
    for t in data.targets:
        require(set(seen[t])==set(data.by[t,'train']) and set(seen[t].values())=={spec['epochs']},'full target coverage')
    require(all(counts[t]['updates']>=spec['epochs'] for t in data.sources),'all source tasks')
    selected=read(out/f'validation_{chosen:03d}.json'); write(out/'selected_validation.json',selected)
    receipt=dict(identity=ident,best_epoch=chosen,selected=history[chosen-1]['score'],
                 selected_diagnostics=history[chosen-1]['validation_diagnostics'],
                 history_sha256=digest(history),checkpoint_sha256=sha(out/'best.pt'),
                 updates=updates,coverage=counts,epochs=len(history),architecture=models.describe(model),
                 elapsed_seconds=time.monotonic()-started,test_evaluated=False,scientific_acceptance='PENDING_REVIEW',
                 environment=dict(python=sys.version,torch=str(torch.__version__),device=str(data.device),
                                  cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if str(data.device).startswith('cuda') else None))
    write(out/'receipt.json',receipt)
    return receipt


def verify_case(data, job, out, commit, spec=SPEC):
    r=read(out/'receipt.json'); name=job['method']
    require(r['identity']==identity(data,job,commit,spec) and r['epochs']==spec['epochs']
            and r['test_evaluated'] is False and r['scientific_acceptance']=='PENDING_REVIEW','V11 receipt identity/scope')
    histories=[]; total=0; expected=[x for x in data.rows if x['split']=='validation']
    for epoch in range(1,spec['epochs']+1):
        h=read(out/f'epoch_{epoch:03d}.json'); histories.append(h)
        require(h['epoch']==epoch and h['score']==s3.s2.metrics(read(out/f'validation_{epoch:03d}.json'),expected),'epoch score')
        schedule=cache.epoch_schedule(data,epoch-1,batch_size=spec['query_batch'])
        require(h['updates']==len(schedule)==len(h['trace']),'epoch updates')
        for step,(record,item) in enumerate(zip(h['trace'],schedule)):
            group=[item['target']]+item['source']
            require(len(record['episodes'])==len(group) and type(record['gradient_norm']) in (int,float)
                    and math.isfinite(record['gradient_norm']) and record['gradient_norm']>=0,'trace batch')
            for j,(v,(task,query)) in enumerate(zip(record['episodes'],group)):
                e=episode(data,task,query,4200000+epoch*10000+step*100+j,spec)
                require(set(v)=={'task','query','context','loss','weight','diagnostics'} and
                        v['task']==task and v['query']==query and v['context']==[a.sample_id for a in e.base.context_rows]
                        and type(v['loss']) in (int,float) and math.isfinite(v['loss']) and
                        v['weight']==(1. if j==0 else spec['source_loss_weight']/len(item['source'])) and
                        v['diagnostics']==expected_diagnostics(e,name),'episode trace identity/diagnostics')
        require(h['validation_diagnostics']=={t:expected_diagnostics(
            episode(data,t,data.by[t,'validation'],42,spec,evaluation=True),name) for t in data.targets},'validation diagnostics')
        total+=len(schedule)
    require(digest(histories)==r['history_sha256'] and total==r['updates'] and
            coverage(data,histories)[1]==r['coverage'],'history/coverage/updates')
    chosen=min(histories,key=lambda h:h['score']['macro_rmse'])
    require(r['best_epoch']==chosen['epoch'] and r['selected']==chosen['score'] and
            r['selected_diagnostics']==chosen['validation_diagnostics'],'earliest minimum selection')
    selected=read(out/'selected_validation.json')
    require(selected==read(out/f'validation_{r["best_epoch"]:03d}.json') and sha(out/'best.pt')==r['checkpoint_sha256'],
            'selected checkpoint/rows binding')
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    require(payload['identity']==r['identity'] and payload['epoch']==r['best_epoch'] and
            payload['updates']==sum(h['updates'] for h in histories[:r['best_epoch']]),'checkpoint provenance')
    model=make_model(data,name,spec); model.load_state_dict(payload['state'],strict=True)
    require(models.describe(model)==r['architecture'] and all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            'model architecture/finite weights')
    optimizer=models.optimizer(model,job['config'])
    optimizer.load_state_dict(payload['optimizer'])
    require(bool(optimizer.state) and all(int(x['step'])==payload['updates'] for x in optimizer.state.values()) and
            all(bool(torch.isfinite(v).all()) for x in optimizer.state.values() for v in x.values() if torch.is_tensor(v)),
            'optimizer state/steps')
    expected_optimizer=models.optimizer(model,job['config'])
    models.schedule(expected_optimizer,model,payload['updates']-1,total)
    require(len(optimizer.param_groups)==len(expected_optimizer.param_groups),'optimizer group count')
    expected_ids={id(p) for p in model.parameters()}
    require({id(p) for group in optimizer.param_groups for p in group['params']}==expected_ids
            and len(optimizer.state)==len(expected_ids),'complete optimizer parameter states')
    for group,expected_group in zip(optimizer.param_groups,expected_optimizer.param_groups):
        require({k:v for k,v in group.items() if k!='params'}=={k:v for k,v in expected_group.items() if k!='params'},
                'optimizer configuration/schedule')
    replay,_,diagnostics=predict(data,name,model,spec); s1.close_predictions(replay,selected,spec)
    require(diagnostics==r['selected_diagnostics'],'selected replay diagnostics')
    return r


def describe_result(x):
    return {k:v for k,v in x.items() if k!='rows'}


def comparison_record(x, fixed, control):
    score=x['score']['macro_rmse']; threshold=min(fixed['score']['macro_rmse'],control['score']['macro_rmse'])
    return dict(**describe_result(x),delta_frozen=score-fixed['score']['macro_rmse'],
                delta_new_control=score-control['score']['macro_rmse'],passes_point_gate=score<threshold,
                relative_gain=(threshold-score)/threshold,
                versus_frozen=s1.grouped_difference(x['rows'],fixed['rows']),
                versus_new_control=s1.grouped_difference(x['rows'],control['rows']))


def compare(root, references):
    settings={}
    for setting in SETTINGS:
        selected={}
        for method in models.NEW+models.CONTROLS:
            options=[read(root/j['id']/'receipt.json') for j in jobs() if j['setting']==setting and j['method']==method]
            r=min(options,key=lambda x:x['selected']['macro_rmse']); job=r['identity']['job']
            selected[method]=dict(name=method,job=job,best_epoch=r['best_epoch'],score=r['selected'],
                                  parameter_count=r['architecture']['parameters'],rows=read(root/job['id']/'selected_validation.json'))
        strongest=min(models.CONTROLS,key=lambda m:selected[m]['score']['macro_rmse'])
        ref=references[setting]
        settings[setting]=dict(frozen_best=describe_result(ref['best']),new_control_best=strongest,
            all_086_methods={m:describe_result(x) for m,x in ref['selected086'].items()},
            all_085_methods={m:describe_result(x) for m,x in ref['selected'].items()},
            fixed_mixtures={m:describe_result(x) for m,x in ref['mixtures'].items()},
            historical_scoreboard=ref['historical']['scoreboard'],
            new_controls={m:describe_result(selected[m]) for m in models.CONTROLS},
            new_candidates={m:comparison_record(selected[m],ref['best'],selected[strongest]) for m in models.NEW})
    passed=[m for m in models.NEW if all(settings[s]['new_candidates'][m]['passes_point_gate'] for s in SETTINGS)]
    def rank(m):
        xs=[settings[s]['new_candidates'][m] for s in SETTINGS]
        return (-min(x['relative_gain'] for x in xs),-sum(x['relative_gain'] for x in xs)/len(xs),
                sum(x['parameter_count'] for x in xs),models.NEW.index(m))
    ranked=sorted(passed,key=rank)
    return dict(settings=settings,development_candidates=ranked,selected_candidate=ranked[0] if ranked else None,
                ranking_parameter_rule='sum_across_three_settings',
                decision='BOUNDED_CONFIRMATION_REQUIRES_REVIEW' if ranked else 'NO_UNIFIED_MULTIVIEW_WINNER_REVIEW_REQUIRED',
                uniform_superiority_established=False)


def gate_check(folder, commit, role):
    require(read(folder/'gate.json')==dict(task=TASK,commit=commit,role=role,tests=list(TESTS),
                                         junit_sha256=sha(folder/'tests.xml')),'V11 code gate identity')
    require(read(folder/'command.json')['exit_code']==0,'V11 code gate exit')
    cases=ET.parse(folder/'tests.xml').findall('.//testcase')
    require(TEST_COUNT>0 and len(cases)==TEST_COUNT and all(not any(c.findall(t) for t in ('failure','error','skipped')) for c in cases)
            and {c.attrib['classname'] for c in cases}=={'tests.'+Path(t).stem for t in TESTS},'complete V11 code gate')


def code_gate(folder, commit, role):
    require(role in ('wsl','server'),'explicit code gate role'); s3.s2.c0.check_code(REPO,commit)
    folder.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(folder/'pytest_tmp'),'--junitxml',str(folder/'tests.xml')]
    with (folder/'tests.log').open('xb') as log:
        p=subprocess.run(argv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    write(folder/'command.json',dict(argv=argv,exit_code=p.returncode)); require(p.returncode==0,'V11 code tests failed')
    write(folder/'gate.json',dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(folder/'tests.xml')))
    gate_check(folder,commit,role)


def command(mode, name, a):
    return [sys.executable,str(Path(__file__).resolve()),'_worker','--mode',mode,'--name',name,
            '--output',str(a.output),'--commit',a.commit,'--split-manifest',str(a.split_manifest),'--source-lock',str(a.source_lock)]


def check_launch(a):
    s3.s2.c0.check_code(REPO,a.commit); v=read(a.output/'launch.json'); locked=refs.lock()
    require(v['task']==TASK and v['commit']==a.commit and v['spec']==SPEC and v['jobs']==jobs()
            and v['split_sha256']==sha(a.split_manifest) and v['source_lock_sha256']==sha(a.source_lock)
            and v['reference_lock_sha256']==sha(refs.LOCK) and v['reference086_manifest_sha256']==locked['manifest_sha256'],
            'V11 launch inputs/commit')
    old=read(a.output/'reference_086/launch.json')
    require(v['split_sha256']==old['split_sha256'] and v['source_lock_sha256']==old['source_lock_sha256'],
            'V11 frozen split/source inputs')
    require(read(REGISTRY/'attempt.json')==dict(task=TASK,commit=a.commit,output=str(a.output.resolve())), 'V11 one-shot claim')
    return v


def worker(a):
    s1.seed_everything(42,deterministic_algorithms=True); launch=check_launch(a)
    if a.mode=='verify':
        require(a.name in SETTINGS and os.environ.get('CUDA_VISIBLE_DEVICES')=='','V11 CPU verification scope')
        torch.set_num_threads(1)
        value=refs.load_features(a.output/'reference_086',a.name)
        f,t,source=s3.context(a.name,a.split_manifest,a.source_lock,'cpu')
        _,audit=s1.audit_cache(f,t,source,a.name,value)
        chem=chemical.load(value,a.output/'chemical'/a.name,refs.graph_manifest_sha(a.output/'reference_086',a.name),a.commit)
        chemical.check(value,chem,raw_replay=True,robust=True); data=chemical.Data(value,chem)
        receipts=[verify_case(data,j,a.output/j['id'],a.commit) for j in jobs() if j['setting']==a.name]
        write(a.output/('verified_'+a.name+'.json'),dict(task=TASK,commit=a.commit,setting=a.name,pid=os.getpid(),
              content_status='PASS',device='cpu',new_optimizer_updates=0,raw_cache_audit=audit,
              chemical_raw_replay='all_unique_SMILES',statistics_recomputed_from='all_legal_train_rows',
              prediction_input='byte_locked_085_feature_cache',
              receipts={r['identity']['job']['id']:sha(a.output/r['identity']['job']['id']/'receipt.json') for r in receipts}))
        return
    require(a.mode=='train','V11 training mode'); job=next(j for j in jobs() if j['id']==a.name)
    assignment=read(a.output/('train_'+a.name+'.assignment.json'))
    require(assignment['gpu'] in launch['gpus'] and assignment['uuid']==launch['gpu_uuids'][str(assignment['gpu'])]
            and os.environ.get('CUDA_VISIBLE_DEVICES')==assignment['uuid']
            and torch.cuda.is_available() and torch.cuda.device_count()==1,'single authorized GPU')
    value=refs.load_features(a.output/'reference_086',job['setting'])
    chem=chemical.load(value,a.output/'chemical'/job['setting'],refs.graph_manifest_sha(a.output/'reference_086',job['setting']),a.commit)
    r=train_case(chemical.Data(value,chem,'cuda:0'),job,a.output/job['id'],a.commit)
    require(r['updates']==refs.lock()['trajectory_updates'][job['setting']],'fixed V11 trajectory updates')


def verify(a):
    check_launch(a)
    for role in ('wsl','server'): gate_check(a.output/(role+'_evidence'),a.commit,role)
    reference=refs.references(a.output/'reference_086')
    for job in jobs(): require(read(a.output/('train_'+job['id']+'.command.json'))['exit_code']==0,'V11 worker exit')
    for setting in SETTINGS:
        orchestration.call_worker(command('verify',setting,a),a.output/('verify_'+setting),
                                  dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    checked=[read(a.output/('verified_'+s+'.json')) for s in SETTINGS]
    for setting,v in zip(SETTINGS,checked):
        require(v['task']==TASK and v['commit']==a.commit and v['setting']==setting and v['device']=='cpu'
                and v['content_status']=='PASS' and v['new_optimizer_updates']==0 and
                v['receipts']=={j['id']:sha(a.output/j['id']/'receipt.json') for j in jobs() if j['setting']==setting},
                'V11 independent verification binding')
    require(len({v['pid'] for v in checked})==3,'three independent CPU processes')
    receipts=[read(a.output/j['id']/'receipt.json') for j in jobs()]
    require(sum(r['epochs'] for r in receipts)==2880 and sum(r['updates'] for r in receipts)==60480,'V11 total matrix')
    result=dict(task=TASK,commit=a.commit,content_status='PASS',scientific_acceptance='PENDING_REVIEW',
                candidates=10,reused_candidates=0,new_candidates=10,new_controls=2,new_jobs=72,new_epochs=2880,
                new_optimizer_updates=60480,verification=checked,**compare(a.output,reference))
    write(a.output/'verification.json',result)
    return result


def run(a):
    from p1d4_batch import free_gpus
    s3.s2.c0.check_code(REPO,a.commit)
    for role in ('wsl','server'): gate_check(getattr(a,role+'_evidence'),a.commit,role)
    require(not a.output.exists() and not REGISTRY.exists(),'existing V11 run/attempt; preserve and report')
    require(a.gpus and len(set(a.gpus))==len(a.gpus) and set(a.gpus)<={0,1,2,3},'GPU scope')
    for path in (a.reference086_root,a.wsl_evidence,a.server_evidence):
        require(not a.output.resolve().is_relative_to(path.resolve()) and not path.resolve().is_relative_to(a.output.resolve()),'output overlaps input')
    refs.references(a.reference086_root,complete=True)
    old=read(a.reference086_root/'launch.json')
    require(sha(a.split_manifest)==old['split_sha256'] and sha(a.source_lock)==old['source_lock_sha256'],'frozen input files')
    for setting in SETTINGS: refs.load_features(a.reference086_root,setting)
    available=free_gpus(a.gpus); require(available,'no free authorized GPU')
    a.output.mkdir(parents=True); REGISTRY.mkdir(parents=True)
    write(REGISTRY/'attempt.json',dict(task=TASK,commit=a.commit,output=str(a.output.resolve())))
    uuids={str(g):s3.s2.c0.gpu_uuid(g) for g in available}
    write(a.output/'launch.json',dict(task=TASK,commit=a.commit,spec=SPEC,jobs=jobs(),gpus=available,gpu_uuids=uuids,
          split_sha256=sha(a.split_manifest),source_lock_sha256=sha(a.source_lock),reference_lock_sha256=sha(refs.LOCK),
          reference086_manifest_sha256=refs.lock()['manifest_sha256']))
    try:
        write(a.output/'reference_import.json',refs.copy_snapshot(a.reference086_root,a.output/'reference_086'))
        for setting in SETTINGS:
            value=refs.load_features(a.output/'reference_086',setting)
            chemical.save(value,a.output/'chemical'/setting,refs.graph_manifest_sha(a.output/'reference_086',setting),a.commit)
        for role in ('wsl','server'):
            shutil.copytree(getattr(a,role+'_evidence'),a.output/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
        def execute(job,gpu):
            require(free_gpus([gpu])==[gpu],'assigned GPU became busy')
            name=job['id']; write(a.output/('train_'+name+'.assignment.json'),dict(gpu=gpu,uuid=uuids[str(gpu)]))
            orchestration.call_worker(command('train',name,a),a.output/('train_'+name),dict(os.environ,
                CUDA_VISIBLE_DEVICES=uuids[str(gpu)],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        orchestration.dispatch(jobs(),available,execute)
        verify(a)
    except BaseException as exc:
        write(a.output/'failed.json',dict(task=TASK,commit=a.commit,error_type=type(exc).__name__,reason=str(exc))); raise


def package(root, commit):
    require(root.is_dir() and read(root/'launch.json')['task']==TASK and read(root/'launch.json')['commit']==commit,'V11 package identity')
    target=root.with_name(root.name+'.zip')
    require(not target.exists() and not Path(str(target)+'.sha256').exists(),'existing V11 package')
    files={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'package symlink')
        if p.is_file():
            name=p.relative_to(root).as_posix()
            require(p.suffix in ('.json','.log','.xml','.pt') or name in ('reference_086/checksums.sha256','reference_086/reference_085/checksums.sha256'),'unexpected V11 package file')
            files[name]=sha(p)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(root/name,name)
        z.writestr('checksums.sha256',''.join(h+'  '+n+'\n' for n,h in files.items()))
    with zipfile.ZipFile(target) as z:
        require(set(z.namelist())==set(files)|{'checksums.sha256'} and z.testzip() is None,'V11 package members/CRC')
        for name,h in files.items(): require(hashlib.sha256(z.read(name)).hexdigest()==h,'V11 package content SHA')
    with Path(str(target)+'.sha256').open('x',encoding='utf8') as f: f.write(sha(target)+'  '+target.name+'\n')
    return dict(archive=str(target),sha256=sha(target),scientific_acceptance='NOT_ASSESSED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('code-gate','unpack-reference','reference-check','run','verify','package','_worker'))
    p.add_argument('--commit',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=('wsl','server')); p.add_argument('--mode',choices=('train','verify')); p.add_argument('--name')
    p.add_argument('--gpus',nargs='+',type=int,choices=range(4))
    for name in ('split-manifest','source-lock','reference086-root','reference086-archive','wsl-evidence','server-evidence'): p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.action=='code-gate': code_gate(a.output,a.commit,a.role)
    elif a.action=='unpack-reference':
        s3.s2.c0.check_code(REPO,a.commit)
        print(refs.unpack_reference(a.reference086_archive,a.output))
    elif a.action=='reference-check':
        s3.s2.c0.check_code(REPO,a.commit)
        values=refs.references(a.reference086_root,complete=True)
        for setting in SETTINGS: refs.load_features(a.reference086_root,setting)
        print({s:dict(name=x['best']['name'],score=x['best']['score']) for s,x in values.items()})
    elif a.action=='run': run(a)
    elif a.action=='_worker': worker(a)
    elif a.action=='verify': verify(a)
    else:
        s3.s2.c0.check_code(REPO,a.commit); print(package(a.output,a.commit))


if __name__=='__main__': main()
