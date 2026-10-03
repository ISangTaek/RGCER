"""V10-S1: frozen-function/context portfolio; 10 methods, 4 controls, 3 settings."""
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

import numpy as np
import torch

import v9_s5_screen as orchestration
import v10_context_models as models
import v10_feature_cache as cache
from reproducibility import seed_everything, state_dict_sha256

s3 = cache.s3
read, write, sha, digest, require = cache.read, cache.write, cache.sha, cache.digest, cache.require
REPO = Path(__file__).resolve().parent
TASK = 'V10_S1_FUNCTION_CONTEXT_SCREEN_20261003'
LOCK = REPO/'configs/v10_s1_reference_lock.json'
REGISTRY = REPO/'.tmp/v10_s1_attempt_20261003'
SETTINGS = ('ToxAcute','A','B')
SPEC = dict(seed=42, epochs=40, configurations=[.001,.0003], hidden=64, query_batch=16,
            context_cap=128, source_queries_per_task_epoch=16, weight_decay=.00001, grad_clip=1.,
            source_loss_weight=1., selection='min_per_config_first_epoch_then_min_configs_first_config',
            feature_statistics='all_source_and_target_train_observations', test_access=False,
            prediction_rtol=.0001, prediction_atol=.00002,
            predictive_latent_draws=16, source_cache_audit_per_task=2)
TESTS = ('tests/test_v10_functional_transfer.py','tests/test_v10_function_bank_integration.py',
         'tests/test_v10_context_models.py','tests/test_v10_s1_screen.py','tests/test_v10_cache_integration.py')
TEST_COUNT = 97


def jobs():
    return [dict(id=f'{s}_{m}_c{c}_s42',setting=s,method=m,config=c,seed=42)
            for s in SETTINGS for m in models.CANDIDATES+models.CONTROLS for c in range(2)]


def check_binding(value, entry):
    ident = value['identity']; expected = entry['identity']
    require(all(ident[k] == expected[k] for k in ('initial_encoder','contract_sha256','train_sha256','validation_sha256'))
            and digest(ident['source']) == expected['source_sha256']
            and ident['source_train_sha256'] == entry['source_sha256'], 'locked feature provenance')
    for split, key in (('train','train_sha256'),('validation','validation_sha256')):
        rows = [r for r in value['rows'] if r['task'] in value['targets'] and r['split'] == split]
        require(digest(rows) == entry[key], 'locked target population')
    source = [r for r in value['rows'] if r['task'] in value['sources']]
    require(digest(source) == entry['source_sha256'], 'locked source population')


def normalize_066(rows, expected):
    lookup={(r['task'],r['sample_id']):r for r in expected}; enriched=[]
    base={'task','sample_id','split','label','prediction'}
    for r in rows:
        require(set(r) in (base,base|{'canonical','group','lower','upper'}),'066 row schema')
        original=lookup[r['task'],r['sample_id']]
        require(all(r[k]==original[k] for k in set(r)&set(original)),'066 population pairing')
        enriched.append(dict(original,prediction=r['prediction']))
    s3.s2.metrics(enriched,expected)
    return enriched


def references(root084, root066, destination=None):
    """Only frozen validation files and source-train evidence, never test files."""
    lock = read(LOCK); require(lock['schema'] == 'v10_s1_reference_v1', 'reference schema')
    roots = {'084':root084,'066':root066}; copied = set(); result = {}
    def get(root, name, expected):
        p = roots[root]/name
        require(p.resolve().is_relative_to(roots[root].resolve()) and not p.is_symlink()
                and sha(p) == expected, 'locked reference file: '+name)
        if destination is not None and (root,name) not in copied:
            out = destination/root/name; out.parent.mkdir(parents=True,exist_ok=True)
            require(not out.exists(), 'existing reference copy'); shutil.copyfile(p,out); copied.add((root,name))
        return read(p)
    for setting, entry in lock['settings'].items():
        evidence = {n:get('084',entry['base']+'/'+n+'.json',h) for n,h in entry['files'].items()}
        expected = evidence['validation_observations']
        scores = []
        for descriptor in entry['old_scoreboard']:
            rows = get(descriptor['root'],descriptor['path'],descriptor['sha256'])
            if descriptor['key'] is not None: rows = rows['methods'][descriptor['key']]
            if descriptor['root'] == '066':
                rows = normalize_066(rows,expected)
            scores.append(dict(name=descriptor['name'],rows=rows,score=s3.s2.metrics(rows,expected),descriptor=descriptor))
        best = min(scores,key=lambda x:x['score']['macro_rmse'])
        require(best['descriptor'] == entry['old'], 'frozen strongest old comparator')
        result[setting] = dict(best=best,scoreboard=[dict(name=x['name'],score=x['score']) for x in scores],
                               validation=expected,training=evidence['train_observations'])
    return result


def make_model(data, name, spec):
    return models.build(name,data.h.shape[1],data.sources,len(next(iter(data.value['metadata']['vectors'].values()))),
                        hidden=spec['hidden']).to(data.device)


def episode_for(data, task, query, name, seed, spec, *, evaluation=False):
    return data.episode(task,query,cap=None if evaluation else spec['context_cap'],seed=seed,
                        no_functions=name=='FCR_NO_FUNCTIONS', stochastic_neighbors=name=='MODERNNCA' and not evaluation)


def predict(data, name, config, model, spec):
    if model is not None: model.eval()
    result = []
    with torch.no_grad():
        for task in data.targets:
            query = data.by[task,'validation']
            # All legal training context; fixed deterministic order, no query labels.
            e = episode_for(data,task,query,name,42,spec,evaluation=True)
            pred = models.closed_form(name,e,config) if name in models.ANALYTIC else model(e)
            s = data.value['scalers'][task]; pred = pred.double()*s['std']+s['mean']
            require(pred.shape == (len(query),1) and bool(torch.isfinite(pred).all()), 'finite validation prediction')
            result.extend(dict(data.rows[i],prediction=float(p)) for i,p in zip(query,pred[:,0].cpu()))
    expected = [r for r in data.rows if r['split']=='validation']
    return result, s3.s2.metrics(result,expected)


def train_case(data, job, out, commit, spec=SPEC):
    require(job in jobs(), 'fixed V10 job'); out.mkdir(parents=True,exist_ok=False)
    seed_everything(42,deterministic_algorithms=True)
    name = job['method']; analytic = name in models.ANALYTIC
    model = None if analytic else make_model(data,name,spec)
    optimizer = None if analytic else torch.optim.AdamW(model.parameters(),lr=spec['configurations'][job['config']],weight_decay=spec['weight_decay'])
    history = []; best = math.inf; updates = 0; coverage = {t:Counter() for t in data.tasks}
    identity = dict(task=TASK,commit=commit,job=job,spec=spec,cache_identity=data.value['identity'],
                    cache_rows_sha256=digest(data.rows),metadata=data.value['metadata'],
                    source_tasks=data.sources,target_tasks=data.targets)
    started = time.monotonic()
    for epoch in range(1, (1 if analytic else spec['epochs'])+1):
        trace = []
        schedule = [] if analytic else cache.epoch_schedule(data,epoch-1,target_only=name in models.TARGET_ONLY,batch_size=spec['query_batch'])
        for step,item in enumerate(schedule):
            model.train(); optimizer.zero_grad(set_to_none=True)
            records = []; group = [item['target']]+item['source']
            for j,(task,query) in enumerate(group):
                seed = 4200000+epoch*10000+step*100+j
                e = episode_for(data,task,query,name,seed,spec)
                loss = model.loss(e,data.y[query]); require(loss.ndim==0 and bool(torch.isfinite(loss)), 'finite training loss')
                weight = 1. if j==0 else spec['source_loss_weight']/len(item['source'])
                (loss*weight).backward(); coverage[task].update(query)
                records.append(dict(task=task,query=query,context=[r.sample_id for r in e.context_rows],loss=float(loss.detach()),weight=weight))
            params = [p for p in model.parameters() if p.grad is not None]
            require(params and all(bool(torch.isfinite(p.grad).all()) for p in params), 'finite gradients')
            norm = torch.nn.utils.clip_grad_norm_(params,spec['grad_clip'],error_if_nonfinite=True)
            optimizer.step(); updates += 1
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'finite model after update')
            trace.append(dict(episodes=records,gradient_norm=float(norm)))
        rows, score = predict(data,name,job['config'],model,spec)
        write(out/f'validation_{epoch:03d}.json',rows)
        entry = dict(epoch=epoch,score=score,updates=len(schedule),trace=trace)
        write(out/f'epoch_{epoch:03d}.json',entry); history.append(entry)
        if score['macro_rmse'] < best:
            best = score['macro_rmse']; chosen = epoch
            state = None if analytic else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            payload = dict(identity=identity,epoch=epoch,state=state,
                           optimizer=None if analytic else optimizer.state_dict(),updates=updates)
            with (out/'best.pending.pt').open('xb') as stream: torch.save(payload,stream)
            os.replace(out/'best.pending.pt',out/'best.pt')
        print(f'{job["id"]} epoch={epoch} macro_rmse={score["macro_rmse"]:.8f}',flush=True)
    counts = {t:dict(query_observations=sum(c.values()),unique_query_observations=len(c),
                    population=len(data.by[t,'train']),updates=sum(t==e['task'] for h in history for b in h['trace'] for e in b['episodes']))
              for t,c in coverage.items()}
    if not analytic:
        for t in data.targets:
            require(set(coverage[t])==set(data.by[t,'train']) and set(coverage[t].values())=={spec['epochs']}, 'full target coverage')
        if name not in models.TARGET_ONLY: require(all(counts[t]['updates'] >= spec['epochs'] for t in data.sources),'all source tasks each epoch')
    selected = read(out/f'validation_{chosen:03d}.json'); write(out/'selected_validation.json',selected)
    receipt = dict(identity=identity,best_epoch=chosen,selected=s3.s2.metrics(selected,[r for r in data.rows if r['split']=='validation']),
                   history_sha256=digest(history),checkpoint_sha256=sha(out/'best.pt'),updates=updates,coverage=counts,
                   epochs=len(history),analytic=analytic,parameter_count=0 if analytic else sum(p.numel() for p in model.parameters()),
                   elapsed_seconds=time.monotonic()-started,test_evaluated=False,scientific_acceptance='PENDING_REVIEW',
                   environment=dict(python=sys.version,torch=str(torch.__version__),device=str(data.device),
                                    cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if str(data.device).startswith('cuda') else None))
    write(out/'receipt.json',receipt); return receipt


def close_predictions(actual, expected, spec):
    require(len(actual)==len(expected),'replay prediction count')
    for a,b in zip(actual,expected):
        require(set(a)==set(b) and all(a[k]==v for k,v in b.items() if k!='prediction'),'replay row identity')
        require(math.isclose(a['prediction'],b['prediction'],rel_tol=spec['prediction_rtol'],abs_tol=spec['prediction_atol']),
                'independent checkpoint prediction differs')


def verify_case(data, job, out, commit, spec=SPEC):
    receipt = read(out/'receipt.json'); identity = receipt['identity']; name = job['method']
    require(identity == dict(task=TASK,commit=commit,job=job,spec=spec,cache_identity=data.value['identity'],
                            cache_rows_sha256=digest(data.rows),metadata=data.value['metadata'],
                            source_tasks=data.sources,target_tasks=data.targets), 'receipt identity')
    require(receipt['test_evaluated'] is False and receipt['scientific_acceptance']=='PENDING_REVIEW', 'no test/acceptance scope')
    count = 1 if name in models.ANALYTIC else spec['epochs']; require(receipt['epochs']==count,'complete epochs')
    histories=[]; seen={t:Counter() for t in data.tasks}; steps=0
    for epoch in range(1,count+1):
        h=read(out/f'epoch_{epoch:03d}.json'); histories.append(h)
        rows=read(out/f'validation_{epoch:03d}.json')
        require(h['score']==s3.s2.metrics(rows,[r for r in data.rows if r['split']=='validation']) and h['epoch']==epoch,'epoch metrics')
        schedule=[] if name in models.ANALYTIC else cache.epoch_schedule(data,epoch-1,target_only=name in models.TARGET_ONLY,batch_size=spec['query_batch'])
        require(h['updates']==len(h['trace'])==len(schedule),'full epoch update count')
        for step,(logged,item) in enumerate(zip(h['trace'],schedule)):
            group=[item['target']]+item['source']; require(len(logged['episodes'])==len(group),'episode trace count')
            require(type(logged['gradient_norm']) in (int,float) and math.isfinite(logged['gradient_norm']) and logged['gradient_norm']>=0,'gradient log')
            for j,(r,(task,query)) in enumerate(zip(logged['episodes'],group)):
                e=episode_for(data,task,query,name,4200000+epoch*10000+step*100+j,spec)
                require(r['task']==task and r['query']==query and r['context']==[x.sample_id for x in e.context_rows]
                        and math.isfinite(r['loss']) and r['weight']==(1. if j==0 else spec['source_loss_weight']/len(item['source'])), 'training schedule/identity')
                seen[task].update(query)
        steps+=len(schedule)
    require(digest(histories)==receipt['history_sha256'] and steps==receipt['updates'],'trace/updates binding')
    coverage={t:dict(query_observations=sum(c.values()),unique_query_observations=len(c),population=len(data.by[t,'train']),
                    updates=sum(t==e['task'] for h in histories for b in h['trace'] for e in b['episodes'])) for t,c in seen.items()}
    require(coverage==receipt['coverage'],'coverage recount')
    chosen=min(histories,key=lambda x:x['score']['macro_rmse'])['epoch']
    require(chosen==receipt['best_epoch'] and sha(out/'best.pt')==receipt['checkpoint_sha256'],'selection/checkpoint')
    selected=read(out/'selected_validation.json')
    require(selected==read(out/f'validation_{chosen:03d}.json') and receipt['selected']==histories[chosen-1]['score'],'selected prediction binding')
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    require(payload['identity']==identity and payload['epoch']==chosen and
            payload['updates']==sum(h['updates'] for h in histories[:chosen]),'checkpoint provenance')
    model=None if name in models.ANALYTIC else make_model(data,name,spec)
    if model is not None:
        model.load_state_dict(payload['state'],strict=True)
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'checkpoint finite')
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec['configurations'][job['config']],weight_decay=spec['weight_decay'])
        optimizer.load_state_dict(payload['optimizer'])
        require(all(int(x['step'])==payload['updates'] for x in optimizer.state.values()),'optimizer step count')
    else: require(payload['state'] is None and payload['optimizer'] is None,'analytic checkpoint')
    replay,_=predict(data,name,job['config'],model,spec); close_predictions(replay,selected,spec)
    return receipt


def grouped_difference(rows, reference, draws=2000):
    # Descriptive paired CI on reused development data, not independent confirmation.
    lookup={(r['task'],r['sample_id']):r for r in reference}; tasks=sorted({r['task'] for r in rows})
    groups=sorted({r['group'] for r in rows}); pos={g:i for i,g in enumerate(groups)}
    values={t:[] for t in tasks}
    for r in rows:
        b=lookup[r['task'],r['sample_id']]
        require(all(r[k]==b[k] for k in ('label','canonical','group','split')),'paired bootstrap population')
        values[r['task']].append((pos[r['group']],(r['prediction']-r['label'])**2,(b['prediction']-b['label'])**2))
    arrays={t:np.asarray(v) for t,v in values.items()}; rng=np.random.default_rng(4210); distribution=[]
    for _ in range(draws):
        weights=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups)); delta=[]
        for a in arrays.values():
            w=weights[a[:,0].astype(int)]
            if w.sum()==0: break
            delta.append(math.sqrt(float(w@a[:,1]/w.sum()))-math.sqrt(float(w@a[:,2]/w.sum())))
        if len(delta)==len(tasks): distribution.append(sum(delta)/len(delta))
    require(len(distribution)>=draws*.9,'insufficient grouped bootstrap replicates')
    return dict(delta_ci95=np.quantile(distribution,[.025,.975]).tolist(),draws=len(distribution),
                interpretation='descriptive_reused_development_not_confirmatory')


def compare(root, refs):
    output={}; passed=[]
    for setting in SETTINGS:
        selected={}
        for method in models.CANDIDATES+models.CONTROLS:
            options=[read(root/j['id']/'receipt.json') for j in jobs() if j['setting']==setting and j['method']==method]
            r=min(options,key=lambda x:x['selected']['macro_rmse']); job=r['identity']['job']
            selected[method]=dict(job=job,best_epoch=r['best_epoch'],score=r['selected'],
                                  rows=read(root/job['id']/'selected_validation.json'))
        controls={m:selected[m] for m in models.CONTROLS}
        strongest=min(controls,key=lambda m:controls[m]['score']['macro_rmse'])
        old=refs[setting]['best']; candidates={}
        for name in models.CANDIDATES:
            x=selected[name]; score=x['score']['macro_rmse']; old_score=old['score']['macro_rmse']; control_score=controls[strongest]['score']['macro_rmse']
            candidates[name]=dict(job=x['job'],best_epoch=x['best_epoch'],score=x['score'],
                delta_old=score-old_score,delta_matched=score-control_score,
                passes_point_gate=score < min(old_score,control_score),
                versus_old=grouped_difference(x['rows'],old['rows']),
                versus_matched=grouped_difference(x['rows'],controls[strongest]['rows']))
        output[setting]=dict(old_best=old['name'],old_score=old['score'],matched_best=strongest,
                             controls={m:{k:v for k,v in x.items() if k!='rows'} for m,x in controls.items()},candidates=candidates)
    for m in models.CANDIDATES:
        if all(output[s]['candidates'][m]['passes_point_gate'] for s in SETTINGS): passed.append(m)
    return dict(settings=output,development_candidates=passed,
                decision='BOUNDED_CONFIRMATION_REQUIRES_REVIEW' if passed else 'STOP_NO_UNIFIED_DEVELOPMENT_WINNER',
                uniform_superiority_established=False)


def gate_check(folder, commit, role):
    require(read(folder/'gate.json')==dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(folder/'tests.xml')),'V10 code gate identity')
    require(read(folder/'command.json')['exit_code']==0,'code gate exit')
    cases=ET.parse(folder/'tests.xml').findall('.//testcase')
    require(TEST_COUNT>0 and len(cases)==TEST_COUNT and all(not any(c.findall(t) for t in ('failure','error','skipped')) for c in cases)
            and {c.attrib['classname'] for c in cases}=={'tests.'+Path(t).stem for t in TESTS},'complete V10 code gate')


def code_gate(folder, commit, role):
    require(role in ('wsl','server'),'explicit gate role'); s3.s2.c0.check_code(REPO,commit)
    folder.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(folder/'pytest_tmp'),'--junitxml',str(folder/'tests.xml')]
    with (folder/'tests.log').open('xb') as f:
        p=subprocess.run(argv,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    write(folder/'command.json',dict(argv=argv,exit_code=p.returncode)); require(p.returncode==0,'V10 code tests failed')
    write(folder/'gate.json',dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(folder/'tests.xml')))
    gate_check(folder,commit,role)


def command(mode, name, a):
    return [sys.executable,str(Path(__file__).resolve()),'_worker','--mode',mode,'--name',name,
            '--output',str(a.output),'--commit',a.commit,'--split-manifest',str(a.split_manifest),'--source-lock',str(a.source_lock)]


def check_launch(a):
    s3.s2.c0.check_code(REPO,a.commit); v=read(a.output/'launch.json')
    require(v['task']==TASK and v['commit']==a.commit and v['spec']==SPEC and v['jobs']==jobs()
            and v['split_sha256']==sha(a.split_manifest) and v['source_lock_sha256']==sha(a.source_lock)
            and v['reference_lock_sha256']==sha(LOCK), 'launch inputs/commit')
    require(read(REGISTRY/'attempt.json')==dict(task=TASK,commit=a.commit,output=str(a.output.resolve())),'one-shot launch claim')
    return v


def audit_cache(factory,trainer,source,setting,value,device='cpu'):
    sampled={t:sorted(random_source_indices(source.counts[t],i)) for i,t in enumerate(source.tasks)}
    fresh=cache.extract(factory,trainer,source,setting,device,source_indices=sampled)
    require(fresh['identity']==value['identity'] and fresh['metadata']==value['metadata']
            and fresh['scalers']==value['scalers'],'fresh feature provenance/scalers')
    index={(r['task'],r['sample_id']):i for i,r in enumerate(value['rows'])}
    result=dict(value); result['h']=value['h'].clone(); result['functions']=value['functions'].clone()
    for j,r in enumerate(fresh['rows']):
        i=index[r['task'],r['sample_id']]; require(r==value['rows'][i],'fresh feature row')
        for key in ('h','functions'):
            require(torch.allclose(fresh[key][j],value[key][i],rtol=.001,atol=.00002),'raw graph feature replay: '+key)
            result[key][i]=fresh[key][j]
    packed=cache.pack({k:v for k,v in value.items() if k not in ('mean','scale','commit','schema','statistics_population_sha256')},value['commit'])
    require(torch.equal(packed['mean'],value['mean']) and torch.equal(packed['scale'],value['scale'])
            and packed['statistics_population_sha256']==value['statistics_population_sha256'],'train-only feature statistics')
    return result,dict(target_rows=sum(r['task'] in value['targets'] for r in fresh['rows']),
                       source_rows=sum(map(len,sampled.values())),source_rule='two_fixed_random_rows_per_task',
                       full_source_raw_replay=False,all_feature_file_sha_verified=True)


def random_source_indices(n,i):
    import random
    return random.Random(428000+i).sample(range(n),min(n,SPEC['source_cache_audit_per_task']))


def worker(a):
    seed_everything(42,deterministic_algorithms=True)
    launch=check_launch(a)
    if a.mode=='verify':
        require(a.name in SETTINGS and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU verification scope')
        torch.set_num_threads(1); folder=a.output/'cache'/a.name
        value=cache.load_cache(folder,a.commit); check_binding(value,read(LOCK)['settings'][a.name])
        f,t,source=s3.context(a.name,a.split_manifest,a.source_lock,'cpu')
        audited,audit=audit_cache(f,t,source,a.name,value); data=cache.Data(audited)
        receipts=[]
        for job in jobs():
            if job['setting']==a.name: receipts.append(verify_case(data,job,a.output/job['id'],a.commit))
        write(a.output/('verified_'+a.name+'.json'),dict(task=TASK,commit=a.commit,setting=a.name,pid=os.getpid(),
              content_status='PASS',device='cpu',new_optimizer_updates=0,raw_cache_audit=audit,
              receipts={r['identity']['job']['id']:sha(a.output/r['identity']['job']['id']/'receipt.json') for r in receipts}))
        return
    assignment=read(a.output/(a.mode+'_'+a.name+'.assignment.json'))
    require(assignment['gpu'] in launch['gpus'] and assignment['uuid']==launch['gpu_uuids'][str(assignment['gpu'])]
            and os.environ.get('CUDA_VISIBLE_DEVICES')==assignment['uuid']
            and torch.cuda.is_available() and torch.cuda.device_count()==1,'single authorized GPU assignment')
    if a.mode=='cache':
        require(a.name in SETTINGS,'cache setting'); f,t,source=s3.context(a.name,a.split_manifest,a.source_lock,'cuda:0')
        entry=read(LOCK)['settings'][a.name]
        raw=cache.extract(f,t,source,a.name,'cuda:0'); check_binding(raw,entry)
        path=a.output/'references/084'/entry['base']/'validation_observations.json'
        require(sha(path)==entry['files']['validation_observations']
                and [r for r in raw['rows'] if r['split']=='validation']==read(path),'fresh validation matches references')
        cache.save_cache(raw,a.output/'cache'/a.name,a.commit)
    else:
        require(a.mode=='train','training mode'); job=next(j for j in jobs() if j['id']==a.name)
        value=cache.load_cache(a.output/'cache'/job['setting'],a.commit); check_binding(value,read(LOCK)['settings'][job['setting']])
        train_case(cache.Data(value,'cuda:0'),job,a.output/job['id'],a.commit)


def verify(a):
    check_launch(a)
    for role in ('wsl','server'): gate_check(a.output/(role+'_evidence'),a.commit,role)
    refs=references(a.output/'references/084',a.output/'references/066')
    for job in jobs(): require(read(a.output/('train_'+job['id']+'.command.json'))['exit_code']==0,'training worker exit')
    for setting in SETTINGS:
        require(read(a.output/('cache_'+setting+'.command.json'))['exit_code']==0,'cache worker exit')
        orchestration.call_worker(command('verify',setting,a),a.output/('verify_'+setting),
                                  dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    checked=[read(a.output/('verified_'+s+'.json')) for s in SETTINGS]
    require(all(v['commit']==a.commit and v['content_status']=='PASS' and v['new_optimizer_updates']==0 for v in checked)
            and len({v['pid'] for v in checked})==3,'three independent scene verification processes')
    result=dict(task=TASK,commit=a.commit,content_status='PASS',scientific_acceptance='PENDING_REVIEW',
                candidates=10,controls=4,jobs=len(jobs()),verification=checked,**compare(a.output,refs))
    write(a.output/'verification.json',result); return result


def run(a):
    from p1d4_batch import free_gpus
    s3.s2.c0.check_code(REPO,a.commit)
    for role in ('wsl','server'): gate_check(getattr(a,role+'_evidence'),a.commit,role)
    references(a.reference_root,a.reference066_root)
    require(not a.output.exists() and not REGISTRY.exists(),'existing run/attempt; preserve and report')
    require(a.gpus and len(set(a.gpus))==len(a.gpus) and set(a.gpus)<={0,1,2,3},'GPU scope')
    for path in (a.reference_root,a.reference066_root,a.wsl_evidence,a.server_evidence):
        require(not a.output.resolve().is_relative_to(path.resolve()) and not path.resolve().is_relative_to(a.output.resolve()),'output overlaps input')
    available=free_gpus(a.gpus); require(available,'no free authorized GPU')
    a.output.mkdir(parents=True); REGISTRY.mkdir(parents=True)
    write(REGISTRY/'attempt.json',dict(task=TASK,commit=a.commit,output=str(a.output.resolve())))
    uuids={str(g):s3.s2.c0.gpu_uuid(g) for g in available}
    write(a.output/'launch.json',dict(task=TASK,commit=a.commit,spec=SPEC,jobs=jobs(),gpus=available,gpu_uuids=uuids,
          split_sha256=sha(a.split_manifest),source_lock_sha256=sha(a.source_lock),reference_lock_sha256=sha(LOCK)))
    try:
        references(a.reference_root,a.reference066_root,a.output/'references')
        for role in ('wsl','server'):
            shutil.copytree(getattr(a,role+'_evidence'),a.output/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
        def execute(item,gpu):
            mode,name=item; require(free_gpus([gpu])==[gpu],'assigned GPU became busy')
            write(a.output/(mode+'_'+name+'.assignment.json'),dict(gpu=gpu,uuid=uuids[str(gpu)]))
            orchestration.call_worker(command(mode,name,a),a.output/(mode+'_'+name),dict(os.environ,
                CUDA_VISIBLE_DEVICES=uuids[str(gpu)],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        orchestration.dispatch([('cache',s) for s in SETTINGS],available,execute)
        orchestration.dispatch([('train',j['id']) for j in jobs()],available,execute)
        verify(a)
    except BaseException as exc:
        write(a.output/'failed.json',dict(task=TASK,commit=a.commit,error_type=type(exc).__name__,reason=str(exc))); raise


def package(root,commit):
    require(root.is_dir() and read(root/'launch.json')['commit']==commit,'package run identity')
    target=root.with_name(root.name+'.zip')
    require(not target.exists() and not Path(str(target)+'.sha256').exists(),'existing package')
    files={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'package symlink')
        if p.is_file():
            require(p.suffix in ('.json','.log','.xml','.pt'),'unexpected package file')
            files[p.relative_to(root).as_posix()]=sha(p)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(root/name,name)
        z.writestr('checksums.sha256',''.join(h+'  '+n+'\n' for n,h in files.items()))
    with zipfile.ZipFile(target) as z:
        require(set(z.namelist())==set(files)|{'checksums.sha256'} and z.testzip() is None,'package members/CRC')
        for name,h in files.items(): require(hashlib.sha256(z.read(name)).hexdigest()==h,'package content SHA')
    with Path(str(target)+'.sha256').open('x',encoding='utf8') as f: f.write(sha(target)+'  '+target.name+'\n')
    return dict(archive=str(target),sha256=sha(target),scientific_acceptance='NOT_ASSESSED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('code-gate','run','verify','package','reference-check','_worker'))
    p.add_argument('--commit',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=('wsl','server')); p.add_argument('--mode',choices=('cache','train','verify')); p.add_argument('--name')
    p.add_argument('--gpus',nargs='+',type=int,choices=range(4))
    for name in ('split-manifest','source-lock','reference-root','reference066-root','wsl-evidence','server-evidence'): p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.action=='code-gate': code_gate(a.output,a.commit,a.role)
    elif a.action=='reference-check':
        s3.s2.c0.check_code(REPO,a.commit)
        values=references(a.reference_root,a.reference066_root)
        print({s:dict(name=x['best']['name'],score=x['best']['score']) for s,x in values.items()})
    elif a.action=='run': run(a)
    elif a.action=='_worker': worker(a)
    elif a.action=='verify': verify(a)
    else:
        s3.s2.c0.check_code(REPO,a.commit); print(package(a.output,a.commit))


if __name__=='__main__': main()
