"""V9-S3: nine paired target-plus-source runs, with a four-update real smoke."""
from copy import deepcopy
from pathlib import Path
import argparse
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch

import v9_s2_screen as s2
import v9_joint_source as aux
from loss import QuantileRegressionLoss
from reproducibility import seed_everything, state_dict_sha256

REPO = Path(__file__).resolve().parent
TASK = 'V9_S3_PAIRED_JOINT_SCREEN_20260930'
REGISTRY = '.tmp/v9s3_attempt_20260930'
LOCK = REPO/'configs/v9_s3_reference_lock.json'
METHODS = ('B1_1E4','B1_1E3','REFINE_GRAPH')
SPEC = dict(s2.SPEC, source_training=True, source_pretraining=False, auxiliary_weight=1.,
            auxiliary_sampling='balanced_shuffled_task_cycles_without_replacement_within_task',
            auxiliary_rng='forked_torch_seed_42000000_plus_step', optimizer_steps_per_pair=1)
SMOKE = (('ToxAcute','B1_1E4'),('A','REFINE_GRAPH'))
TESTS = ('tests/test_v9_s3_screen.py',*s2.TESTS)
TEST_COUNT = 155
read, write, sha, digest, require = s2.read,s2.write,s2.sha,s2.digest,s2.require


def jobs():
    return [dict(id=f'{s}_{m}_s42',setting=s,method=m,seed=42) for s in s2.c0.SETTINGS for m in METHODS]


def references(root):
    lock = read(LOCK); results = {}
    require(lock['schema'] == 'v9_s3_077_reference_v1', 'reference schema')
    for job in s2.jobs():
        entry = lock['jobs'][job['id']]; folder = root/job['id']
        for name, expected_sha in entry['files'].items():
            require(sha(folder/name) == expected_sha, '077 reference file '+job['id']+'/'+name)
        identity = read(folder/'identity.json'); expected = read(folder/'validation_observations.json')
        rows = read(folder/'selected_validation.json'); receipt = read(folder/'receipt.json')
        require(receipt['identity'] == identity and receipt['job'] == job and receipt['commit'] == lock['commit']
                and receipt['selected'] == s2.metrics(rows,expected), '077 reference metrics/identity')
        results[job['id']] = dict(identity=identity,selected=receipt['selected'],rows=rows,
                                  initial=read(folder/'initial_validation.json'),best_epoch=receipt['best_epoch'])
    return results


def copy_references(source, destination):
    references(source)
    destination.mkdir(exist_ok=False)
    for job, entry in read(LOCK)['jobs'].items():
        (destination/job).mkdir()
        for name in entry['files']: shutil.copyfile(source/job/name,destination/job/name)


def paired_initial(rows, old):
    require(len(rows) == len(old), 'paired initial population')
    for row, reference in zip(rows,old):
        require(set(row) == set(reference) and all(row[k] == v for k,v in reference.items() if k != 'prediction')
                and math.isclose(row['prediction'],reference['prediction'],rel_tol=1e-6,abs_tol=1e-6),
                'paired initial predictions differ')


def build(base, source, method, counts, setting, device):
    require(method in METHODS and not set(counts) & set(source.tasks), 'joint model task scope')
    expanded = deepcopy(base)
    expanded.decoders.update(deepcopy(source.heads))
    expanded.task_name = list(base.task_name)+source.tasks; expanded.task_num = len(expanded.task_name)
    return s2.build(expanded,method,dict(counts,**source.counts),None,setting,device)


def step(trainer, setting, batch, task, epoch, source, item, optimizer, device, *, auxiliary=True):
    """One target backward, one isolated source backward, one common clip/step."""
    model = trainer.model; optimizer.zero_grad(set_to_none=True)
    if setting == 'ToxAcute': target_loss,_ = trainer._training_step(batch,task,epoch)
    else:
        scaler = trainer.scalers[task]
        target_loss = QuantileRegressionLoss().compute_loss(model(batch,task_name=task)[task],
            (batch.y.reshape(-1,1)-scaler['mean'])/scaler['std'])
    if getattr(model,'method',None) == 'REFINE_GRAPH': target_loss = target_loss+model.regularization()
    require(torch.isfinite(target_loss).item(), 'finite target loss'); target_loss.backward()
    auxiliary_loss = None
    if auxiliary:
        # Restores both CPU and selected CUDA RNG. Data/schedule RNGs are local.
        with torch.random.fork_rng(devices=[0] if str(device).startswith('cuda') else []):
            torch.manual_seed(42000000+item['step'])
            ab = source.batch(item['task'],item['indices'],device); scaler = source.scalers[item['task']]
            auxiliary_loss = QuantileRegressionLoss().compute_loss(model(ab,task_name=item['task'])[item['task']],
                (ab.y.reshape(-1,1)-scaler['mean'])/scaler['std'])
            require(torch.isfinite(auxiliary_loss).item(), 'finite auxiliary loss'); auxiliary_loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(target=float(target_loss.detach()),auxiliary=None if auxiliary_loss is None else float(auxiliary_loss.detach()),
                gradient_norm=float(norm))


def optimizer_steps(model, optimizer):
    return {n:int(optimizer.state[p]['step']) for n,p in model.named_parameters() if p in optimizer.state}


def identity_for(trainer, source, job, training, validation, commit, spec):
    return dict(task=TASK,job=job,commit=commit,spec=spec,contract_sha256=digest(s2.c0.contract(job['setting'])),
                initial_encoder=state_dict_sha256(trainer.model.encoder),initial_target_heads=state_dict_sha256(trainer.model.decoders),
                source=source.identity,train_sha256=digest(training),validation_sha256=digest(validation))


def train_one(factory, trainer, source, job, out, commit, device, reference, spec=SPEC):
    require(job in jobs(), 'job outside nine-run matrix')
    setting,method = job['setting'],job['method']; tasks,counts = s2.c0.tasks_and_counts(setting)
    require({t:len(d) for t,d in s2.datasets_for(factory,trainer,setting)['train'].items()} == counts, 'target counts')
    training = s2.observations(factory,trainer,setting,'train'); validation = s2.observations(factory,trainer,setting,'validation')
    require(not {r['canonical'] for r in source.rows} & {r['canonical'] for r in validation}
            and not {r['group'] for r in source.rows} & {r['group'] for r in validation}, 'auxiliary/target validation overlap')
    identity = identity_for(trainer,source,job,training,validation,commit,spec)
    old = reference['identity']
    require(identity['initial_encoder'] == read(s2.c2b.LOCK)['initial_encoders'][setting] == old['initial_encoder']
            and identity['initial_target_heads'] == old['initial_heads']
            and all(identity[k] == old[k] for k in ('contract_sha256','train_sha256','validation_sha256')), '077 pairing')
    seed_everything(42,deterministic_algorithms=True)
    trainer.model,opt = build(trainer.model,source,method,counts,setting,device); model = trainer.model
    for name,value in [('identity',identity),('train_observations',training),('validation_observations',validation),
                       ('source_train_observations',source.rows)]: write(out/(name+'.json'),value)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    rows,score = s2.evaluate(factory,trainer,setting,None,validation,device,0); paired_initial(rows,reference['initial'])
    write(out/'initial_validation.json',rows)
    schedule = aux.Schedule(source.tasks,source.counts,s2.c0.BATCH_SIZES[setting])
    history=[]; all_aux=[]; best=math.inf; updates=0; started=time.monotonic()
    for epoch in range(spec['max_epochs']):
        phase = s2.begin_epoch(model,method,epoch,spec)
        target_plan = s2.epoch_plan(tasks,counts,s2.c0.BATCH_SIZES[setting],epoch)
        auxiliary_plan=[]; losses=[]
        for item in target_plan:
            a = schedule.next(); auxiliary_plan.append(a); all_aux.append(a)
            batch = s2.batch_for(factory,trainer,setting,'train',item['task'],item['indices'],device)
            losses.append(step(trainer,setting,batch,item['task'],epoch,source,a,opt,device)); updates += 1
        if phase == 'HEAD_WARMUP':
            require(all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items()
                        if not k.startswith('decoders.')), 'warmup changed shared parameters')
        if method == 'REFINE_GRAPH': require(state_dict_sha256(model.encoder) == identity['initial_encoder'], 'source anchor drift')
        rows,score = s2.evaluate(factory,trainer,setting,None,validation,device,epoch)
        h = dict(epoch=epoch+1,validation=score,updates=len(target_plan),total_updates=updates,
                 schedule_sha256=digest(target_plan),auxiliary_schedule=auxiliary_plan,losses=losses,
                 phase=s2.phase_identity(model,method,phase),optimizer_steps=optimizer_steps(model,opt))
        write(out/f'validation_epoch_{epoch+1:03d}.json',rows); write(out/f'epoch_{epoch+1:03d}.json',h); history.append(h)
        if score['macro_rmse'] < best:
            best = score['macro_rmse']; best_epoch = epoch+1
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            with (out/'best.pending.pt').open('xb') as stream:
                torch.save(dict(identity=identity,epoch=best_epoch,model_state=state,optimizer_state=opt.state_dict()),stream)
            os.replace(out/'best.pending.pt',out/'best.pt')
        print(f'{job["id"]} epoch={epoch+1} target_macro_rmse={score["macro_rmse"]:.8f} best_epoch={best_epoch}',flush=True)
    require(s2.select(history,spec)[0] == best_epoch, 'selection')
    final_encoder = state_dict_sha256(model.encoder)
    coverage = aux.coverage(source,all_aux); require(all(r['updates'] > 0 for r in coverage.values()), 'unvisited auxiliary task')
    write(out/'auxiliary_coverage.json',coverage)
    payload = torch.load(out/'best.pt',map_location='cpu',weights_only=True); model.load_state_dict(payload['model_state'],strict=True)
    rows,score = s2.evaluate(factory,trainer,setting,None,validation,device,best_epoch-1)
    require(rows == read(out/f'validation_epoch_{best_epoch:03d}.json'), 'selected checkpoint replay')
    write(out/'selected_validation.json',rows)
    result = dict(task=TASK,job=job,commit=commit,identity=identity,history=history,best_epoch=best_epoch,updates=updates,
                  selected=score,checkpoint_sha256=sha(out/'best.pt'),selected_state_sha256=state_dict_sha256(model),
                  selected_encoder=state_dict_sha256(model.encoder),final_encoder=final_encoder,
                  checkpoint_validation_replay=True,test_evaluated=False,scientific_acceptance='PENDING_REVIEW',
                  auxiliary_batches=len(all_aux),target_batches=updates,coverage=coverage,elapsed_seconds=time.monotonic()-started,
                  environment=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,device=str(device),
                                   gpu_name=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None))
    write(out/'receipt.json',result); return result


def check_optimizer(model, opt, actual, recorded, target_plan, auxiliary_plan, warming_steps, method):
    require(set(actual) == {'state','param_groups'} and actual['param_groups'] == opt.state_dict()['param_groups'], 'optimizer groups')
    parameters = [p for g in opt.param_groups for p in g['params']]; names = {id(p):n for n,p in model.named_parameters()}
    task_steps = {t:sum(r['task'] == t for r in target_plan+auxiliary_plan) for t in model.task_name}
    expected = {}
    for i,p in enumerate(parameters):
        n = names[id(p)]; count = task_steps[n.split('.')[1]] if n.startswith('decoders.') else len(target_plan)-warming_steps
        # The unused bond representation is intentionally outside the forward path.
        mode = model.encoder.backbone.edge_bias_mode
        if (mode == 'path' and '.direct_bond_embeddings.' in '.'+n) or (mode == 'direct' and '.path_bond_embeddings.' in '.'+n): count=0
        if count: expected[i]=(n,count,p)
    require(set(actual['state']) == set(expected), 'optimizer parameter coverage')
    require(recorded == {n:count for n,count,p in expected.values()}, 'optimizer step accounting')
    for i,(n,count,p) in expected.items():
        value = actual['state'][i]
        require(set(value) == {'step','exp_avg','exp_avg_sq'} and float(value['step']) == count
                and all(torch.isfinite(v).all().item() for v in value.values())
                and all(value[k].shape == p.shape and value[k].dtype == p.dtype for k in ('exp_avg','exp_avg_sq')), 'Adam state '+n)


def verify_job(factory, trainer, source, job, out, commit, reference, spec=SPEC):
    r = read(out/'receipt.json'); setting=job['setting']; tasks,counts=s2.c0.tasks_and_counts(setting)
    training=s2.observations(factory,trainer,setting,'train'); validation=s2.observations(factory,trainer,setting,'validation')
    identity=identity_for(trainer,source,job,training,validation,commit,spec)
    require(r['identity'] == read(out/'identity.json') == identity and r['task'] == TASK and r['job'] == job and r['commit'] == commit, 'receipt identity')
    for name,value in [('train_observations',training),('validation_observations',validation),('source_train_observations',source.rows)]:
        require(read(out/(name+'.json')) == value, 'trusted '+name)
    old=reference['identity']
    require(identity['initial_target_heads'] == old['initial_heads'] and identity['initial_encoder'] == old['initial_encoder']
            == read(s2.c2b.LOCK)['initial_encoders'][setting] and all(identity[k] == old[k] for k in
            ('contract_sha256','train_sha256','validation_sha256')), '077 pairing')
    paired_initial(read(out/'initial_validation.json'),reference['initial'])
    schedule=aux.Schedule(source.tasks,source.counts,s2.c0.BATCH_SIZES[setting]); ap=[]; tp=[]
    history=r['history']; best,_=s2.select(history,spec)
    for epoch,h in enumerate(history):
        plan=s2.epoch_plan(tasks,counts,s2.c0.BATCH_SIZES[setting],epoch); aq=[schedule.next() for _ in plan]; tp+=plan;ap+=aq
        require(h == read(out/f'epoch_{epoch+1:03d}.json') and h['schedule_sha256'] == digest(plan) and h['auxiliary_schedule'] == aq
                and h['updates'] == len(plan) and h['total_updates'] == len(tp), 'paired update accounting')
        require(h['validation'] == s2.metrics(read(out/f'validation_epoch_{epoch+1:03d}.json'),validation), 'independent metrics')
        warming=job['method'] == 'REFINE_GRAPH' and epoch < spec['warmup_epochs']; phase=h['phase']
        require(phase['phase'] == ('HEAD_WARMUP' if warming else 'JOINT') and phase['encoder_trainable'] == (job['method'] != 'REFINE_GRAPH')
                and phase['adaptive_trainable'] is False, 'training phase')
        if job['method'] == 'REFINE_GRAPH': require(phase['encoder_sha256'] == identity['initial_encoder'], 'frozen source')
        require(len(h['losses']) == len(plan) and all(type(v) in (int,float) and math.isfinite(v) and v >= 0
                for loss in h['losses'] for v in loss.values()), 'finite recorded losses/norm')
    require(r['best_epoch'] == best and r['updates'] == r['target_batches'] == r['auxiliary_batches'] == len(tp), 'total updates/selection')
    require(r['coverage'] == read(out/'auxiliary_coverage.json') == aux.coverage(source,ap), 'auxiliary coverage')
    rows=read(out/'selected_validation.json')
    require(rows == read(out/f'validation_epoch_{best:03d}.json') and r['selected'] == s2.metrics(rows,validation), 'selected metrics')
    require(sha(out/'best.pt') == r['checkpoint_sha256'], 'checkpoint bytes')
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    require(set(payload) == {'identity','epoch','model_state','optimizer_state'} and payload['identity'] == identity and payload['epoch'] == best, 'checkpoint binding')
    model,opt=build(trainer.model,source,job['method'],counts,setting,'cpu'); initial=model.state_dict(); actual=payload['model_state']
    require(set(actual) == set(initial), 'state keys')
    for k,v in initial.items():
        x=actual[k]
        require(isinstance(x,torch.Tensor) and x.shape == v.shape and x.dtype == v.dtype and torch.isfinite(x).all().item(), 'tensor '+k)
        if k in dict(model.named_buffers()) or (job['method'] == 'REFINE_GRAPH' and (k.startswith('encoder.') or
                (best <= spec['warmup_epochs'] and not k.startswith('decoders.')))):
            require(torch.equal(x.cpu(),v.cpu()), 'protected tensor '+k)
    per_epoch=len(tp)//len(history); n=per_epoch*best
    check_optimizer(model,opt,payload['optimizer_state'],history[best-1]['optimizer_steps'],tp[:n],ap[:n],
                    min(best,spec['warmup_epochs'])*per_epoch if job['method'] == 'REFINE_GRAPH' else 0,job['method'])
    model.load_state_dict(actual,strict=True)
    require(state_dict_sha256(model) == r['selected_state_sha256'] and state_dict_sha256(model.encoder) == r['selected_encoder'], 'selected tensors')
    if job['method'] == 'REFINE_GRAPH': require(r['final_encoder'] == r['selected_encoder'] == identity['initial_encoder'], 'source drift')
    require(r['checkpoint_validation_replay'] is True and r['test_evaluated'] is False and r['scientific_acceptance'] == 'PENDING_REVIEW', 'scope')
    return r


def compare(results, old):
    require({r['job']['id'] for r in results} == {j['id'] for j in jobs()} and len(results) == 9, 'nine results required')
    values={r['job']['id']:r for r in results}; records=[]
    for r in results:
        j=r['job']; s=j['setting']; score=r['selected']['macro_rmse']; own=old[j['id']]['selected']
        control={m:values[f'{s}_{m}_s42']['selected']['macro_rmse'] for m in METHODS[:2]}
        previous={m:old[f'{s}_{m}_s42']['selected']['macro_rmse'] for m in s2.METHODS}
        records.append(dict(job=j,best_epoch=r['best_epoch'],selected=r['selected'],paired_077=own,
            paired_delta=score-own['macro_rmse'],endpoint_deltas={t:v['rmse']-own['endpoints'][t]['rmse'] for t,v in r['selected']['endpoints'].items()},
            delta_vs_joint_B1={m:score-v for m,v in control.items()},all_077_macro_rmse=previous,
            endpoint_deltas_vs_joint_B1={m:{t:v['rmse']-values[f'{s}_{m}_s42']['selected']['endpoints'][t]['rmse']
                for t,v in r['selected']['endpoints'].items()} for m in METHODS[:2]},
            delta_vs_best_077=score-min(previous.values()),relative_vs_best_077=score/min(previous.values())-1))
    graph=[r for r in records if r['job']['method'] == 'REFINE_GRAPH']
    signal=all(r['delta_vs_best_077'] < 0 and all(v < 0 for v in r['delta_vs_joint_B1'].values()) for r in graph)
    return dict(comparisons=records,refine_all_scene_candidate_signal=signal,
                worst_relative_vs_077={m:max(r['relative_vs_best_077'] for r in records if r['job']['method'] == m) for m in METHODS},
                scope='SINGLE_SEED_VALIDATION_SCREEN_ONLY',unified_superiority_confirmed=False)


def smoke(factory, trainer, source, setting, method, out, commit, device):
    require((setting,method) in SMOKE, 'smoke scope')
    tasks,counts=s2.c0.tasks_and_counts(setting); seed_everything(42,deterministic_algorithms=True)
    trainer.model,opt=build(trainer.model,source,method,counts,setting,device);model=trainer.model
    s2.begin_epoch(model,method,5); before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    schedule=aux.Schedule(source.tasks,source.counts,s2.c0.BATCH_SIZES[setting]); rows=[]
    # Source-only smoke proves auxiliary supervision reaches the common representation.
    for i in range(2):
        item=schedule.next(); batch=source.batch(item['task'],item['indices'],device);scaler=source.scalers[item['task']]
        opt.zero_grad(set_to_none=True)
        loss=QuantileRegressionLoss().compute_loss(model(batch,task_name=item['task'])[item['task']],(batch.y.reshape(-1,1)-scaler['mean'])/scaler['std'])
        require(torch.isfinite(loss).item(),'smoke finite loss');loss.backward()
        gradients={scope:sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters()
                   if n.startswith(scope+'.') and p.grad is not None) for scope in ('encoder','target','fusion','decoders')}
        require(gradients['decoders'] > 0 and (gradients['encoder'] > 0 if method.startswith('B1') else
                gradients['encoder'] == 0 and gradients['fusion'] > 0 and (i == 0 or gradients['target'] > 0)), 'auxiliary gradient route')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
        rows.append(dict(schedule=item,loss=float(loss.detach()),gradient_l1=gradients))
    require(any(not torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items() if not k.startswith('decoders.')), 'smoke shared parameters did not change')
    if method == 'REFINE_GRAPH':require(all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items() if k.startswith('encoder.')), 'smoke anchor drift')
    write(out/'receipt.json',dict(task=TASK,commit=commit,setting=setting,method=method,updates=2,
                                 source=source.identity,rows=rows,scope='DISCARDED_AUXILIARY_PATH_SMOKE',content_status='PASS'))


def gate_check(root, commit, role):
    require(read(root/'gate.json') == dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')), 'code gate identity')
    require(read(root/'command.json')['exit_code'] == 0, 'code gate failed')
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases) == TEST_COUNT and all(not any(c.findall(t) for t in ('error','failure','skipped')) for c in cases)
            and {c.attrib['classname'] for c in cases} == {'tests.'+Path(t).stem for t in TESTS}, 'complete passing gate required')


def code_gate(root, commit, role):
    require(role in ('wsl','server'),'gate role');s2.c0.check_code(REPO,commit);root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(root/'pytest_tmp'),'--junitxml',str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as stream:
        p=subprocess.run(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json',dict(argv=argv,exit_code=p.returncode));require(p.returncode == 0,'code tests failed')
    write(root/'gate.json',dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')));gate_check(root,commit,role)


def context(setting, split, source_lock, device):
    f,t=s2.factory_for(setting,split,source_lock,device)
    return f,t,aux.load(f,t,setting,REPO,source_lock)


def verify(root, commit, split, source_lock):
    launch=read(root/'launch.json')
    require(launch['task'] == TASK and launch['commit'] == commit and launch['spec'] == SPEC and launch['jobs'] == jobs(), 'launch identity')
    for role in ('wsl','server'):gate_check(root/(role+'_evidence'),commit,role)
    old=references(root/'reference_077');results=[]
    for setting in s2.c0.SETTINGS:
        f,t,source=context(setting,split,source_lock,'cpu')
        for job in [j for j in jobs() if j['setting'] == setting]:
            results.append(verify_job(f,t,source,job,root/job['id'],commit,old[job['id']]))
    for setting,method in SMOKE:
        r=read(root/f'smoke_{setting}_{method}'/'receipt.json')
        require(r['commit'] == commit and r['task'] == TASK and r['setting'] == setting and r['method'] == method
                and r['updates'] == 2 and r['content_status'] == 'PASS', 'real smoke receipt')
    return dict(task=TASK,commit=commit,content_status='PASS',scientific_acceptance='PENDING_REVIEW',
                total_epochs=sum(len(r['history']) for r in results),total_updates=sum(r['updates'] for r in results),smoke_updates=4,
                **compare(results,old))


def run(a):
    from p1d4_batch import free_gpus
    s2.c0.check_code(REPO,a.commit)
    for role in ('wsl','server'):gate_check(getattr(a,role+'_evidence'),a.commit,role)
    references(a.reference_root);require(free_gpus([a.gpu]) == [a.gpu],'GPU busy')
    root=a.output;root.mkdir(parents=True,exist_ok=False)
    registry=REPO/REGISTRY;registry.mkdir(parents=True,exist_ok=True)
    write(registry/'attempt.json',dict(task=TASK,commit=a.commit,output=str(root.resolve()),jobs=jobs()))
    uuid=s2.c0.gpu_uuid(a.gpu)
    write(root/'launch.json',dict(task=TASK,commit=a.commit,spec=SPEC,jobs=jobs(),gpu_uuid=uuid,gpu=a.gpu,smoke=[list(v) for v in SMOKE]))
    try:
        for role in ('wsl','server'):shutil.copytree(getattr(a,role+'_evidence'),root/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
        copy_references(a.reference_root,root/'reference_077')
        dispatch=[('smoke',dict(setting=s,method=m,id=f'smoke_{s}_{m}')) for s,m in SMOKE]+[('train',j) for j in jobs()]
        for mode,job in dispatch:
            require(free_gpus([a.gpu]) == [a.gpu],'GPU occupied')
            argv=[sys.executable,str(Path(__file__).resolve()),'_worker','--mode',mode,'--job',job['id'],'--commit',a.commit,
                  '--output',str(root),'--split-manifest',str(a.split_manifest),'--source-lock',str(a.source_lock)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/(job['id']+'.log')).open('xb') as log:p=subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(root/(job['id']+'.command.json'),dict(argv=argv,exit_code=p.returncode));require(p.returncode == 0,'job failed; stop and preserve evidence')
        write(root/'verification.json',verify(root,a.commit,a.split_manifest,a.source_lock))
    except BaseException as exc:
        write(root/'failed.json',dict(task=TASK,error_type=type(exc).__name__,reason=str(exc)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('code-gate','run','_worker','verify','package'))
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=('wsl','server'))
    p.add_argument('--gpu',type=int,choices=range(4));p.add_argument('--mode',choices=('smoke','train'));p.add_argument('--job')
    for name in ('split-manifest','source-lock','wsl-evidence','server-evidence','reference-root'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.action == 'code-gate':code_gate(a.output,a.commit,a.role)
    elif a.action == 'package':print(s2.package(a.output,a.commit))
    elif a.action == 'run':
        require(all(getattr(a,k) is not None for k in ('gpu','split_manifest','source_lock','wsl_evidence','server_evidence','reference_root')),'runtime arguments');run(a)
    else:
        s2.c0.check_code(REPO,a.commit);require(a.split_manifest is not None and a.source_lock is not None,'asset arguments')
        if a.action == 'verify':print(verify(a.output,a.commit,a.split_manifest,a.source_lock));return
        claim=read(REPO/REGISTRY/'attempt.json');launch=read(a.output/'launch.json')
        require(claim == dict(task=TASK,commit=a.commit,output=str(a.output),jobs=jobs()) and launch['task'] == TASK
                and launch['commit'] == a.commit and launch['jobs'] == jobs() and launch['spec'] == SPEC,'worker claim/launch')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == launch['gpu_uuid'] and torch.cuda.is_available() and torch.cuda.device_count() == 1,'single GPU binding')
        if a.mode == 'smoke':
            pair=next((v for v in SMOKE if f'smoke_{v[0]}_{v[1]}' == a.job),None);require(pair is not None,'smoke job')
            setting,method=pair
        else:
            require(a.mode == 'train' and a.job in [j['id'] for j in jobs()],'training job')
            for s,m in SMOKE:require(read(a.output/f'smoke_{s}_{m}'/'receipt.json')['content_status'] == 'PASS','real smoke required')
            job=next(j for j in jobs() if j['id'] == a.job);setting,method=job['setting'],job['method']
        out=a.output/a.job;out.mkdir(exist_ok=False);write(out/'started.json',dict(task=TASK,commit=a.commit,mode=a.mode,job=a.job))
        f,t,source=context(setting,a.split_manifest,a.source_lock,'cuda:0')
        if a.mode == 'smoke':smoke(f,t,source,setting,method,out,a.commit,'cuda:0')
        else:train_one(f,t,source,job,out,a.commit,'cuda:0',references(a.output/'reference_077')[a.job])


if __name__ == '__main__':main()
