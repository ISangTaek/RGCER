"""Nine fixed V9-S4 task-sharing runs, preceded by two small real smokes."""
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

import v9_s3_screen as s3
import v9_s3r1_screen as r1
import v9_task_grouping as group

REPO=Path(__file__).resolve().parent
TASK='V9_S4_TASK_SHARING_SCREEN_20261002'
REGISTRY='.tmp/v9s4_attempt_20261002'
SPEC=dict(s3.SPEC,method_schema='v9_s4_v1',warmup_epochs=5,grouping=group.SPEC,
          relation_updates=0,grouping_validation_access=False,gradient_replay_rtol=.001,gradient_replay_atol=.00001)
METHODS=group.METHODS
SMOKE=('ToxAcute','A')
TESTS=('tests/test_v9_task_grouping.py','tests/test_v9_s4_screen.py',*r1.TESTS)
TEST_COUNT=228
read,write,sha,digest,require=s3.read,s3.write,s3.sha,s3.digest,s3.require


def jobs():
    return [dict(id=f'{s}_{m}_s42',setting=s,method=m,seed=42) for s in s3.s2.c0.SETTINGS for m in METHODS]


def identity_for(trainer,source,job,training,validation,commit,spec):
    return s3.identity_for(trainer,source,job,training,validation,commit,spec,task_id=TASK)


def pairing(identity, reference):
    old=reference['identity']
    require(identity['initial_encoder']==old['initial_encoder'] and identity['initial_target_heads']==old['initial_heads']
            and all(identity[k]==old[k] for k in ('contract_sha256','train_sha256','validation_sha256')), '077 population/initialization pairing')


def probe(factory, trainer, source, setting, training, spec=SPEC, *, probe_tasks=None):
    """No validation object accepted. Gradients change neither weights nor Adam."""
    model=trainer.model;tasks=model.task_name if probe_tasks is None else probe_tasks
    require(set(tasks)<=set(model.task_name),'probe task scope')
    selection=group.probe_selection([r for r in training+source.rows if r['task'] in tasks],tasks,spec['grouping'])
    by_target={t:{r['sample_id']:i for i,r in enumerate(v)} for t,v in
               ((t,[r for r in training if r['task']==t]) for t in tasks if t not in source.tasks)}
    by_source={t:{r['sample_id']:i for i,r in enumerate(source.by_task[t])} for t in source.tasks}
    before=s3.state_dict_sha256(model);flags={n:p.requires_grad for n,p in model.named_parameters()}
    device=str(next(model.parameters()).device);vectors={};model.eval();model.requires_grad_(False)
    weight=model.encoder.backbone.layers[-1].ffn[3].weight;weight.requires_grad_(True)
    try:
        for task in tasks:
            values=[]
            for row in selection[task]:
                if task in source.tasks:
                    batch=source.batch(task,[by_source[task][row['sample_id']]],device);scaler=source.scalers[task]
                else:
                    batch=s3.s2.batch_for(factory,trainer,setting,'train',task,[by_target[task][row['sample_id']]],device)
                    scaler=(trainer.task_scalers if setting=='ToxAcute' else trainer.scalers)[task]
                require(list(batch.sample_id)==[row['sample_id']] and list(batch.canonical_smiles)==[row['canonical']], 'probe graph identity')
                model._active=task
                try:representation=model.encoder(batch)
                finally:model._active=None
                raw=model.decoders[task](representation)
                loss=s3.QuantileRegressionLoss().compute_loss(raw,(batch.y.reshape(-1,1)-scaler['mean'])/scaler['std'])
                require(torch.isfinite(loss).item(),'probe finite loss')
                grad=torch.autograd.grad(loss,weight)[0].detach().float().cpu().flatten()
                require(torch.isfinite(grad).all(),'probe finite gradient');values.append(grad)
            vectors[task]=torch.stack(values)
    finally:
        for name,p in model.named_parameters():p.requires_grad_(flags[name])
        model._active=None
    require(s3.state_dict_sha256(model)==before,'probe changed model')
    return selection,vectors


def parameter_counts(model):
    return dict(encoder=sum(p.numel() for p in model.encoder.parameters()),
                adapters=sum(p.numel() for p in model.target.parameters()),
                heads=sum(p.numel() for p in model.decoders.parameters()))


def save_state(path, model, opt, identity, epoch):
    with path.open('xb') as stream:
        torch.save(dict(identity=identity,epoch=epoch,groups=model.groups,
                        model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        optimizer_state=opt.state_dict()),stream)


def train_one(factory,trainer,source,job,out,commit,device,reference,spec=SPEC):
    require(job in jobs(),'S4 fixed job');setting=job['setting'];tasks,counts=s3.s2.c0.tasks_and_counts(setting)
    training=s3.s2.observations(factory,trainer,setting,'train');validation=s3.s2.observations(factory,trainer,setting,'validation')
    populations=s3.population_check(factory,source,setting,training,validation)
    identity=identity_for(trainer,source,job,training,validation,commit,spec);pairing(identity,reference)
    require({t:len(v) for t,v in s3.s2.datasets_for(factory,trainer,setting)['train'].items()}==counts,'target counts')
    s3.seed_everything(42,deterministic_algorithms=True)
    trainer.model=group.GroupedGraph(trainer.model,source,job['method']).to(device);model=trainer.model;opt=group.optimizer_for(model)
    sizes=parameter_counts(model)
    for name,value in [('identity',identity),('train_observations',training),('validation_observations',validation),
                       ('source_train_observations',source.rows),('population_check',populations)]:write(out/(name+'.json'),value)
    rows,_=s3.s2.evaluate(factory,trainer,setting,None,validation,device,0);s3.paired_initial(rows,reference['initial'])
    write(out/'initial_validation.json',rows)
    schedule=s3.aux.Schedule(source.tasks,source.counts,s3.s2.c0.BATCH_SIZES[setting])
    history=[];all_aux=[];best=math.inf;updates=0;start=time.monotonic();topology=None
    for epoch in range(spec['max_epochs']):
        if epoch==spec['warmup_epochs']:
            save_state(out/'warmup.pt',model,opt,identity,epoch)
            selection,vectors=probe(factory,trainer,source,setting,training,spec)
            with (out/'gradients.pt').open('xb') as stream:torch.save(vectors,stream)
            relations=group.similarities(vectors,selection,model.task_name,spec['grouping'])
            groups,decision=group.choose_groups(job['method'],relations)
            topology=dict(selection=selection,relations=relations,groups=groups,decision=decision,
                          warmup_sha256=sha(out/'warmup.pt'),gradients_sha256=sha(out/'gradients.pt'),
                          gradient_dimension=next(iter(vectors.values())).shape[1],optimizer_updates=0,
                          gradient_calls=sum(len(v) for v in vectors.values()))
            write(out/'topology.json',topology);group.repartition(model,opt,groups)
            require(parameter_counts(model)==sizes,'capacity changed with groups')
        phase=group.begin_epoch(model,epoch,spec['warmup_epochs'])
        plan=s3.s2.epoch_plan(tasks,counts,s3.s2.c0.BATCH_SIZES[setting],epoch);auxiliary=[];losses=[]
        for item in plan:
            a=schedule.next();auxiliary.append(a);all_aux.append(a)
            batch=s3.s2.batch_for(factory,trainer,setting,'train',item['task'],item['indices'],device)
            losses.append(s3.step(trainer,setting,batch,item['task'],epoch,source,a,opt,device));updates+=1
        require(s3.state_dict_sha256(model.encoder)==identity['initial_encoder'],'source anchor changed')
        if phase=='HEAD_WARMUP':
            require(not any(n.startswith('target.') for n in s3.optimizer_steps(model,opt)),'adapter updated during warmup')
            require(all(torch.count_nonzero(p)==0 for n,p in model.target.named_parameters() if '.B.' in n),'warmup adapter changed')
        rows,score=s3.s2.evaluate(factory,trainer,setting,None,validation,device,epoch)
        h=dict(epoch=epoch+1,validation=score,updates=len(plan),total_updates=updates,schedule_sha256=digest(plan),
               auxiliary_schedule=auxiliary,losses=losses,phase=phase,groups=model.groups,
               encoder_sha256=s3.state_dict_sha256(model.encoder),optimizer_steps=s3.optimizer_steps(model,opt))
        write(out/f'validation_epoch_{epoch+1:03d}.json',rows);write(out/f'epoch_{epoch+1:03d}.json',h);history.append(h)
        if score['macro_rmse']<best:
            best=score['macro_rmse'];best_epoch=epoch+1
            save_state(out/'best.pending.pt',model,opt,identity,best_epoch);os.replace(out/'best.pending.pt',out/'best.pt')
        print(f'{job["id"]} epoch={epoch+1} macro={score["macro_rmse"]:.8f} best={best_epoch}',flush=True)
    require(topology is not None and s3.s2.select(history,spec)[0]==best_epoch,'complete screen')
    coverage=s3.aux.coverage(source,all_aux);require(all(v['updates']>0 for v in coverage.values()),'all source tasks must participate')
    write(out/'auxiliary_coverage.json',coverage)
    payload=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.configure(payload['groups']);model.load_state_dict(payload['model_state'],strict=True)
    rows,score=s3.s2.evaluate(factory,trainer,setting,None,validation,device,best_epoch-1)
    require(rows==read(out/f'validation_epoch_{best_epoch:03d}.json'),'best prediction replay');write(out/'selected_validation.json',rows)
    r=dict(task=TASK,commit=commit,job=job,identity=identity,history=history,best_epoch=best_epoch,updates=updates,
           selected=score,checkpoint_sha256=sha(out/'best.pt'),selected_state_sha256=s3.state_dict_sha256(model),
           selected_encoder=s3.state_dict_sha256(model.encoder),coverage=coverage,parameters=sizes,
           topology_sha256=sha(out/'topology.json'),elapsed_seconds=time.monotonic()-start,
           test_evaluated=False,checkpoint_validation_replay=True,scientific_acceptance='PENDING_REVIEW')
    write(out/'receipt.json',r);return r


def check_optimizer(model,opt,state,recorded,target_plan,aux_plan,warming):
    require(set(state)=={'state','param_groups'} and state['param_groups']==opt.state_dict()['param_groups'],'S4 optimizer groups')
    names={id(p):n for n,p in model.named_parameters()};expected={};steps={}
    plan=list(zip(target_plan,aux_plan));parameters=[p for g in opt.param_groups for p in g['params']]
    for i,p in enumerate(parameters):
        n=names[id(p)]
        if n.startswith('decoders.'):
            task=n.split('.')[1];count=sum(x['task']==task for x in target_plan+aux_plan)
        else:
            key=n.split('.')[1];layer=int(key.split('__')[1]);g=int(n.split('.')[3])
            members=set(model.task_name if layer<len(model.encoder.backbone.layers)//2 else model.groups[g])
            count=sum(a['task'] in members or b['task'] in members for a,b in plan[warming:])
        if count:expected[i]=p;steps[n]=count
    require(set(state['state'])==set(expected) and recorded==steps,'S4 Adam step accounting')
    for i,p in expected.items():
        v=state['state'][i];n=names[id(p)]
        require(set(v)=={'step','exp_avg','exp_avg_sq'} and float(v['step'])==steps[n] and
                all(torch.isfinite(x).all() for x in v.values()) and all(v[k].shape==p.shape and v[k].dtype==p.dtype
                for k in ('exp_avg','exp_avg_sq')), 'S4 Adam tensor '+n)


def verify_job(factory,trainer,source,job,out,commit,reference,spec=SPEC):
    r=read(out/'receipt.json');setting=job['setting'];tasks,counts=s3.s2.c0.tasks_and_counts(setting)
    training=s3.s2.observations(factory,trainer,setting,'train');validation=s3.s2.observations(factory,trainer,setting,'validation')
    identity=identity_for(trainer,source,job,training,validation,commit,spec);pairing(identity,reference)
    require(r['identity']==read(out/'identity.json')==identity and r['task']==TASK and r['commit']==commit and r['job']==job,'S4 identity')
    for name,value in [('train_observations',training),('validation_observations',validation),('source_train_observations',source.rows),
                       ('population_check',s3.population_check(factory,source,setting,training,validation))]:
        require(read(out/(name+'.json'))==value,'trusted '+name)
    s3.paired_initial(read(out/'initial_validation.json'),reference['initial'])
    model=group.GroupedGraph(trainer.model,source,job['method']);shared=[model.task_name.copy()]
    top=read(out/'topology.json');selection=group.probe_selection(training+source.rows,model.task_name,spec['grouping'])
    require(top['selection']==selection and sha(out/'gradients.pt')==top['gradients_sha256'] and
            sha(out/'warmup.pt')==top['warmup_sha256'] and sha(out/'topology.json')==r['topology_sha256'],'train-only topology identity')
    vectors=torch.load(out/'gradients.pt',map_location='cpu',weights_only=True)
    dim=model.encoder.backbone.layers[-1].ffn[3].weight.numel()
    require(top['gradient_dimension']==dim and all(v.ndim==2 and v.shape[1]==dim for v in vectors.values()),'trusted gradient dimension')
    relations=group.similarities(vectors,selection,model.task_name,spec['grouping']);groups,decision=group.choose_groups(job['method'],relations)
    require(top['relations']==relations and top['groups']==groups and top['decision']==decision and top['optimizer_updates']==0
            and top['gradient_calls']==sum(len(v) for v in vectors.values()),'recomputed task topology')
    history=r['history'];best,_=s3.s2.select(history,spec);tp=[];ap=[];schedule=s3.aux.Schedule(source.tasks,source.counts,s3.s2.c0.BATCH_SIZES[setting])
    for e,h in enumerate(history):
        plan=s3.s2.epoch_plan(tasks,counts,s3.s2.c0.BATCH_SIZES[setting],e);aq=[schedule.next() for _ in plan];tp+=plan;ap+=aq
        expected_groups=shared if e<spec['warmup_epochs'] else groups
        require(h==read(out/f'epoch_{e+1:03d}.json') and h['updates']==len(plan) and h['total_updates']==len(tp)
                and h['schedule_sha256']==digest(plan) and h['auxiliary_schedule']==aq,'paired schedule')
        require(h['validation']==s3.s2.metrics(read(out/f'validation_epoch_{e+1:03d}.json'),validation),'S4 epoch metric')
        require(h['phase']==('HEAD_WARMUP' if e<spec['warmup_epochs'] else 'GROUPED_ADAPTATION') and h['groups']==expected_groups
                and h['encoder_sha256']==identity['initial_encoder'],'S4 phase identity')
        require(len(h['losses'])==len(plan) and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for v in h['losses'] for x in v.values()),'S4 finite losses')
    per=len(tp)//spec['max_epochs'];selected=read(out/'selected_validation.json')
    require(r['best_epoch']==best and r['updates']==len(tp) and selected==read(out/f'validation_epoch_{best:03d}.json') and
            r['selected']==s3.s2.metrics(selected,validation) and r['test_evaluated'] is False and
            r['checkpoint_validation_replay'] is True and r['scientific_acceptance']=='PENDING_REVIEW','S4 selected result')
    coverage=s3.aux.coverage(source,ap)
    require(coverage==r['coverage']==read(out/'auxiliary_coverage.json') and all(v['updates']>0 for v in coverage.values()),'S4 auxiliary coverage')
    for name,epoch in [('warmup.pt',spec['warmup_epochs']),('best.pt',best)]:
        payload=torch.load(out/name,map_location='cpu',weights_only=True)
        expected_groups=shared if epoch<=spec['warmup_epochs'] else groups;model.configure(expected_groups);opt=group.optimizer_for(model)
        require(set(payload)=={'identity','epoch','groups','model_state','optimizer_state'} and payload['identity']==identity and
                payload['epoch']==epoch and payload['groups']==expected_groups,'S4 state identity')
        expected=model.state_dict();actual=payload['model_state'];require(set(actual)==set(expected),'S4 state keys')
        for k,v in expected.items():
            x=actual[k];require(isinstance(x,torch.Tensor) and x.shape==v.shape and x.dtype==v.dtype and torch.isfinite(x).all(),'S4 state tensor '+k)
            if k.startswith('encoder.') or epoch<=spec['warmup_epochs'] and k.startswith('target.'):
                require(torch.equal(x,v),'protected S4 tensor '+k)
        check_optimizer(model,opt,payload['optimizer_state'],history[epoch-1]['optimizer_steps'],tp[:epoch*per],ap[:epoch*per],min(epoch,spec['warmup_epochs'])*per)
        model.load_state_dict(actual,strict=True)
        if name=='warmup.pt':
            original=trainer.model;trainer.model=model
            try:replayed_selection,replayed=probe(factory,trainer,source,setting,training,spec)
            finally:trainer.model=original
            require(replayed_selection==selection and set(replayed)==set(vectors),'gradient replay members')
            require(all(torch.allclose(replayed[t],vectors[t],rtol=spec['gradient_replay_rtol'],atol=spec['gradient_replay_atol'])
                        for t in vectors),'gradient values disagree with trusted train replay')
    require(sha(out/'best.pt')==r['checkpoint_sha256'] and s3.state_dict_sha256(model)==r['selected_state_sha256'] and
            s3.state_dict_sha256(model.encoder)==r['selected_encoder']==identity['initial_encoder'] and parameter_counts(model)==r['parameters'],'S4 selected state')
    return r


def smoke(factory,trainer,source,setting,out,commit,device):
    training=s3.s2.observations(factory,trainer,setting,'train')
    trainer.model=group.GroupedGraph(trainer.model,source,'SGTA').to(device);model=trainer.model
    # Force both routes solely to exercise the new branch, not to select topology.
    groups=[list(trainer.model.task_name[:len(trainer.model.task_name)-len(source.tasks)]),list(source.tasks)]
    model.configure(groups);opt=group.optimizer_for(model);group.begin_epoch(model,5,5)
    before=s3.state_dict_sha256(model.encoder);schedule=s3.aux.Schedule(source.tasks,source.counts,s3.s2.c0.BATCH_SIZES[setting]);rows=[]
    tasks,counts=s3.s2.c0.tasks_and_counts(setting);plan=s3.s2.epoch_plan(tasks,counts,s3.s2.c0.BATCH_SIZES[setting],0)
    for i in range(2):
        item=plan[i%len(plan)];a=schedule.next();batch=s3.s2.batch_for(factory,trainer,setting,'train',item['task'],item['indices'],device)
        loss=s3.step(trainer,setting,batch,item['task'],5,source,a,opt,device)
        gradients={}
        for g in (0,1):
            values=[p.grad.abs().sum().item() for n,p in model.target.named_parameters()
                    if int(n.split('__')[1])>=len(model.encoder.backbone.layers)//2 and n.endswith('.'+str(g)) and p.grad is not None]
            gradients[str(g)]=sum(values)
        require(all(v>0 and math.isfinite(v) for v in gradients.values()),'both real task branches must receive gradients')
        require(s3.state_dict_sha256(model.encoder)==before,'smoke frozen anchor');rows.append(dict(loss=loss,branch_gradients=gradients))
    _,probes=probe(factory,trainer,source,setting,training,dict(SPEC,grouping=dict(group.SPEC,probe_groups_per_task=1)),
                   probe_tasks=[tasks[0],source.tasks[0]])
    require(all(torch.isfinite(v).all() and torch.count_nonzero(v)>0 for v in probes.values()),'real gradient probe path')
    with (out/'state.pt').open('xb') as stream:torch.save(model.state_dict(),stream)
    write(out/'receipt.json',dict(task=TASK,commit=commit,setting=setting,updates=2,content_status='PASS',rows=rows,
          checkpoint_sha256=sha(out/'state.pt'),gradient_probe_calls=2,test_evaluated=False,scope='REAL_TWO_BRANCH_AND_PROBE_EXECUTION_ONLY'))


def compare(results,reference):
    require(len(results)==9 and {r['job']['id'] for r in results}=={j['id'] for j in jobs()},'complete S4 matrix')
    rows=[];winners=[]
    for setting in s3.s2.c0.SETTINGS:
        subset=[r for r in results if r['job']['setting']==setting]
        require(len({digest(r['parameters']) for r in subset})==1,'capacity-matched arms')
        # All heads and the warm-up trajectory must be identical across arms.
        for e in range(SPEC['warmup_epochs']):
            require(len({digest(r['history'][e]) for r in subset})==1,'paired warmup trajectory')
        strong=min(reference[f'{setting}_{m}_s42']['selected']['macro_rmse'] for m in s3.s2.METHODS)
        for r in subset:
            score=r['selected']['macro_rmse'];rows.append(dict(job=r['job'],selected=r['selected'],best_epoch=r['best_epoch'],
                best_077=strong,delta_vs_best_077=score-strong,
                delta_vs_shared=score-next(x for x in subset if x['job']['method']=='SHARED_LORA')['selected']['macro_rmse'],
                delta_vs_tg=score-next(x for x in subset if x['job']['method']=='TG_LORA_GRAPH')['selected']['macro_rmse']))
    for method in METHODS:
        rr=[v for v in rows if v['job']['method']==method]
        if all(v['delta_vs_best_077']<0 and (method=='SHARED_LORA' or v['delta_vs_shared']<0) and
               (method!='SGTA' or v['delta_vs_tg']<0) for v in rr):winners.append(method)
    return dict(comparisons=rows,all_scene_candidate_signals=winners,unified_superiority_confirmed=False)


def gate_check(root,commit,role):
    require(read(root/'gate.json')==dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')),'S4 gate identity')
    require(read(root/'command.json')['exit_code']==0,'S4 gate exit')
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    require(len(cases)==TEST_COUNT and all(not any(c.findall(t) for t in ('failure','error','skipped')) for c in cases)
            and {c.attrib['classname'] for c in cases}=={'tests.'+Path(t).stem for t in TESTS},'S4 complete gate')


def code_gate(root,commit,role):
    require(role in ('wsl','server'),'S4 gate role');s3.s2.c0.check_code(REPO,commit);root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','pytest',*TESTS,'-q','--basetemp',str(root/'pytest_tmp'),'--junitxml',str(root/'tests.xml')]
    with (root/'tests.log').open('xb') as stream:
        p=subprocess.run(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    write(root/'command.json',dict(argv=argv,exit_code=p.returncode));require(p.returncode==0,'S4 tests failed')
    write(root/'gate.json',dict(task=TASK,commit=commit,role=role,tests=list(TESTS),junit_sha256=sha(root/'tests.xml')));gate_check(root,commit,role)


def claim_value(root,commit):return dict(task=TASK,commit=commit,output=str(root.resolve()),spec=SPEC,jobs=jobs(),smoke=list(SMOKE))


def check_launch(root,commit):
    claim=claim_value(root,commit);require(read(REPO/REGISTRY/'attempt.json')==claim,'S4 one-shot claim')
    launch=read(root/'launch.json');require(launch['claim']==claim and launch['task']==TASK and launch['commit']==commit,'S4 launch')
    return launch


def verify(root,commit,split,source_lock):
    check_launch(root,commit)
    for role in ('wsl','server'):gate_check(root/(role+'_evidence'),commit,role)
    reference=s3.references(root/'reference_077');results=[]
    for setting in SMOKE:
        out=root/('smoke_'+setting);r=read(out/'receipt.json')
        require(read(root/(out.name+'.command.json'))['exit_code']==0 and r['task']==TASK and r['commit']==commit and
                r['setting']==setting and r['updates']==2 and r['content_status']=='PASS' and len(r['rows'])==2
                and sha(out/'state.pt')==r['checkpoint_sha256'] and r['gradient_probe_calls']==2
                and all(set(row['branch_gradients'])=={'0','1'} and all(v>0 and math.isfinite(v) for v in row['branch_gradients'].values()) for row in r['rows']),'S4 real smoke')
    for job in jobs():
        require(read(root/(job['id']+'.command.json'))['exit_code']==0,'S4 worker exit')
        f,t,a=s3.context(job['setting'],split,source_lock,'cpu')
        results.append(verify_job(f,t,a,job,root/job['id'],commit,reference[f'{job["setting"]}_B1_1E4_s42']))
    require(sum(r['updates'] for r in results)==3600,'S4 total updates')
    return dict(task=TASK,commit=commit,content_status='PASS',scientific_acceptance='PENDING_REVIEW',
                total_epochs=360,total_updates=3600,smoke_updates=4,**compare(results,reference))


def run(a):
    from p1d4_batch import free_gpus
    s3.s2.c0.check_code(REPO,a.commit)
    for role in ('wsl','server'):gate_check(getattr(a,role+'_evidence'),a.commit,role)
    s3.references(a.reference_root);root=a.output
    require(not (REPO/REGISTRY/'attempt.json').exists(),'S4 attempt consumed')
    require(free_gpus([a.gpu])==[a.gpu],'GPU busy')
    root.mkdir(parents=True,exist_ok=False);(REPO/REGISTRY).mkdir(parents=True,exist_ok=True)
    claim=claim_value(root,a.commit);write(REPO/REGISTRY/'attempt.json',claim)
    uuid=s3.s2.c0.gpu_uuid(a.gpu);write(root/'launch.json',dict(task=TASK,commit=a.commit,claim=claim,gpu=a.gpu,gpu_uuid=uuid))
    try:
        for role in ('wsl','server'):shutil.copytree(getattr(a,role+'_evidence'),root/(role+'_evidence'),ignore=shutil.ignore_patterns('pytest_tmp'))
        s3.copy_references(a.reference_root,root/'reference_077')
        sequence=[('smoke',s,'smoke_'+s) for s in SMOKE]+[('train',j['id'],j['id']) for j in jobs()]
        for mode,key,name in sequence:
            require(free_gpus([a.gpu])==[a.gpu],'GPU occupied')
            argv=[sys.executable,str(Path(__file__).resolve()),'_worker','--mode',mode,'--job',key,'--commit',a.commit,
                  '--output',str(root),'--split-manifest',str(a.split_manifest),'--source-lock',str(a.source_lock)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/(name+'.log')).open('xb') as log:p=subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(root/(name+'.command.json'),dict(argv=argv,exit_code=p.returncode));require(p.returncode==0,'S4 job failed; preserve and stop')
        write(root/'verification.json',verify(root,a.commit,a.split_manifest,a.source_lock))
    except BaseException as exc:
        write(root/'failed.json',dict(task=TASK,error_type=type(exc).__name__,reason=str(exc)));raise


def worker(a):
    launch=check_launch(a.output,a.commit)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==launch['gpu_uuid'] and torch.cuda.is_available() and torch.cuda.device_count()==1,'S4 single GPU binding')
    sequence=[('smoke',s,'smoke_'+s) for s in SMOKE]+[('train',j['id'],j['id']) for j in jobs()]
    item=next((x for x in sequence if x[:2]==(a.mode,a.job)),None);require(item is not None,'S4 fixed worker scope')
    for _,_,name in sequence[:sequence.index(item)]:
        r=read(a.output/name/'receipt.json');require(r['task']==TASK and r['commit']==a.commit and read(a.output/(name+'.command.json'))['exit_code']==0,'preceding S4 work incomplete')
    setting=a.job if a.mode=='smoke' else next(j for j in jobs() if j['id']==a.job)['setting']
    f,t,source=s3.context(setting,a.split_manifest,a.source_lock,'cuda:0')
    training=s3.s2.observations(f,t,setting,'train');validation=s3.s2.observations(f,t,setting,'validation')
    s3.population_check(f,source,setting,training,validation)
    out=a.output/item[2];out.mkdir(exist_ok=False);write(out/'started.json',dict(task=TASK,commit=a.commit,mode=a.mode,job=a.job))
    if a.mode=='smoke':smoke(f,t,source,setting,out,a.commit,'cuda:0')
    else:
        ref=s3.references(a.output/'reference_077')[f'{setting}_B1_1E4_s42']
        train_one(f,t,source,next(j for j in jobs() if j['id']==a.job),out,a.commit,'cuda:0',ref)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('code-gate','run','_worker','verify','package'))
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=('wsl','server'))
    p.add_argument('--gpu',type=int,choices=range(4));p.add_argument('--mode',choices=('smoke','train'));p.add_argument('--job',choices=[*SMOKE,*[j['id'] for j in jobs()]])
    for name in ('split-manifest','source-lock','reference-root','wsl-evidence','server-evidence'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.action=='code-gate':code_gate(a.output,a.commit,a.role)
    elif a.action=='package':print(s3.s2.package(a.output,a.commit))
    else:
        s3.s2.c0.check_code(REPO,a.commit)
        require(a.split_manifest is not None and a.source_lock is not None,'S4 asset paths')
        if a.action=='run':
            require(all(getattr(a,k) is not None for k in ('gpu','reference_root','wsl_evidence','server_evidence')),'S4 run arguments');run(a)
        elif a.action=='verify':print(verify(a.output,a.commit,a.split_manifest,a.source_lock))
        else:worker(a)


if __name__=='__main__':main()
