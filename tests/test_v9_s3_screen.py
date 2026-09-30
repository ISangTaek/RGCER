from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import random

import pytest
import torch

import v9_s3_screen as mod
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime,one_cpu_thread,setup_factory


@pytest.fixture
def joint(runtime2):
    def make(setting):
        f,t=runtime2(setting); datasets=mod.s2.datasets_for(f,t,setting)['train']
        first=next(iter(datasets));scaler=(t.task_scalers if setting=='ToxAcute' else t.scalers)[first]
        rows=mod.s2.observations(f,t,setting,'train');original=[r for r in rows if r['task']==first]
        names=['aux_one','aux_two']
        source=mod.aux.Source({n:datasets[first] for n in names},[dict(r,task=n) for n in names for r in original],
            torch.nn.ModuleDict({n:deepcopy(t.model.decoders[first]).cpu() for n in names}),
            {n:deepcopy(scaler) for n in names},dict(synthetic=True))
        validation=mod.s2.observations(f,t,setting,'validation')
        initial,_=mod.s2.evaluate(f,t,setting,None,validation,'cpu',0)
        ref=dict(identity=dict(initial_encoder=mod.state_dict_sha256(t.model.encoder),initial_heads=mod.state_dict_sha256(t.model.decoders),
            contract_sha256=mod.digest(mod.s2.c0.contract(setting)),train_sha256=mod.digest(rows),validation_sha256=mod.digest(validation)),initial=initial)
        return f,t,source,ref
    return make


@pytest.mark.parametrize('job',mod.jobs(),ids=lambda j:j['id'])
def test_nine_joint_paths_full_selection_and_verification(job,tmp_path,joint):
    f,t,a,ref=joint(job['setting']);out=tmp_path/'joint';out.mkdir();spec=dict(mod.SPEC,max_epochs=3,warmup_epochs=1)
    r=mod.train_one(f,t,a,job,out,'a'*40,'cpu',ref,spec)
    f,t,a,ref=joint(job['setting']);checked=mod.verify_job(f,t,a,job,out,'a'*40,ref,spec)
    assert r==checked and len(r['history'])==3 and r['updates']==r['auxiliary_batches']==r['target_batches']
    assert all(v['updates']>0 for v in r['coverage'].values())


@pytest.mark.parametrize('method',mod.METHODS)
@pytest.mark.parametrize('setting',mod.s2.c0.SETTINGS)
def test_aux_disabled_exact_s2_pairing_with_dropout(method,setting,joint):
    f,t,source,_=joint(setting);base=deepcopy(t.model);_,counts=mod.s2.c0.tasks_and_counts(setting)
    # Same construction/seed/order as each runner, including ReFine's warm-up.
    trajectories=[]
    for joint_path in (False,True):
        mod.seed_everything(42,deterministic_algorithms=True)
        model,opt=(mod.build(base,source,method,counts,setting,'cpu') if joint_path else mod.s2.build(base,method,counts,None,setting,'cpu'))
        t.model=model;expected=mod.s2.observations(f,t,setting,'validation');mod.s2.evaluate(f,t,setting,None,expected,'cpu',0)
        for epoch in range(2):
            mod.s2.begin_epoch(model,method,epoch,dict(mod.SPEC,warmup_epochs=1))
            for item in mod.s2.epoch_plan(list(counts),counts,mod.s2.c0.BATCH_SIZES[setting],epoch):
                batch=mod.s2.batch_for(f,t,setting,'train',item['task'],item['indices'],'cpu')
                if joint_path:mod.step(t,setting,batch,item['task'],epoch,source,None,opt,'cpu',auxiliary=False)
                else:
                    opt.zero_grad(set_to_none=True)
                    if setting=='ToxAcute':loss,_=t._training_step(batch,item['task'],epoch)
                    else:
                        s=t.scalers[item['task']];loss=mod.QuantileRegressionLoss().compute_loss(model(batch,task_name=item['task'])[item['task']],(batch.y.reshape(-1,1)-s['mean'])/s['std'])
                    if method=='REFINE_GRAPH':loss=loss+model.regularization()
                    loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
            mod.s2.evaluate(f,t,setting,None,expected,'cpu',epoch)
        trajectories.append(({k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('decoders.aux_')},
                             torch.get_rng_state(),mod.optimizer_steps(model,opt)))
    assert trajectories[0][2]==trajectories[1][2]
    assert torch.equal(trajectories[0][1],trajectories[1][1])
    assert all(torch.equal(v,trajectories[1][0][k]) for k,v in trajectories[0][0].items())


@pytest.mark.parametrize('setting,method',mod.SMOKE)
def test_source_only_smoke_proves_shared_gradient_route(setting,method,tmp_path,joint):
    f,t,a,_=joint(setting);mod.smoke(f,t,a,setting,method,tmp_path,'a'*40,'cpu')
    r=mod.read(tmp_path/'receipt.json');assert r['updates']==2 and r['content_status']=='PASS'
    if method=='REFINE_GRAPH':assert r['rows'][1]['gradient_l1']['target']>0 and r['rows'][1]['gradient_l1']['encoder']==0


def test_auxiliary_does_not_advance_target_torch_or_python_rng(joint):
    f,t,a,_=joint('A');_,counts=mod.s2.c0.tasks_and_counts('A');base=deepcopy(t.model);results=[]
    for enabled in (False,True):
        mod.seed_everything(42,deterministic_algorithms=True);t.model,opt=mod.build(base,a,'B1_1E4',counts,'A','cpu')
        mod.s2.begin_epoch(t.model,'B1_1E4',0)
        item=mod.s2.epoch_plan(list(counts),counts,32,0)[0];batch=mod.s2.batch_for(f,t,'A','train',item['task'],item['indices'],'cpu')
        schedule=mod.aux.Schedule(a.tasks,a.counts,32)
        mod.step(t,'A',batch,item['task'],0,a,schedule.next(),opt,'cpu',auxiliary=enabled)
        results.append((torch.get_rng_state(),random.getstate()))
    assert torch.equal(results[0][0],results[1][0]) and results[0][1]==results[1][1]


def test_balanced_task_cycles_no_replacement_or_tail_loss():
    tasks=['x','y','z'];counts=dict(x=65,y=3,z=31);s=mod.aux.Schedule(tasks,counts,32)
    rows=[s.next() for _ in range(30)];other=mod.aux.Schedule(tasks,counts,32)
    assert rows==[other.next() for _ in range(30)]
    for i in range(0,30,3):assert {r['task'] for r in rows[i:i+3]}==set(tasks)
    x=[r['indices'] for r in rows if r['task']=='x']
    assert [len(v) for v in x[:3]]==[32,32,1] and sorted(sum(x[:3],[]))==list(range(65))
    assert all(len(r['indices'])==3 for r in rows if r['task']=='y')


@pytest.mark.parametrize('damage',['source_encoder','head_missing','head_shape','head_nan','extra_task'])
def test_restored_source_heads_fail_closed(damage,joint):
    _,t,a,_=joint('A');state={'encoder.'+k:v.detach().clone() for k,v in t.model.encoder.state_dict().items()}
    state.update({'decoders.'+k:v.detach().clone() for k,v in a.heads.state_dict().items()})
    valid=mod.aux.restored_heads(t.model,a.tasks,state);assert mod.state_dict_sha256(valid)==mod.state_dict_sha256(a.heads)
    key=next(k for k in state if k.startswith('decoders.'))
    if damage=='source_encoder':state[next(k for k in state if k.startswith('encoder.'))].add_(1)
    if damage=='head_missing':del state[key]
    if damage=='head_shape':state[key]=torch.zeros(1)
    if damage=='head_nan':state[key].fill_(float('nan'))
    if damage=='extra_task':state['decoders.extra.weight']=torch.zeros(1)
    with pytest.raises(ValueError):mod.aux.restored_heads(t.model,a.tasks,state)


@pytest.mark.parametrize('damage',['source_scaler','validation_label','auxiliary_schedule','optimizer_steps','anchor_tensor','truncated'])
def test_verifier_rejects_self_consistent_forgery(damage,tmp_path,joint):
    job=mod.jobs()[2];f,t,a,ref=joint(job['setting']);spec=dict(mod.SPEC,max_epochs=3,warmup_epochs=1)
    r=mod.train_one(f,t,a,job,tmp_path,'a'*40,'cpu',ref,spec)
    if damage=='source_scaler':r['identity']['source']['scalers'][a.tasks[0]]['mean']+=1
    elif damage=='validation_label':
        path=tmp_path/'validation_epoch_001.json';rows=mod.read(path);rows[0]['label']+=1;path.write_text(json.dumps(rows),encoding='utf8')
    elif damage in ('auxiliary_schedule','optimizer_steps'):
        h=r['history'][r['best_epoch']-1]
        if damage=='auxiliary_schedule':h['auxiliary_schedule'][0]['indices'].reverse()
        else:h['optimizer_steps'][next(iter(h['optimizer_steps']))]+=1
        (tmp_path/f'epoch_{r["best_epoch"]:03d}.json').write_text(json.dumps(h),encoding='utf8')
    elif damage=='anchor_tensor':
        p=tmp_path/'best.pt';payload=torch.load(p,weights_only=True);payload['model_state'][next(k for k in payload['model_state'] if k.startswith('encoder.'))].add_(1)
        torch.save(payload,p);r['checkpoint_sha256']=mod.sha(p)
    else:r['history']=r['history'][:1]
    (tmp_path/'receipt.json').write_text(json.dumps(r),encoding='utf8')
    (tmp_path/'identity.json').write_text(json.dumps(r['identity']),encoding='utf8')
    f,t,a,ref=joint(job['setting'])
    with pytest.raises(ValueError):mod.verify_job(f,t,a,job,tmp_path,'a'*40,ref,spec)


def test_comparison_keeps_best_old_controls_and_no_scene_collage():
    old={j['id']:dict(selected=dict(macro_rmse=1.,endpoints={'t':dict(rmse=1.)})) for j in mod.s2.jobs()}
    results=[dict(job=j,best_epoch=4,selected=dict(macro_rmse=.9 if j['method']=='REFINE_GRAPH' else 1.,endpoints={'t':dict(rmse=.9)})) for j in mod.jobs()]
    report=mod.compare(results,old);assert report['refine_all_scene_candidate_signal'] and not report['unified_superiority_confirmed']
    old['A_HF_1E4_s42']['selected']['macro_rmse']=.85
    assert not mod.compare(results,old)['refine_all_scene_candidate_signal']
    with pytest.raises(ValueError):mod.compare(results[:-1],old)


def test_dispatch_smoke_failure_preserves_claim_and_never_trains(tmp_path,monkeypatch):
    import p1d4_batch
    monkeypatch.setattr(mod,'REPO',tmp_path);monkeypatch.setattr(mod.s2.c0,'check_code',lambda *a:None)
    monkeypatch.setattr(mod,'gate_check',lambda *a:None);monkeypatch.setattr(mod,'references',lambda *a:{})
    monkeypatch.setattr(mod,'copy_references',lambda *a:None);monkeypatch.setattr(p1d4_batch,'free_gpus',lambda a:a)
    monkeypatch.setattr(mod.s2.c0,'gpu_uuid',lambda *a:'GPU-synthetic');calls=[]
    def failure(argv,**kw):calls.append(argv);return SimpleNamespace(returncode=17)
    monkeypatch.setattr(mod.subprocess,'run',failure)
    for role in ('wsl','server'):(tmp_path/role).mkdir()
    a=SimpleNamespace(commit='a'*40,gpu=0,output=tmp_path/'output',wsl_evidence=tmp_path/'wsl',server_evidence=tmp_path/'server',
                      reference_root=tmp_path/'reference',split_manifest=tmp_path/'split',source_lock=tmp_path/'source')
    with pytest.raises(ValueError,match='job failed'):mod.run(a)
    assert len(calls)==1 and calls[0][calls[0].index('--mode')+1]=='smoke'
    assert (tmp_path/mod.REGISTRY/'attempt.json').is_file() and (a.output/'failed.json').is_file()
    assert not (a.output/'verification.json').exists()


def test_pubchem_loader_restores_all_104_heads_and_train_scalers(tmp_path,joint):
    from dataset115_adapter import TrainOnlyScaler
    from dataset115_contract import semantic_digest
    f,t,_,_=joint('A');view=f._table.view('A','source','train')
    scaler=TrainOnlyScaler.fit(view);tasks=list(view.tasks)
    heads=torch.nn.ModuleDict({n:deepcopy(next(iter(t.model.decoders.values()))) for n in tasks})
    with torch.no_grad():
        for i,h in enumerate(heads.values()):next(h.parameters()).add_(.01*(i+1))
    state={'encoder.'+k:v for k,v in t.model.encoder.state_dict().items()}
    state.update({'decoders.'+k:v for k,v in heads.state_dict().items()})
    identity=dict(task_names=tasks,scaler=scaler.to_dict())
    payload=dict(epoch=39,identity=identity,identity_sha256=semantic_digest(identity),model_state=state)
    path=tmp_path/'epoch_039.pt';torch.save(payload,path);f._output=tmp_path
    f._expected['source_identity'].update(teacher_sha256=mod.sha(path),source_identity_sha256=semantic_digest(identity))
    source=mod.aux.load(f,t,'A',tmp_path,None)
    assert len(source.tasks)==104 and mod.state_dict_sha256(source.heads)==mod.state_dict_sha256(heads)
    assert source.scalers==scaler.trainer_scalers() and source.batch(tasks[-1],[0,1],'cpu').y.numel()==2
    # A self-consistent saved identity still cannot substitute different scalers.
    payload['identity']['scaler']['scaler']['means'][0]+=1
    payload['identity_sha256']=semantic_digest(payload['identity']);torch.save(payload,path)
    f._expected['source_identity'].update(teacher_sha256=mod.sha(path),source_identity_sha256=payload['identity_sha256'])
    with pytest.raises(ValueError,match='scaler identity'):mod.aux.load(f,t,'A',tmp_path,None)


def test_animal56_loader_uses_sample_ids_and_train_only_indices(tmp_path,joint,monkeypatch):
    import numpy as np
    import dataset115_source
    from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
    from preprocess_data import get_graph_data_from_smiles
    f,t,_,_=joint('ToxAcute');tasks=list(ANIMAL_SOURCE_TASKS)
    meta=dict(datastore_fingerprint='f'*64,split_manifest_hash='e'*64,feature_schema_version='synthetic')
    smiles=['CC','CCC'];ids=['original_row_100','original_row_700']
    mod.write(tmp_path/'split_manifest.json',dict(records=[dict(sample_id=s,split='train',canonical_smiles=c,split_group=c,row_index=1000+i)
              for i,(s,c) in enumerate(zip(ids,smiles))]))
    requested=[]
    def indices(task,*,split,max_nodes):
        requested.append((task,split,max_nodes));assert split=='train' and max_nodes==512
        return np.asarray([0,1])
    f.store=SimpleNamespace(root=tmp_path,metadata=meta,task_names=tasks,task_index=tasks.index,get_task_indices=indices,
        sample_ids=ids,get_label=lambda i,task:float(2*i),get_graph_data=lambda i,task_name,label:get_graph_data_from_smiles(
            smiles[i],label,sample_id=ids[i],task_name=task_name,max_path_distance=8))
    heads=torch.nn.ModuleDict({n:deepcopy(next(iter(t.model.decoders.values()))) for n in tasks})
    state={'encoder.'+k:v for k,v in t.model.encoder.state_dict().items()};state.update({'decoders.'+k:v for k,v in heads.state_dict().items()})
    scalers={t:dict(mean=1.,std=1.,count=2) for t in tasks};data=dict(meta,max_nodes_filter=512)
    path=tmp_path/'teacher.pt';torch.save(dict(epoch=39,model_state=state,data_config=data,architecture_config={},task_scalers=scalers),path)
    teacher=dict(kind='teacher',epoch=39,path='/home/shangzeli/RGCER/teacher.pt',sha256=mod.sha(path),size_bytes=path.stat().st_size,
                 architecture_config={},data_config=data,scalers=[dict(task=n,**s) for n,s in scalers.items()])
    monkeypatch.setattr(dataset115_source,'load_binding',lambda *a:(teacher,{}))
    source=mod.aux.load(f,t,'ToxAcute',tmp_path,path)
    assert len(source.tasks)==56 and len(requested)==56 and source.rows[0]['sample_id']==ids[0]
    assert source.rows[1]['sample_id']==ids[1] and source.batch(tasks[-1],[1],'cpu').sample_id==[ids[1]]
    assert mod.state_dict_sha256(source.heads)==mod.state_dict_sha256(heads)
    # Hash verification belongs to the existing strict legacy deserializer.
    teacher['sha256']='0'*64
    with pytest.raises(Exception):mod.aux.load(f,t,'ToxAcute',tmp_path,path)


def test_reference_lock_binds_metrics_and_every_old_control(tmp_path,monkeypatch):
    lock=dict(schema='v9_s3_077_reference_v1',commit='b'*40,jobs={})
    for job in mod.s2.jobs():
        folder=tmp_path/job['id'];folder.mkdir()
        row=dict(task='task',sample_id='sid',split='validation',canonical='CC',group='CC',label=1.)
        predictions=[dict(row,prediction=2.)];identity=dict(synthetic=True)
        values={'identity.json':identity,'validation_observations.json':[row],'initial_validation.json':predictions,
            'selected_validation.json':predictions,'receipt.json':dict(identity=identity,job=job,commit='b'*40,best_epoch=2,selected=mod.s2.metrics(predictions,[row]))}
        for name,value in values.items():mod.write(folder/name,value)
        lock['jobs'][job['id']]=dict(files={name:mod.sha(folder/name) for name in values})
    path=tmp_path/'lock.json';mod.write(path,lock);monkeypatch.setattr(mod,'LOCK',path)
    assert len(mod.references(tmp_path))==18
    (tmp_path/'A_HF_1E4_s42'/'selected_validation.json').write_text('[]',encoding='utf8')
    with pytest.raises(ValueError,match='077 reference file'):mod.references(tmp_path)
