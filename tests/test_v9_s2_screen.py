from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

import v9_screen as s1
import v9_s2_screen as mod
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory


@pytest.fixture
def runtime2(runtime,monkeypatch):
    monkeypatch.setattr(mod,'observations',s1.observations)
    return runtime


@pytest.mark.parametrize('job',mod.jobs(),ids=lambda j:j['id'])
def test_complete_paths_warmup_thaw_and_verification(job,tmp_path,runtime2):
    spec=dict(mod.SPEC,max_epochs=3,warmup_epochs=1)
    f,t=runtime2(job['setting']);out=tmp_path/job['id'];out.mkdir()
    r=mod.train_one(f,t,job,out,'a'*40,'cpu',None,spec)
    f,t=runtime2(job['setting']);checked=mod.verify_job(f,t,job,out,'a'*40,None,spec)
    assert r['selected'] == checked['selected'] and len(r['history']) == 3
    assert not r['test_evaluated'] and r['checkpoint_validation_replay']
    assert r['history'][0]['phase']['phase'] == ('HEAD_WARMUP' if job['method'] in mod.WARMUP_METHODS else 'JOINT')
    assert r['history'][1]['phase']['phase'] == 'JOINT'
    if job['method'] == 'SRGT_PLASTIC':
        assert r['history'][0]['phase']['adaptive_sha256'] == r['identity']['initial_encoder']
        assert r['history'][-1]['phase']['adaptive_sha256'] != r['identity']['initial_encoder']
        assert r['final_encoder'] == r['identity']['initial_encoder']


def test_plastic_identity_and_real_gradient_paths(runtime2):
    f,t=runtime2('ToxAcute');base=t.model
    _,counts=mod.c0.tasks_and_counts('ToxAcute');model,opt=mod.build(base,'SRGT_PLASTIC',counts,None,'ToxAcute','cpu')
    task=next(iter(counts));batch=mod.batch_for(f,t,'ToxAcute','train',task,[0,1],'cpu')
    bank=mod.srgt.TrainSupport(mod.srgt.training_records(f,t,'ToxAcute'))
    batch.v9_support=bank.for_train_batch(task,batch.sample_id,batch.canonical_smiles)
    base.eval();model.eval()
    with torch.no_grad():assert torch.equal(base(batch,task_name=task)[task],model(batch,task_name=task)[task])
    mod.begin_epoch(model,'SRGT_PLASTIC',5)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss=(model(batch,task_name=task)[task]-1).square().mean();loss.backward()
        assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
        assert sum(float(p.grad.abs().sum()) for p in model.adaptive.parameters() if p.grad is not None)>0
        if step:assert sum(float(p.grad.abs().sum()) for p in model.target.parameters() if p.grad is not None)>0
        opt.step()


@pytest.mark.parametrize('method',mod.WARMUP_METHODS)
def test_continuous_head_adam_and_no_backbone_state_during_warmup(method,runtime2):
    f,t=runtime2('ToxAcute');_,counts=mod.c0.tasks_and_counts('ToxAcute')
    model,opt=mod.build(t.model,method,counts,None,'ToxAcute','cpu')
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    task=next(iter(counts));batch=mod.batch_for(f,t,'ToxAcute','train',task,[0,1],'cpu')
    bank=mod.srgt.TrainSupport(mod.srgt.training_records(f,t,'ToxAcute'))
    batch.v9_support=bank.for_train_batch(task,batch.sample_id,batch.canonical_smiles)
    for epoch in range(6):
        mod.begin_epoch(model,method,epoch);opt.zero_grad(set_to_none=True)
        model(batch,task_name=task)[task].square().mean().backward();opt.step()
        for n,p in model.named_parameters():
            if epoch<5 and not n.startswith('decoders.'):
                assert torch.equal(p,initial[n]) and p not in opt.state
            if n.startswith('decoders.'+task+'.') and p in opt.state:assert opt.state[p]['step']==epoch+1


def test_fixed_epochs_does_not_truncate_delayed_improvement():
    values=[1.1]+[1.2]*11+[1.09]*11+[1.04]+[1.08]*16
    h=[dict(epoch=i+1,validation=dict(macro_rmse=v)) for i,v in enumerate(values)]
    assert len(h)==40 and mod.select(h)==(24,False)
    with pytest.raises(ValueError,match='complete fixed epochs'):mod.select(h[:9])


@pytest.mark.parametrize('field',['phase','warmup_tensor','selected_epoch','truncated'])
def test_independent_verifier_rejects_corruption(field,tmp_path,runtime2):
    job=next(j for j in mod.jobs() if j['setting']=='ToxAcute' and j['method']=='SRGT_PLASTIC')
    spec=dict(mod.SPEC,max_epochs=3,warmup_epochs=1);out=tmp_path/'job';out.mkdir()
    f,t=runtime2(job['setting']);r=mod.train_one(f,t,job,out,'a'*40,'cpu',None,spec)
    if field in ('phase','warmup_tensor'):
        p=out/'epoch_001.json';row=mod.read(p)
        if field=='phase':row['phase']['adaptive_trainable']=True
        else:row['phase']['adaptive_sha256']='0'*64
        p.write_text(json.dumps(row),encoding='utf8');r['history'][0]=row
    if field=='selected_epoch':r['best_epoch']=999
    if field=='truncated':r['history']=r['history'][:1]
    (out/'receipt.json').write_text(json.dumps(r),encoding='utf8')
    f,t=runtime2(job['setting'])
    with pytest.raises(ValueError):mod.verify_job(f,t,job,out,'a'*40,None,spec)


def test_three_strong_control_rates(runtime2):
    f,t=runtime2('ToxAcute');_,counts=mod.c0.tasks_and_counts('ToxAcute')
    for method,lr in [('B1_1E4',1e-4),('B1_1E3',1e-3),('HF_1E4',1e-4)]:
        model,opt=mod.build(t.model,method,counts,None,'ToxAcute','cpu')
        assert [g['lr'] for g in opt.param_groups]==[lr,1e-3]
        assert mod.begin_epoch(model,method,0)==('HEAD_WARMUP' if method=='HF_1E4' else 'JOINT')


def test_dispatch_failure_stops_remaining_jobs(tmp_path,monkeypatch):
    import p1d4_batch
    monkeypatch.setattr(mod,'REPO',tmp_path)
    monkeypatch.setattr(mod.c0,'check_code',lambda *a:None)
    monkeypatch.setattr(mod,'gate_check',lambda *a:None)
    monkeypatch.setattr(p1d4_batch,'free_gpus',lambda a:a)
    monkeypatch.setattr(mod.c0,'gpu_uuid',lambda a:'GPU-synthetic')
    calls=[]
    def failed(argv,**kw):calls.append(argv);return SimpleNamespace(returncode=17)
    monkeypatch.setattr(mod.subprocess,'run',failed)
    for role in ('wsl','server'):(tmp_path/role).mkdir()
    a=SimpleNamespace(commit='a'*40,gpu=0,output=tmp_path/'screen',wsl_evidence=tmp_path/'wsl',
        server_evidence=tmp_path/'server',stats_root=None,split_manifest=tmp_path/'split',source_lock=tmp_path/'source')
    with pytest.raises(ValueError,match='job failed'):mod.run(a)
    assert len(calls)==1 and (a.output/'failed.json').is_file()
    assert not (a.output/'verification.json').exists()
    assert '--stats-root' not in calls[0]


def test_entire_matrix_independent_verifier_and_initial_pairing(tmp_path,runtime2,monkeypatch):
    monkeypatch.setitem(mod.SPEC,'max_epochs',2)
    monkeypatch.setitem(mod.SPEC,'warmup_epochs',1)
    monkeypatch.setattr(mod,'gate_check',lambda *a:None)
    monkeypatch.setattr(mod,'factory_for',lambda setting,*a:runtime2(setting))
    mod.write(tmp_path/'launch.json',dict(task=mod.TASK,commit='a'*40,spec=mod.SPEC,jobs=mod.jobs()))
    for job in mod.jobs():
        f,t=runtime2(job['setting']);out=tmp_path/job['id'];out.mkdir()
        mod.train_one(f,t,job,out,'a'*40,'cpu',None)
    checked=mod.verify(tmp_path,'a'*40,tmp_path/'split',tmp_path/'source',None)
    assert checked['content_status']=='PASS' and checked['total_epochs']==36
    assert len(checked['results'])==18 and not checked['unified_superiority_confirmed']
