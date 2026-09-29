"""Real source archive/RouteFactory chain, tiny synthetic molecules only.

The CUDA provenance below is simulated by rebinding every synthetic receipt;
this is a portability regression, never evidence of a GPU experiment.
"""
from copy import deepcopy
import json
import shutil

import pytest
import torch

import dataset115_route_a_training as source_mod
import p1d_routes as routes
from dataset115_contract import ContractError, digest, semantic_digest
from dataset115_training import EpochConfig
from tests.test_dataset115_adapter import args
from tests.test_dataset115_route_a_training import source_runner

SMALL=EpochConfig(2,2,.001,1e-5,1.)


def read(path):return json.loads(path.read_bytes())
def write(path,value):path.write_text(json.dumps(value,allow_nan=False),encoding='utf8')


@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    before=torch.get_num_threads();torch.set_num_threads(1)
    path=tmp_path_factory.mktemp('portable_source')/'source'
    source_runner().run(path)
    identity=read(path/'resolved_config.json');identity['device']='cuda:0'
    identity_sha=semantic_digest(identity);write(path/'resolved_config.json',identity)
    summary=read(path/'training_summary.json');summary['identity_sha256']=identity_sha
    for epoch in range(SMALL.epochs):
        checkpoint=path/f'epoch_{epoch:03d}.pt';p=torch.load(checkpoint,weights_only=True)
        p.update(identity=identity,identity_sha256=identity_sha)
        torch.save(p,checkpoint)
        receipt=dict(epoch=epoch,path=checkpoint.name,size_bytes=checkpoint.stat().st_size,sha256=digest(checkpoint))
        write(path/f'epoch_{epoch:03d}.receipt.json',receipt);summary['checkpoints'][epoch]=receipt
    write(path/'training_summary.json',summary)
    # Different recorded numbers must not be advertised as a CPU replay PASS.
    probe=read(path/'train_reload_probe.json');probe[0]['standardized_prediction'][0][0]+=.001
    write(path/'train_reload_probe.json',probe)
    encoder={k[8:]:v for k,v in p['model_state'].items() if k.startswith('encoder.')}
    expected=dict(seed=42,route='A',selection='fixed_final_epoch39',
                  teacher_sha256=digest(checkpoint),init_sha256=source_mod.state_digest(encoder),
                  source_identity_sha256=identity_sha)
    yield path,expected
    torch.set_num_threads(before)


def test_legacy_cpu_loader_reproduces_072_identity_failure(archive):
    with pytest.raises(ContractError,match='source config identity'):
        source_mod.verify_source(archive[0],source_runner())


def test_portable_content_is_bound_to_external_identity_without_any_forward(archive,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('CPU content verification must not forward or train')
    trainer=source_runner();monkeypatch.setattr(trainer,'probe',forbidden)
    monkeypatch.setattr(trainer.model,'forward',forbidden)
    receipt=source_mod.verify_source(archive[0],trainer,reuse_expectation=archive[1])
    assert receipt['identity_sha256']==archive[1]['source_identity_sha256']
    assert receipt['verification_mode']=='LOCKED_CPU_CONTENT'
    assert receipt['source_device']=='cuda:0' and receipt['runtime_device']=='cpu'
    assert receipt['train_probe_replayed'] is False and receipt['source_forward_calls']==0
    assert trainer.identity['device']=='cpu'  # No mutation or masquerading as CUDA.


@pytest.mark.parametrize('field',['teacher_sha256','init_sha256','source_identity_sha256','seed','unknown'])
def test_bad_external_expectation_rejected(archive,field):
    expected=deepcopy(archive[1]);expected[field]=True if field in ('seed','unknown') else 'f'*64
    with pytest.raises(ContractError):
        source_mod.verify_source(archive[0],source_runner(),reuse_expectation=expected)


@pytest.mark.parametrize('field',['train_ids','train_observations','input_identity','scaler','architecture','torch_version','unknown'])
def test_cpu_portability_does_not_relax_scientific_identity(archive,field):
    trainer=source_runner();trainer.identity[field]='forged'
    with pytest.raises(ContractError,match='source config identity'):
        source_mod.verify_source(archive[0],trainer,reuse_expectation=archive[1])


@pytest.mark.parametrize('change',['optimizer','probe_ids','probe_nonfinite','probe_missing_task','source_device'])
def test_self_consistent_source_tampering_still_rejected(archive,tmp_path,change):
    path=tmp_path/'source';shutil.copytree(archive[0],path);expected=deepcopy(archive[1])
    if change=='optimizer':
        checkpoint=path/'epoch_000.pt';p=torch.load(checkpoint,weights_only=True)
        p['optimizer_state']['state'][0]['step']=torch.tensor(999.)
        torch.save(p,checkpoint)
        rp=path/'epoch_000.receipt.json';r=read(rp);r.update(sha256=digest(checkpoint),size_bytes=checkpoint.stat().st_size)
        write(rp,r);s=read(path/'training_summary.json');s['checkpoints'][0]=r;write(path/'training_summary.json',s)
    elif change=='source_device':
        p=path/'resolved_config.json';r=read(p);r['device']='cuda:1';write(p,r)
        expected['source_identity_sha256']=semantic_digest(r)
    else:
        p=path/'train_reload_probe.json';r=read(p)
        if change=='probe_ids':r[0]['sample_ids'][0]='forged'
        if change=='probe_missing_task':r.pop()
        if change=='probe_nonfinite':r[0]['standardized_prediction'][0][0]='NaN'
        write(p,r)
    with pytest.raises(ContractError):
        source_mod.verify_source(path,source_runner(),reuse_expectation=expected)


def test_portable_loading_returns_historical_identity_and_refuses_nonformal_source(archive,monkeypatch):
    with pytest.raises(ContractError,match='nonformal'):
        source_mod.load_source(archive[0],source_runner(),reuse_expectation=archive[1])
    monkeypatch.setattr(source_mod,'CONFIG',SMALL)
    _,binding,receipt=source_mod.load_source(archive[0],source_runner(),reuse_expectation=archive[1])
    assert binding==archive[1] and receipt['formal_source'] is True


@pytest.mark.parametrize('method',['REFINE_GRAPH','SRGT'])
def test_real_route_factory_to_candidate_save_restore_and_cpu_verification(tmp_path,monkeypatch,method):
    """No source loader/identity mocks: actual source, graph, optimizer and reload."""
    import numpy as np
    import v9_c1_probe as probe
    from dataset115_adapter import Dataset115Table
    from dataset115_contract import PRIMARY
    from tests.test_dataset115_adapter import view
    from tests.test_dataset115_training import validation_view
    from tests.test_dataset115_route_a_smoke import TASKS
    before=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setattr(source_mod,'CONFIG',SMALL);monkeypatch.setattr(routes,'CONFIG',SMALL)
    monkeypatch.setattr(routes,'ARCH',vars(args()))
    table=Dataset115Table();table.identity=view().input_identity;table.tasks=PRIMARY+tuple(TASKS)
    table.source_tasks=tuple(TASKS);table.overlaps=frozenset()
    table.records=tuple(dict(row_index=i,raw_smiles=s,canonical_smiles=s,split_group=s,
                             split='train' if i<2 else 'validation') for i,s in enumerate(('CC','CCC','CO','CN')))
    table.values=np.concatenate([np.concatenate([view().labels,validation_view().labels]),
                                  np.tile([[0.],[2.],[0.],[2.]],(1,104))],axis=1)
    table.values.setflags(write=False)
    path=tmp_path/'source'
    def make_source():return source_mod.SourceTrainer(args(),table.view('A','source','train'),seed=42,config=SMALL,device='cpu')
    make_source().run(path)
    encoder,binding,_=source_mod.load_source(path,make_source())
    original=source_mod.RouteATrainer(args(),encoder,binding,table.view('A','target','train'),
        table.view('A','target','validation'),method='B1',seed=42,config=SMALL,device='cpu')
    from reproducibility import state_dict_sha256
    v9_initial=state_dict_sha256(original.model.encoder)
    expected=routes.expected_identity_from_record(original.identity,initial_encoder=original.initial_encoder,
                                                 initial_heads=original.initial_heads)
    factory=routes.RouteFactory(table,route='A',seed=42,expected_identity=expected,source_output=path)
    # From here source training or a source forward is forbidden.
    def forbidden(*a,**kw):pytest.fail('CPU reuse cannot retrain or replay source')
    monkeypatch.setattr(source_mod.SourceTrainer,'run',forbidden)
    monkeypatch.setattr(source_mod.SourceTrainer,'probe',forbidden)
    trainer=probe.make_trainer(factory,'A','cpu')
    counts={t:len(ds) for t,ds in trainer.datasets['train'].items()}
    monkeypatch.setattr(probe.c0,'tasks_and_counts',lambda s:(list(counts),counts))
    lock=tmp_path/'lock.json';write(lock,dict(initial_encoders={'A':v9_initial}));monkeypatch.setattr(probe,'LOCK',lock)
    job=dict(setting='A',method=method,seed=42,id=f'A_{method}_s42')
    out=tmp_path/'candidate';out.mkdir()
    result=probe.one_job(factory,trainer,job,out,'a'*40,'cpu')
    fresh=probe.make_trainer(factory,'A','cpu')
    model=probe.DualGraph(fresh.model,method,counts);opt=probe.optimizer_for(model)
    probe.load_smoke(out/'smoke_state.pt',model,opt,result['identity'],12)
    assert probe.digest(probe.TrainSupport(probe.training_records(factory,fresh,'A')).identity)==probe.digest(result['identity']['support'])
    assert result['train_replay_identical'] is True and result['restore_updates']==0
    assert result['initial_encoder']==result['final_encoder']==v9_initial
    torch.set_num_threads(before)


def test_gpu_dispatch_retains_default_source_replay():
    import v9_c1_probe as probe
    calls=[]
    class Factory:
        def make_trainer(self,**kw):calls.append(kw)
    probe.make_trainer(Factory(),'A','cpu')
    probe.make_trainer(Factory(),'A','cuda:0')
    assert calls==[dict(original=True,device='cpu',source_content_only=True),dict(original=True,device='cuda:0')]
