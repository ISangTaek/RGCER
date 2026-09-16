from dataclasses import replace
import json
import math

import pytest
import torch

from dataset115_contract import ContractError, digest
from dataset115_smoke import state_digest
from dataset115_training import EpochConfig
import dataset115_route_a_training as mod
from tests.test_dataset115_route_a_smoke import Table, TASKS
from tests.test_dataset115_adapter import args, view
from tests.test_dataset115_training import validation_view

SMALL=EpochConfig(2,2,.001,1e-5,1.)

def source_runner(**kw):
    defaults=dict(seed=42,config=SMALL,device='cpu');defaults.update(kw)
    return mod.SourceTrainer(args(),Table().view('A','source','train'),**defaults)

def test_source_schedule_complete_and_paired():
    counts={t:i%7+1 for i,t in enumerate(TASKS)}
    a=mod.source_batches(TASKS,counts,3,42,0)
    assert a==mod.source_batches(TASKS,counts,3,42,0)
    assert a!=mod.source_batches(TASKS,counts,3,43,0)
    assert len(a)==sum(math.ceil(n/3) for n in counts.values())
    for t in TASKS:assert sorted(i for task,ids in a if task==t for i in ids)==list(range(counts[t]))

@pytest.mark.parametrize('split',['validation','test','calibration'])
def test_source_forbids_holdout(split):
    with pytest.raises(ContractError,match='source train only'):
        mod.SourceTrainer(args(),replace(Table().view('A','source','train'),split=split),seed=42)

@pytest.fixture
def trained(tmp_path):
    path=tmp_path/'source';source_runner().run(path)
    return path

def test_source_two_epochs_content_and_no_formal_reuse(trained):
    r=mod.verify_source(trained,source_runner())
    assert r['total_steps']==208 and r['selected_epoch']==1 and not r['formal_source']
    with pytest.raises(ContractError,match='nonformal'):mod.load_source(trained,source_runner())
    with pytest.raises(ContractError,match='exists'):source_runner().run(trained)

@pytest.mark.parametrize('field',['selected_epoch','human_validation_accessed','exposure','loss','sha'])
def test_source_summary_tamper_rejected(trained,field):
    path=trained/'training_summary.json';v=json.loads(path.read_text())
    if field=='selected_epoch':v[field]=0
    elif field=='human_validation_accessed':v[field]=True
    elif field=='exposure':v['history'][0]['exposure'][TASKS[0]]=1
    elif field=='loss':v['history'][0]['mean_batch_loss'][TASKS[0]]=float('nan')
    elif field=='sha':v['checkpoints'][1]['sha256']='0'*64
    path.write_text(json.dumps(v))
    with pytest.raises((ContractError,ValueError)):mod.verify_source(trained,source_runner())

def test_source_wrong_seed_and_raw_identity_rejected(trained):
    with pytest.raises(ContractError,match='config identity'):mod.verify_source(trained,source_runner(seed=43))
    r=source_runner();r.identity=dict(r.identity,train_ids='f'*64)
    with pytest.raises(ContractError,match='config identity'):mod.verify_source(trained,r)

def test_source_self_consistent_optimizer_tamper_rejected(trained):
    path=trained/'epoch_001.pt';p=torch.load(path,weights_only=True)
    # Keep file hashes and receipts self-consistent; content must still reject.
    p['optimizer_state']['state'][0]['step']=torch.tensor(99.)
    torch.save(p,path)
    rp=trained/'epoch_001.receipt.json';r=json.loads(rp.read_text());r.update(sha256=digest(path),size_bytes=path.stat().st_size)
    rp.write_text(json.dumps(r));sp=trained/'training_summary.json';s=json.loads(sp.read_text());s['checkpoints'][1]=r;sp.write_text(json.dumps(s))
    with pytest.raises(ContractError,match='optimizer steps'):mod.verify_source(trained,source_runner())

@pytest.mark.parametrize('method',['B0','B1','RPT'])
def test_route_a_target_training_preserves_contract(tmp_path,method):
    from dataset115_route_b_run import verify_training_run
    initial=source_runner().model.encoder.state_dict()
    identity=dict(route='A',seed=42,selection='fixed_final_epoch39',teacher_sha256='a'*64,
        init_sha256=state_digest(initial),source_identity_sha256='b'*64)
    def make():
        return mod.RouteATrainer(args(),None if method=='B0' else initial,None if method=='B0' else identity,
            replace(view(),route='A'),replace(validation_view(),route='A'),method=method,seed=42,config=SMALL,device='cpu')
    r=make();out=tmp_path/method;r.run(out)
    verified=verify_training_run(out,make())
    assert verified['epochs_checked']==2 and r.identity['schema']=='dataset115_route_a_epoch_v1'
    if method=='RPT':assert state_digest(r.model.encoder.state_dict())==identity['init_sha256']

def test_target_refuses_route_b_and_smoke_binding():
    with pytest.raises(ContractError,match='source binding'):
        mod.RouteATrainer(args(),{},dict(route='B'),replace(view(),route='A'),replace(validation_view(),route='A'),
            method='B1',seed=42,config=SMALL,device='cpu')


def test_full_budget_synthetic_source_can_bind(tmp_path):
    # Tiny synthetic molecules/architecture, not real GPU or scientific evidence.
    out=tmp_path/'full';source_runner(config=mod.CONFIG).run(out)
    encoder,binding,receipt=mod.load_source(out,source_runner(config=mod.CONFIG))
    assert receipt['formal_source'] is True and receipt['epochs_checked']==40
    assert receipt['selected_epoch']==39 and binding['seed']==42
    assert binding['init_sha256']==state_digest(encoder)
    assert binding['teacher_sha256']==digest(out/'epoch_039.pt')

def test_source_cache_bounded():
    r=source_runner()
    for j in range(2):
        for t in TASKS:r.batch(t,[j])
    assert len(r.cache)==208
    r.cache.update({('unused',j):None for j in range(60)})
    r.batch(TASKS[0],[0]);assert len(r.cache)==256


def test_cli_no_budget_or_holdout_override():
    from scripts.train_dataset115_route_a import parser
    base=['--csv','c','--split-manifest','s','--tox-manifest','t','--expected-commit','a'*40,
          '--output','o','--role','source','--seed','42']
    assert parser().parse_args(base).role=='source'
    for extra in (['--epochs','80'],['--split','test'],['--device','cpu'],['--seed','47'],['--resume','x']):
        with pytest.raises(SystemExit):parser().parse_args(base+extra)


@pytest.mark.parametrize('fail_at',[None,0,5])
def test_queue_source_barrier_and_fail_stop(tmp_path,monkeypatch,fail_at):
    from scripts import run_dataset115_route_a_batch as batch
    from types import SimpleNamespace
    f=tmp_path/'input';f.write_text('fixture');calls=[]
    def launch(cmd,**kw):
        i=len(calls);calls.append(cmd)
        assert kw['env']['CUDA_VISIBLE_DEVICES']=='0'
        return SimpleNamespace(returncode=2 if i==fail_at else 0)
    monkeypatch.setattr(batch.subprocess,'run',launch)
    argv=[]
    for k in ('csv','split-manifest','tox-manifest'):argv+=['--'+k,str(f)]
    out=tmp_path/'out';argv+=['--expected-commit','1'*40,'--output',str(out),'--gpus','0']
    assert batch.main(argv)==(0 if fail_at is None else 2)
    assert len(calls)==(20 if fail_at is None else fail_at+1)
    s=json.loads((out/'batch_summary.json').read_text());assert len(s['results'])+len(s['not_started'])==20
    for i,c in enumerate(calls):
        assert c[c.index('--role')+1]==('source' if i<5 else 'target')
        if '--method' in c and c[c.index('--method')+1]!='B0':
            assert c[c.index('--source-output')+1].endswith('source_seed'+c[c.index('--seed')+1])
