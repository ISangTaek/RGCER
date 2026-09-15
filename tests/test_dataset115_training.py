from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
import torch

from dataset115_contract import PRIMARY, ContractError
from dataset115_model import build_human5_model
from dataset115_training import EpochConfig, RouteBTrainer, task_batches, validation_metrics
from tests.test_dataset115_adapter import args, view


def validation_view():
    return replace(view('validation'), sample_ids=('dataset115:row_2','dataset115:row_3'),
                   smiles=('CO','CN'), canonical=('CO','CN'), groups=('CO','CN'))


def runner(method='B1', **changes):
    source=build_human5_model(args(),method='B0',seed=123).encoder.state_dict()
    kw=dict(method=method,seed=42,config=EpochConfig(3,1,0.001,1e-5,1),device='cpu')
    kw.update(changes)
    return RouteBTrainer(args(),None if method=='B0' else source,
        None if method=='B0' else dict(seed=42,teacher_sha256='1'*64,init_sha256='2'*64),
        view(),validation_view(),**kw)


@pytest.mark.parametrize('method',['B0','B1','RPT'])
def test_epoch_training_and_exact_resume(tmp_path,method):
    # Compare uninterrupted to a new-directory continuation including dropout.
    full=runner(method).run(tmp_path/'full')
    first=runner(method).run(tmp_path/'first',stop_after=1)
    path=tmp_path/'first'/first['checkpoints'][-1]['path']
    resumed=runner(method).run(tmp_path/'resume',resume=path,resume_sha=first['checkpoints'][-1]['sha256'])
    assert full['history']==resumed['history']
    assert full['best_validation']==resumed['best_validation']
    a=torch.load(tmp_path/'full/epoch_002.pt',weights_only=True)
    b=torch.load(tmp_path/'resume/epoch_002.pt',weights_only=True)
    assert a['model_digest']==b['model_digest']
    assert all(torch.equal(a['model_state'][k],b['model_state'][k]) for k in a['model_state'])
    assert full['best_epoch']==min(range(3),key=lambda i:full['history'][i]['validation']['macro_rmse'])
    assert all(h['exposure']==dict.fromkeys(PRIMARY,2) for h in full['history'])
    assert full['history'][-1]['total_steps']==30
    assert not full['test_predictions_accessed'] and not full['calibration_predictions_accessed']


def test_schedule_pairing_coverage():
    counts=dict(zip(PRIMARY,[30,29,64,61,77]))
    s=task_batches(counts,32,42,0)
    assert s==task_batches(counts,32,42,0)
    assert s!=task_batches(counts,32,43,0)
    for t in PRIMARY:assert sorted(i for task,ids in s if task==t for i in ids)==list(range(counts[t]))
    assert len(s)==9


@pytest.mark.parametrize('change',[dict(epochs=True),dict(epochs=0),dict(batch_size=0),
    dict(learning_rate=float('nan')),dict(weight_decay=-1),dict(grad_clip=0)])
def test_config_rejected(change):
    kw=dict(epochs=3,batch_size=2,learning_rate=0.001,weight_decay=1e-5,grad_clip=1);kw.update(change)
    with pytest.raises(ContractError):EpochConfig(**kw)


def test_metrics_original_scale_and_macro():
    rows=[dict(task=t,split='validation',sample_id='dataset115:row_0',label=1.,prediction=float(j+1)) for j,t in enumerate(PRIMARY)]
    m=validation_metrics(rows)
    assert m['macro_rmse']==2 and m['macro_mae']==2
    with pytest.raises(ContractError,match='duplicate'):validation_metrics(rows+rows)
    with pytest.raises(ContractError,match='missing'):validation_metrics(rows[:-1])
    with pytest.raises(ContractError,match='nonfinite'):validation_metrics([dict(r,prediction=float('nan')) for r in rows])


@pytest.mark.parametrize('split',['test','calibration','train'])
def test_target_split_not_selection_input(split):
    with pytest.raises(ContractError,match='train/validation only'):
        RouteBTrainer(args(),None,None,view(),replace(validation_view(),split=split),
            method='B0',seed=42,config=EpochConfig(1,2,.001,0,1),device='cpu')


def test_leakage_rejected():
    with pytest.raises(ContractError,match='leakage'):
        RouteBTrainer(args(),None,None,view(),view('validation'),method='B0',seed=42,
            config=EpochConfig(1,2,.001,0,1),device='cpu')


def test_resume_identity_sha_and_no_overwrite(tmp_path):
    r=runner();summary=r.run(tmp_path/'first',stop_after=1)
    path=tmp_path/'first/epoch_000.pt';sha=summary['checkpoints'][0]['sha256']
    with pytest.raises(ContractError,match='exists'):runner().run(tmp_path/'first')
    with pytest.raises(ContractError,match='SHA'):runner().run(tmp_path/'bad',resume=path,resume_sha='0'*64)
    with pytest.raises(ContractError,match='contract'):
        runner(config=EpochConfig(4,1,.001,1e-5,1)).run(tmp_path/'changed',resume=path,resume_sha=sha)
    payload=torch.load(path,weights_only=True);payload['best_epoch']=1
    bad=tmp_path/'tampered.pt';torch.save(payload,bad)
    with pytest.raises(ContractError,match='best selection'):
        runner().run(tmp_path/'tampered',resume=bad,resume_sha=hashlib.sha256(bad.read_bytes()).hexdigest())


def test_best_tie_keeps_first(tmp_path,monkeypatch):
    r=runner();rows=[dict(task=t,split='validation',sample_id='dataset115:row_2',label=1.,prediction=2.) for t in PRIMARY]
    monkeypatch.setattr(r,'evaluate_validation',lambda:(rows,validation_metrics(rows)))
    summary=r.run(tmp_path/'run')
    assert summary['best_epoch']==0


@pytest.mark.parametrize('case',['pass','label','missing','best','checkpoint'])
def test_verifier_reads_real_artifacts(tmp_path,case):
    from dataset115_route_b_run import verify_training_run
    r=runner(config=EpochConfig(2,2,.001,1e-5,1))
    out=tmp_path/'run';r.run(out)
    fresh=runner(config=EpochConfig(2,2,.001,1e-5,1))
    if case=='pass':
        checked=verify_training_run(out,fresh)
        assert checked['epochs_checked']==2 and checked['validation_observations']==10
        return
    if case in ('label','missing'):
        p=out/'validation_epoch_000.json';d=json.loads(p.read_text())
        if case=='label':d[0]['label']+=1
        else:d.pop()
        p.write_text(json.dumps(d))
    elif case=='best':
        p=out/'training_summary.json';d=json.loads(p.read_text());d['best_epoch']=99;p.write_text(json.dumps(d))
    else:(out/'epoch_000.pt').write_bytes(b'not a checkpoint')
    with pytest.raises(ContractError):verify_training_run(out,fresh)


def test_cli_forbids_scope_expansion():
    from scripts.train_dataset115_route_b import parser
    base=['--csv','c','--split-manifest','s','--tox-manifest','t','--source-lock','l',
          '--expected-commit','a'*40,'--output','o','--method','B0','--seed','42']
    assert parser().parse_args(base).seed==42
    for extra in (['--epochs','80'],['--split','test'],['--device','cpu'],['--seed','47'],['--method','RouteA']):
        with pytest.raises(SystemExit):parser().parse_args(base+extra)


def test_batch_matrix_and_argv():
    from scripts.run_dataset115_route_b_batch import MATRIX,make_command
    from types import SimpleNamespace
    from pathlib import Path
    assert len(MATRIX)==15 and len(set(MATRIX))==15
    assert set(MATRIX)=={(m,s) for m in ('B0','B1','RPT') for s in range(42,47)}
    a=SimpleNamespace(csv='path with spaces.csv',split_manifest='split',tox_manifest='tox',
        source_lock='lock',expected_commit='1'*40)
    cmd=make_command('python',Path('/repo'),a,'RPT',46,Path('/new output'))
    assert cmd[cmd.index('--csv')+1]=='path with spaces.csv'
    assert cmd[-4:]==['--seed','46','--output',str(Path('/new output'))]


def test_batch_scheduler_single_worker_stops_without_retry(tmp_path,monkeypatch):
    from scripts import run_dataset115_route_b_batch as batch
    p=tmp_path/'input';p.write_text('fixture')
    calls=[]
    def failing(cmd,**kw):
        from types import SimpleNamespace
        calls.append((cmd,kw['env']['CUDA_VISIBLE_DEVICES']))
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(batch.subprocess,'run',failing)
    out=tmp_path/'batch'
    argv=[]
    for key in ('csv','split-manifest','tox-manifest','source-lock'):argv+=['--'+key,str(p)]
    argv+=['--expected-commit','1'*40,'--output',str(out),'--gpus','0']
    assert batch.main(argv)==2 and len(calls)==1 and calls[0][1]=='0'
    summary=json.loads((out/'batch_summary.json').read_text())
    assert len(summary['not_started'])==14 and summary['automatic_retries']==0
