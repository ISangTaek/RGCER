from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import torch
import pytest
from dataset115_contract import ContractError,semantic_digest
from dataset115_route_a_test import load_lock,export_all,verify_saved,validate_lock,TASK_ID,TRAINING_COMMIT,LOCK_SHA
from tests.test_dataset115_test_export import Table,fixture,records


class ATable(Table):
    def view(self,route,role,split):
        assert route=='A'
        return replace(super().view(route,role,split),route='A')


def a_fixture(tmp_path):
    _,lock=fixture(tmp_path)
    lock.update(schema='s5g_route_a_selected_v1',training_commit=TRAINING_COMMIT)
    for r in lock['runs']:
        r['identity']['schema']='dataset115_route_a_epoch_v1';r['identity']['scaler']['route']='A'
        r['identity_sha256']=semantic_digest(r['identity'])
        p=torch.load(r['checkpoint_path'],weights_only=True)
        p['identity']=r['identity'];p['identity_sha256']=r['identity_sha256'];torch.save(p,r['checkpoint_path'])
        from pathlib import Path
        raw=Path(r['checkpoint_path']).read_bytes();r['size_bytes']=len(raw);r['checkpoint_sha256']=hashlib.sha256(raw).hexdigest()
    return ATable(),lock


def test_a_complete_fifteen_regressions_before_any_test(tmp_path):
    table,lock=a_fixture(tmp_path);calls=[]
    def predict(state,row,view,device,cache):calls.append((view.route,view.split));return records(view)
    result=export_all(lock,table,tmp_path/'out',device='cpu',predictor=predict)
    assert calls==[('A','validation')]*15+[('A','test')]*15
    assert result['task_id']==TASK_ID and result['selection_lock_sha256']==LOCK_SHA
    assert len(result['runs'])==15 and result['training_performed'] is False
    assert verify_saved(lock,table,tmp_path/'out')['runs_checked']==15


@pytest.mark.parametrize('case',['label','metric','sha','selection'])
def test_a_saved_content_independent_recheck(tmp_path,case):
    table,lock=a_fixture(tmp_path);out=tmp_path/'out'
    export_all(lock,table,out,device='cpu',predictor=lambda state,row,v,device,cache:records(v))
    path=out/'test_summary.json';summary=json.loads(path.read_text());r=summary['runs'][0]
    if case=='label':
        p=out/r['prediction_file'];v=json.loads(p.read_text());v[0]['label']+=1;p.write_text(json.dumps(v))
        r['prediction_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    elif case=='metric':r['metrics']['macro_rmse']=0
    elif case=='sha':r['prediction_sha256']='0'*64
    else:r['best_epoch']=39
    path.write_text(json.dumps(summary))
    with pytest.raises(ContractError):verify_saved(lock,table,out)


@pytest.mark.parametrize('failure_at',[0,14])
def test_a_validation_failure_stops_before_test(tmp_path,failure_at):
    table,lock=a_fixture(tmp_path);calls=[]
    def predict(state,row,view,device,cache):
        calls.append(view.split);out=records(view)
        if len(calls)==failure_at+1:out[0]['prediction']+=.1
        return out
    with pytest.raises(ContractError,match='regression'):export_all(lock,table,tmp_path/'out',device='cpu',predictor=predict)
    assert calls==['validation']*(failure_at+1)


@pytest.mark.parametrize('case',['route','schema','seed','best','matrix'])
def test_a_lock_scope_rejection(tmp_path,case):
    _,lock=a_fixture(tmp_path)
    if case=='route':lock['runs'][0]['identity']['scaler']['route']='B'
    elif case=='schema':lock['runs'][0]['identity']['schema']='dataset115_route_b_epoch_v1'
    elif case=='seed':lock['runs'][0]['seed']=True
    elif case=='best':lock['runs'][0]['best_epoch']=True
    else:lock['runs'].pop()
    with pytest.raises(ContractError):validate_lock(lock)


def test_a_unapproved_lock_bytes_rejected(tmp_path):
    p=tmp_path/'lock.json';p.write_text('{}')
    with pytest.raises(ContractError,match='unapproved'):load_lock(p)


def test_a_cli_no_training_or_subset_override():
    from scripts.export_dataset115_route_a_test import parser
    base=[]
    for k in ('csv','split-manifest','tox-manifest','selection-lock','expected-commit','output'):base+=['--'+k,'value']
    for extra in [['--epochs','40'],['--split','calibration'],['--seed','42'],['--device','cpu'],['--route','B']]:
        with pytest.raises(SystemExit):parser().parse_args(base+extra)
