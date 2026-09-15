from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib,json
import numpy as np
import pytest,torch
from dataset115_contract import ContractError,PRIMARY,semantic_digest
from dataset115_adapter import TrainOnlyScaler
from dataset115_test_export import (load_lock,load_selected,metrics,compare_validation,export_all,MATRIX)
from dataset115_smoke import state_digest
from tests.test_dataset115_adapter import view

class Table:
    identity='a'*64
    def view(self,route,role,split):
        v=view(split);offset={'train':0,'validation':2,'test':4}[split]
        return replace(v,sample_ids=tuple(f'dataset115:row_{offset+i}' for i in range(2)))

def records(v):
    return [dict(task=t,split=v.split,sample_id=v.sample_ids[i],label=float(v.labels[i,j]),
                 canonical=v.canonical[i],group=v.groups[i],prediction=float(v.labels[i,j])+.1,
                 lower=float(v.labels[i,j])-1,upper=float(v.labels[i,j])+1)
            for j,t in enumerate(PRIMARY) for i in range(2)]

def fixture(tmp_path):
    table=Table();runs=[]
    for m,s in sorted(MATRIX):
        state={'x':torch.ones(2)}
        identity=dict(method=m,seed=s,input_identity=table.identity,scaler=TrainOnlyScaler.fit(table.view('B','target','train')).to_dict())
        v=records(table.view('B','target','validation'))
        p=dict(identity=identity,identity_sha256=semantic_digest(identity),epoch=39,best_epoch=3,
               best_model_digest=state_digest(state),best_model_state=state,best_rows=v)
        path=tmp_path/f'{m}{s}.pt';torch.save(p,path);b=path.read_bytes()
        runs.append(dict(method=m,seed=s,checkpoint_path=str(path),checkpoint_sha256=hashlib.sha256(b).hexdigest(),
            size_bytes=len(b),identity=identity,identity_sha256=p['identity_sha256'],best_epoch=3,
            best_model_digest=p['best_model_digest'],validation_rows=v))
    return table,dict(runs=runs)

def test_unknown_lock_fails_before_parse(tmp_path):
    p=tmp_path/'lock';p.write_text('{}')
    with pytest.raises(ContractError,match='unapproved'):load_lock(p)

def test_locked_weights_not_last_or_changed(tmp_path):
    _,lock=fixture(tmp_path);r=lock['runs'][0]
    assert set(load_selected(r['checkpoint_path'],r))=={'x'}
    changed=deepcopy(r);changed['best_epoch']=39
    with pytest.raises(ContractError,match='selected best'):load_selected(r['checkpoint_path'],changed)
    Path(r['checkpoint_path']).write_bytes(b'changed')
    with pytest.raises(ContractError,match='bytes'):load_selected(r['checkpoint_path'],r)

@pytest.mark.parametrize('case',['duplicate','missing','label','group','task','nan','quantile','split','bool'])
def test_prediction_content_rejects_wrong_records(case):
    v=Table().view('B','target','test');rows=records(v)
    if case=='duplicate':rows.append(rows[0])
    elif case=='missing':rows.pop()
    elif case=='label':rows[0]['label']+=1
    elif case=='group':rows[0]['group']='wrong'
    elif case=='task':rows[0]['task']='wrong'
    elif case=='nan':rows[0]['prediction']=float('nan')
    elif case=='quantile':rows[0]['upper']=-100
    elif case=='bool':rows[0]['prediction']=True
    else:rows[0]['split']='validation'
    with pytest.raises(ContractError):metrics(rows,v)

def test_validation_tolerance_and_failure():
    rows=records(Table().view('B','target','validation'));other=deepcopy(rows)
    other[0]['prediction']+=1e-8
    assert compare_validation(other,rows)['status']=='PASS'
    other[0]['prediction']+=.01
    with pytest.raises(ContractError,match='regression'):compare_validation(other,rows)

def test_all_regressions_precede_test_and_output_verified(tmp_path):
    table,lock=fixture(tmp_path);calls=[]
    def predict(state,row,v,device,cache):calls.append(v.split);return records(v)
    result=export_all(lock,table,tmp_path/'out',device='cpu',predictor=predict)
    assert calls==['validation']*15+['test']*15
    assert len(result['runs'])==15 and not result['training_performed']
    assert result['runs'][0]['metrics']['macro_rmse']==pytest.approx(.1)
    with pytest.raises(ContractError,match='exists'):export_all(lock,table,tmp_path/'out',device='cpu',predictor=predict)

def test_failed_validation_never_enters_test(tmp_path):
    table,lock=fixture(tmp_path);calls=[]
    def wrong(state,row,v,device,cache):
        calls.append(v.split);out=records(v);out[0]['prediction']+=.01;return out
    with pytest.raises(ContractError,match='regression'):export_all(lock,table,tmp_path/'out',device='cpu',predictor=wrong)
    assert calls==['validation'] and not (tmp_path/'out/test_summary.json').exists()

def test_matrix_and_scaler_not_self_declared(tmp_path):
    table,lock=fixture(tmp_path);lock['runs'][0]['identity']['scaler']['route']='A'
    with pytest.raises(ContractError,match='scaler'):export_all(lock,table,tmp_path/'out',device='cpu')
    lock['runs'].pop()
    with pytest.raises(ContractError,match='matrix'):export_all(lock,table,tmp_path/'out',device='cpu')
