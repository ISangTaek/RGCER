from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from dataset115_adapter import LabelView
from dataset115_contract import ContractError,PRIMARY
from dataset115_route_a_smoke import check_source_tasks,build_source,inventory,run_smoke,ARCH
from tests.test_dataset115_adapter import view,args

TASKS=tuple(f'rat_route{i}_LD50' for i in range(104))
class Table:
    identity='a'*64
    source_tasks=TASKS
    def __init__(self):self.requests=[]
    def view(self,route,role,split):
        self.requests.append((route,role,split));assert route=='A' and split=='train'
        v=replace(view(),route='A')
        if role=='target':return v
        return LabelView('A','source','train',TASKS,v.sample_ids,v.smiles,v.canonical,v.groups,
                         np.tile([[0.],[2.]],(1,104)),self.identity)

@pytest.mark.parametrize('case',['short','duplicate','human','bad_name'])
def test_source_task_rejection(case):
    t=list(TASKS)
    if case=='short':t.pop()
    if case=='duplicate':t[1]=t[0]
    if case=='human':t[0]='women_oral_TDLo'
    if case=='bad_name':t[0]='rat'
    with pytest.raises(ContractError):check_source_tasks(t)

def test_inventory_only_requests_a_train_and_records_costs():
    table=Table();s,t,sc,r=inventory(table)
    assert table.requests==[('A','source','train'),('A','target','train')]
    assert r['batch32_updates_per_epoch']=={'source':104,'target':5}
    assert len(sc['source'].trainer_scalers())==104 and len(sc['target'].trainer_scalers())==5
    assert not set(s.tasks)&set(t.tasks) and s.sample_ids==t.sample_ids

def test_empty_source_task_is_not_dropped():
    class Empty(Table):
        def view(self,*a):
            v=super().view(*a)
            if a[1]=='source':
                x=v.labels.copy();x[:,0]=np.nan;return replace(v,labels=x)
            return v
    with pytest.raises(ValueError,match='No finite training'):inventory(Empty())

def test_wrong_population_and_route_fail():
    class Wrong(Table):
        def view(self,*a):
            v=super().view(*a)
            return replace(v,groups=('changed','groups')) if a[1]=='target' else v
    with pytest.raises(ContractError,match='population'):inventory(Wrong())

def test_source_factory_seed_and_scope():
    model=build_source(args(),TASKS,42)
    assert list(model.decoders)==list(TASKS)
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ContractError,match='seed'):build_source(args(),TASKS,True)
    with pytest.raises(ContractError):build_source(args(),TASKS,47)

def test_full_source_target_chain_synthetic_cpu(tmp_path):
    result=run_smoke(Table(),tmp_path/'run',device='cpu',args=args())
    assert result['formal_source_reusable'] is False and result['formal_training_ready'] is False
    assert len(result['source']['head_changed'])==104 and all(result['source']['head_changed'].values())
    assert all(v==0 for v in result['source']['reload_max_error'].values())
    assert result['target_runs'][1]['encoder_before']==result['source']['encoder_after']==result['target_runs'][2]['encoder_after']
    assert result['target_runs'][1]['encoder_after']!=result['source']['encoder_after']
    p=torch.load(tmp_path/'run/source_smoke.pt',weights_only=True,map_location='cpu')
    assert p['formal_source'] is False and p['steps']==1 and p['identity']['train_molecules']==2
    assert not set(p['task_names'])&set(PRIMARY)
    assert result['test_predictions_accessed'] is False
    assert json.loads((tmp_path/'run/smoke_result.json').read_text())==result
    from dataset115_route_a_smoke import verify_saved
    info=json.loads((tmp_path/'run/data_inventory.json').read_text())
    _,_,sc,_=inventory(Table())
    assert verify_saved(tmp_path/'run',info,sc['target'].trainer_scalers())['target_checkpoints']==3
    result['source']['encoder_after']='0'*64
    (tmp_path/'run/smoke_result.json').write_text(json.dumps(result))
    with pytest.raises(ContractError,match='tensor identity'):verify_saved(tmp_path/'run',info,sc['target'].trainer_scalers())

def test_existing_output_kept(tmp_path):
    with pytest.raises(ContractError,match='exists'):run_smoke(Table(),tmp_path,device='cpu',args=args())

def test_cli_has_no_epoch_test_or_seed_override():
    from scripts.smoke_dataset115_route_a import parser
    names={a.dest for a in parser()._actions}
    assert not names&{'epochs','seed','split','device','method'}
    assert ARCH['a_layers']==8 and ARCH['edge_bias_mode']=='path' and ARCH['hidden_dim']==96

@pytest.mark.parametrize('case',['commit','dirty','cpu'])
def test_cli_fail_closed_before_data_or_training(monkeypatch,tmp_path,case):
    from scripts import smoke_dataset115_route_a as cli
    expected='a'*40
    def git(cmd,**kw):
        if cmd[-1]=='--porcelain':return ' M changed.py' if case=='dirty' else ''
        return 'b'*40 if case=='commit' else expected
    monkeypatch.setattr(cli.subprocess,'check_output',git)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    target=tmp_path/'absent'
    assert cli.main(['--csv','missing','--split-manifest','missing','--tox-manifest','missing',
        '--expected-commit',expected,'--output',str(target)])==2
    assert not target.exists()
