from types import SimpleNamespace
import pytest
import torch
from torch import nn
import v9_c2a_source_stats as mod

def records(n=40):
    return [dict(sample_id=str(i),canonical='molecule'+str(i),group='group'+str(i),split='train',index=i) for i in range(n)]

def test_hash_selection_is_order_invariant_and_capped():
    a,p=mod.select_records(records(300));b,_=mod.select_records(list(reversed(records(300))))
    assert a==b and len(a)==256 and p['eligible_unique']==300

@pytest.mark.parametrize('bad',['holdout','duplicate_id','cross_group','short','unknown'])
def test_source_membership_rejects_bad_inputs(bad):
    rows=records()
    if bad=='holdout':rows[0]['split']='validation'
    if bad=='duplicate_id':rows[1]['sample_id']=rows[0]['sample_id']
    if bad=='cross_group':rows[1]['canonical']=rows[0]['canonical']
    if bad=='short':rows=rows[:31]
    if bad=='unknown':rows[0]['label']=1
    with pytest.raises(ValueError):mod.select_records(rows)

def test_atoms_are_equal_within_molecules_and_molecules_equal_across_sizes():
    x=torch.tensor([[[999.],[2.],[999.],[999.]],[[999.],[4.],[4.],[4.]]])
    m=torch.tensor([[True,False,False],[True,True,True]])
    rows=mod.molecule_moments(x,m)
    assert (rows[0][0]+rows[1][0]).item()/2==3
    assert rows[0][1].item()==4 and rows[1][1].item()==16

@pytest.mark.parametrize('bad',['mask','empty','nonfinite'])
def test_activation_errors_stop(bad):
    x=torch.randn(2,4,16);m=torch.ones(2,3,dtype=torch.bool)
    if bad=='mask':m=m.float()
    if bad=='empty':m[0]=False
    if bad=='nonfinite':x[0,1,0]=float('nan')
    with pytest.raises(ValueError):mod.molecule_moments(x,m)

def test_covariance_shrinkage_and_rank16_projector():
    torch.manual_seed(42);x=torch.randn(40,24,dtype=torch.float64)
    r=mod.derive(x.sum(0),x.T@x,40);p=r['projector']
    assert torch.allclose(p@p,p,atol=1e-12) and abs(float(torch.trace(p))-16)<1e-10
    assert torch.linalg.eigvalsh(r['covariance']).min()>0

@pytest.fixture
def payload():
    torch.manual_seed(42);x=torch.randn(40,24,dtype=torch.float64)
    s=dict(count=[20,20],sum_mean=torch.stack([x[:20].sum(0),x[20:].sum(0)]),
           sum_second=torch.stack([x[:20].T@x[:20],x[20:].T@x[20:]]))
    s['derived']=[mod.derive(s['sum_mean'][j],s['sum_second'][j],20) for j in range(2)]
    s['pooled']=mod.derive(s['sum_mean'].sum(0),s['sum_second'].sum(0),40)
    identity=dict(selected=records());return dict(identity=identity,stats={'module':s}),identity

@pytest.mark.parametrize('bad',['identity','count','missing_module','projector','nan','shape'])
def test_validator_rejects_tampered_statistics(payload,bad):
    p,identity=payload
    if bad=='identity':p['identity']=dict(selected=[])
    if bad=='count':p['stats']['module']['count']=[19,21]
    if bad=='missing_module':p['stats'].clear()
    if bad=='projector':p['stats']['module']['pooled']['projector'].zero_()
    if bad=='nan':p['stats']['module']['sum_mean'][0,0]=float('nan')
    if bad=='shape':p['stats']['module']['sum_mean']=torch.ones(2,1)
    with pytest.raises(ValueError):mod.verify_payload(p,identity,{'module':24})

def test_statistics_roundtrip(payload,tmp_path):
    p,identity=payload;path=tmp_path/'moments.pt';torch.save(p,path)
    value=mod.verify_payload(torch.load(path,weights_only=True),identity,{'module':24})
    assert 0<=value['module']['half_projector_distance']<=1

def test_real_graph_hooks_collect_once_without_updates():
    from architecture.graphormer_backbone import MolecularGraphormerBackbone
    from preprocess_data import get_graph_data_from_smiles
    from dataset import DataCollator
    from reproducibility import state_dict_sha256
    before=torch.get_num_threads();torch.set_num_threads(1)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=MolecularGraphormerBackbone(hidden_dim=24,num_heads=4,num_layers=1,ffn_dim=24,dropout=.1)
        def forward(self,batch):return self.backbone(batch)
    encoder=Encoder();graphs=[get_graph_data_from_smiles('C'*(i+1),0.,sample_id=str(i),max_path_distance=8) for i in range(32)]
    rows=[dict(sample_id=str(i),canonical=g.canonical_smiles,group='g'+str(i),split='train',index=i) for i,g in enumerate(graphs)]
    identity=dict(selected=rows,encoder_sha256=state_dict_sha256(encoder))
    p,r=mod.collect(encoder,lambda row:graphs[row['index']],DataCollator(),identity,'cpu')
    assert r['forward_batches']==4 and r['molecules']==32 and r['optimizer_updates']==r['head_predictions']==0
    assert len(p['stats'])==4 and all(not m._forward_pre_hooks for m in mod.targets(encoder).values())
    mod.verify_payload(p,identity,{k:v.in_features for k,v in mod.targets(encoder).items()})
    torch.set_num_threads(before)

def test_parent_stops_on_worker_failure_and_keeps_claim(tmp_path,monkeypatch):
    import p1d4_batch
    monkeypatch.setattr(mod,'REPO',tmp_path);monkeypatch.setattr(mod.c0,'check_code',lambda *a:None)
    monkeypatch.setattr(mod,'gate',lambda *a:None);monkeypatch.setattr(mod,'prior_gate',lambda *a:None)
    monkeypatch.setattr(mod.c0,'gpu_uuid',lambda *a:'GPU-test');monkeypatch.setattr(p1d4_batch,'free_gpus',lambda x:x)
    for name in ('wsl','server'):(tmp_path/name).mkdir()
    calls=[]
    def child(*a,**kw):calls.append(a);assert 0<kw['timeout']<=1800;return SimpleNamespace(returncode=1)
    monkeypatch.setattr(mod.subprocess,'run',child)
    with pytest.raises(ValueError):mod.run(tmp_path/'run','a'*40,0,tmp_path/'split',tmp_path/'source',tmp_path/'wsl',tmp_path/'server',tmp_path/'prior')
    assert len(calls)==1 and (tmp_path/mod.REGISTRY/'attempt.json').exists() and (tmp_path/'run/failed.json').exists()

@pytest.mark.parametrize('change',['ok','changed','extra','missing'])
def test_accepted_073_gate_is_byte_and_population_locked(tmp_path,monkeypatch,change):
    root=tmp_path/'073';root.mkdir();mod.write(root/'verification.json',dict(accepted=True))
    lock=tmp_path/'lock.json';mod.write(lock,dict(accepted_073_files={'verification.json':mod.sha(root/'verification.json')}))
    monkeypatch.setattr(mod,'LOCK',lock)
    if change=='changed':(root/'verification.json').write_text('{}')
    if change=='extra':mod.write(root/'unknown.json',{})
    if change=='missing':(root/'verification.json').unlink()
    if change=='ok':mod.prior_gate(root)
    else:
        with pytest.raises(ValueError):mod.prior_gate(root)

def test_actual_pubchem_source_view_excludes_validation_and_unlabelled_rows(tmp_path,monkeypatch):
    import numpy as np
    import p1d4_runtime
    from dataset115_adapter import Dataset115Table
    from dataset115_contract import PRIMARY
    from tests.test_dataset115_route_a_smoke import TASKS
    from tests.test_dataset115_adapter import args
    from dataset115_model import build_human5_model
    from reproducibility import state_dict_sha256
    table=Dataset115Table();table.identity='a'*64;table.tasks=PRIMARY+tuple(TASKS);table.source_tasks=tuple(TASKS)
    table.overlaps=frozenset();table.records=tuple(dict(row_index=i,raw_smiles='C'*(i+1),canonical_smiles='C'*(i+1),split_group='g'+str(i),split='train' if i<33 else 'validation') for i in range(34))
    table.values=np.ones((34,len(table.tasks)));table.values[0,len(PRIMARY):]=np.nan
    model=build_human5_model(args(),method='B0',seed=42)
    monkeypatch.setattr(p1d4_runtime,'factory_for',lambda *a,**kw:SimpleNamespace(_table=table))
    monkeypatch.setattr(mod.c1,'make_trainer',lambda *a:SimpleNamespace(model=model))
    lock=tmp_path/'c1.json';mod.write(lock,dict(initial_encoders={'A':state_dict_sha256(model.encoder)}));monkeypatch.setattr(mod.c1,'LOCK',lock)
    _,graph,_,identity=mod.source_context('Nonhuman104',tmp_path/'split',tmp_path/'source')
    assert {r['index'] for r in identity['selected']}==set(range(1,33))
    row=identity['selected'][0];g=graph(row);assert g.sample_id==row['sample_id'] and g.canonical_smiles==row['canonical']
