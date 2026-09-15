import copy
import math
import numpy as np
import pytest
import s4e_formal as m
from s4e_mechanism_design import DesignError,write_json,read_json,sha
from s4e_mechanism_metrics import linear_cka,functional_forgetting
from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS


def docs():
    runs=[dict(run_id=f'{a}_{s}_{f}',method=a,seed=s,declared_fraction_percent=f,
        teacher_asset_id='teacher',init_asset_id='init',best={'sha256':'b'*64})
        for a in ('B1','RPT') for s in range(42,47) for f in (10,25,50,75,100)]
    return {'mechanism_asset_lock.json':{'runs':runs}}


def test_exact_matrix():
    d=docs();assert len(m.matrix(d))==50
    for bad in ('missing','duplicate','wrongseed'):
        x=copy.deepcopy(d);r=x['mechanism_asset_lock.json']['runs']
        if bad=='missing':r.pop()
        elif bad=='duplicate':r[-1]=r[0]
        else:r[-1]['seed']=47
        with pytest.raises(DesignError):m.matrix(x)


def test_drift_complete_partition():
    ref={m.PREFIX+f'layers.{i}.x':np.ones(2) for i in range(8)};ref[m.PREFIX+'final_norm.x']=np.ones(1)
    out=m.grouped_drift(ref,{k:2*v for k,v in ref.items()})
    assert len(out)==10 and out['global']['relative_l2']==1
    assert sum(len(v['keys']) for k,v in out.items() if k!='global')==len(ref)
    bad=dict(ref);bad[m.PREFIX+'layers.9.x']=np.ones(1)
    with pytest.raises(DesignError):m.grouped_drift(bad,bad)


def test_formal_verifier_and_coherently_rehashed_tampering(tmp_path,monkeypatch):
    d=docs();keys=[m.PREFIX+f'layers.{i}.x' for i in range(8)]+[m.PREFIX+'final_norm.x']
    d['mechanism_asset_lock.json']['assets']=[{'asset_id':'init','sha256':'i'*64,'backbone':{'tensors':[{'key':k} for k in keys]}},
        {'asset_id':'teacher','sha256':'t'*64,'architecture_config':{'hidden_dim':6}}]
    ids=[str(i) for i in range(228)];d['probe_manifest.json']={'ordered_samples':[{'sample_key':i} for i in ids]}
    tasks=[];rows=[];labels={};sp={};hp={};eid={}
    for j,t in enumerate(ANIMAL_SOURCE_TASKS):
        count=14090//56+(j<14090%56)
        obs=[dict(sample_id=str(i),global_index=i,label_raw=0.,mask=True) for i in range(count)]
        tasks.append({'task':t,'observations':obs});eid[t]=[str(i) for i in range(count)]
        labels[t]=[0.]*count;sp[t]=(eid[t],[0.]*count);hp[t]=(eid[t],[1.]*count)
        rows.extend(dict(o,task=t,source_prediction=0.,hybrid_prediction=1.) for o in obs)
    d['animal56_validation_manifest.json']={'tasks':tasks}
    monkeypatch.setattr(m,'load_design',lambda *a:({},d))
    scope=dict(run_id='B1_42_10',scope='FORMAL_S4E5',method='B1',seed=42,fraction_percent=10,
        design_sha256='design',authorization_sha256='auth',probe_ids=ids,
        source={'sha256':'t'*64},init={'sha256':'i'*64},best={'sha256':'b'*64})
    write_json(tmp_path/'scope.json',scope);write_json(tmp_path/'predictions.json',rows)
    x=np.random.RandomState(4).randn(228,6);layers=[f'block_{i}' for i in range(8)]+['final_graph_token']
    np.savez_compressed(tmp_path/'embeddings.npz',**{role+'_'+k:x for role in ('source','hybrid') for k in layers})
    ref={k:np.ones(1) for k in keys};drift=m.grouped_drift(ref,ref)
    norms={k:m.parameter_drift({k:ref[k]},{k:ref[k]}) for k in keys}
    metrics=dict(scope='FORMAL_S4E5',drift=drift,cka={k:linear_cka(x,x,ids,ids) for k in layers},
        functional=functional_forgetting(labels,sp,hp,list(ANIMAL_SOURCE_TASKS),eid))
    write_json(tmp_path/'metrics.json',metrics);write_json(tmp_path/'parameter_norms.json',norms)
    m.output_checksums(tmp_path)
    assert m.verify(tmp_path,'design',tmp_path,'auth')['observations']==14090
    # Wrong label with matching file hashes must still be rejected.
    rows[0]['label_raw']=2.;(tmp_path/'predictions.json').write_text(__import__('json').dumps(rows))
    checks=read_json(tmp_path/'checksums.json');checks['predictions.json']=sha(tmp_path/'predictions.json')
    (tmp_path/'checksums.json').write_text(__import__('json').dumps(checks))
    with pytest.raises(DesignError,match='label or sample'):m.verify(tmp_path,'design',tmp_path,'auth')


def test_full_export_orchestration_with_synthetic_weights(tmp_path,monkeypatch):
    """Exercise all 228/14090 records; synthetic data is not GPU evidence."""
    import torch,sys
    from types import SimpleNamespace
    import toxacute_datastore
    from torch_geometric.data import Data
    from s4e_mechanism_smoke import SourceModel
    a=dict(architecture='Graphormer',prediction_mode='quantile',task_names=list(ANIMAL_SOURCE_TASKS),
        card_enabled=False,hidden_dim=8,a_heads=2,a_layers=8,mid_dim=12,spatial_pos_max_clip=20,
        edge_bias_mode='path',head_hidden_dim=8,head_dropout=0.1)
    torch.set_num_threads(1);torch.manual_seed(9);state=SourceModel(a).state_dict()
    initial={k:v for k,v in state.items() if k.startswith(m.PREFIX)}
    d=docs();teacher={'asset_id':'teacher','sha256':'t'*64,'architecture_config':a,
        'scalers':[dict(task=t,mean=0.,std=1.) for t in ANIMAL_SOURCE_TASKS]}
    init={'asset_id':'init','sha256':'i'*64,'backbone':{'tensors':[{'key':k} for k in initial]}}
    d['mechanism_asset_lock.json']['assets']=[teacher,init]
    samples=[dict(sample_key=f'probe_{i}',canonical_smiles='CC',input_record=dict(
        source='toxacute',global_index=i,sample_id=f'row_{i}',num_nodes=2)) for i in range(228)]
    d['probe_manifest.json']={'ordered_samples':samples}
    tasks=[]
    for j,t in enumerate(ANIMAL_SOURCE_TASKS):
        n=14090//56+(j<14090%56)
        tasks.append({'task':t,'observations':[dict(sample_id=f'row_{i}',global_index=i,label_raw=float(i%3),mask=True) for i in range(n)]})
    d['animal56_validation_manifest.json']={'tasks':tasks}
    monkeypatch.setattr(m,'load_design',lambda *args:({'datastore_semantic_identity':{}},d))
    monkeypatch.setattr(m,'datastore_semantic_identity',lambda p:{})
    def loaded(repo,asset):
        st=initial if asset.get('asset_id')=='init' else state
        return {'task_scalers':{t:{'mean':0.,'std':1.} for t in ANIMAL_SOURCE_TASKS}},st,{'sha256':asset['sha256']}
    monkeypatch.setattr(m,'state_from_asset',loaded)
    def graph(i):
        return Data(x=torch.zeros((2,9),dtype=torch.long),in_degree=torch.ones(2,dtype=torch.long),
            out_degree=torch.ones(2,dtype=torch.long),spatial_pos=torch.tensor([[0,1],[1,0]]),
            attn_edge_type=torch.zeros((2,2,4),dtype=torch.long),edge_input=torch.zeros((2,2,2,4),dtype=torch.long),
            sample_id=f'row_{i}',canonical_smiles='CC',y=torch.tensor([float(i%3)]),feature_schema_version='atom_v2_bond_v1_pathavg_v1')
    class Store:
        metadata={'max_path_distance':8};sample_ids=[f'row_{i}' for i in range(252)]
        def __init__(self,p):pass
        def get_graph_data(self,i):return graph(i)
        def get_label(self,i,t):return float(i%3)
        def close(self):pass
    monkeypatch.setattr(toxacute_datastore,'ToxAcuteDataStore',Store)
    monkeypatch.setitem(sys.modules,'preprocess_data',SimpleNamespace(get_graph_data_from_smiles=lambda *a,**k:None))
    out=tmp_path/'formal'
    m.evaluate(tmp_path,tmp_path,'design',tmp_path,'RPT_42_10',out,'cpu','auth')
    v=m.verify(tmp_path,'design',out,'auth')
    assert v['observations']==14090
    result=read_json(out/'metrics.json');assert result['functional']['macro_delta_rmse']==0
    assert len(result['drift'])==10
    with pytest.raises(DesignError,match='existing output'):
        m.evaluate(tmp_path,tmp_path,'design',tmp_path,'RPT_42_10',out,'cpu','auth')
