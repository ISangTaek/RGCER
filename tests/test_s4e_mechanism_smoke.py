from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from architecture.Graphormer import Encoder, Graphormer
from architecture.prediction_heads import TaskPredictionHead, decode_prediction
from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
from s4e_mechanism_design import DesignError, write_json, sha
from s4e_mechanism_smoke import SourceModel, embeddings, resolve_asset, load_design


def architecture():
    return dict(architecture='Graphormer', prediction_mode='quantile',
        task_names=list(ANIMAL_SOURCE_TASKS), card_enabled=False, hidden_dim=8,
        a_heads=2, a_layers=2, mid_dim=12, spatial_pos_max_clip=20,
        edge_bias_mode='path', head_hidden_dim=8, head_dropout=0.1)


def batch():
    return SimpleNamespace(x=torch.ones((3,2,9),dtype=torch.long),
        in_degree=torch.ones((3,2),dtype=torch.long), out_degree=torch.ones((3,2),dtype=torch.long),
        spatial_pos=torch.ones((3,2,2),dtype=torch.long), attn_bias=torch.zeros((3,3,3)),
        attn_edge_type=torch.ones((3,2,2,4),dtype=torch.long),
        edge_input=torch.ones((3,2,2,2,4),dtype=torch.long), padding_mask=torch.zeros((3,2),dtype=torch.bool))


def test_source_topology_matches_original_graphormer():
    torch.set_num_threads(1); torch.manual_seed(42)
    a=architecture(); new=SourceModel(a).eval()
    args=SimpleNamespace(**a, spatial_pos_clip=20)
    heads=nn.ModuleDict({t:TaskPredictionHead(8,mode='quantile',head_hidden_dim=8,dropout=0.1) for t in ANIMAL_SOURCE_TASKS})
    original=Graphormer(list(ANIMAL_SOURCE_TASKS),Encoder,heads,torch.device('cpu'),args).eval()
    original.load_state_dict(new.state_dict(),strict=True)
    with torch.no_grad():
        out=new(batch()); z=original.encoder(batch())
        for t in ANIMAL_SOURCE_TASKS:
            assert torch.equal(out[t],decode_prediction(original.decoders[t](z),mode='quantile').median[:,0])


def test_hooks_capture_graph_token_and_are_removed():
    model=SourceModel(architecture()).eval()
    out=embeddings(model,batch()); again=embeddings(model,batch())
    assert set(out)=={'block_0','block_1','final_graph_token'}
    assert all(x.shape==(3,8) for x in out.values())
    assert all(np.array_equal(out[k],again[k]) for k in out)
    assert all(not layer._forward_hooks for layer in model.encoder.backbone.layers)


@pytest.mark.parametrize('field,value', [('card_enabled',True),('architecture','Graphormer_rgcer'),('task_names',list(reversed(ANIMAL_SOURCE_TASKS)))])
def test_unsupported_source_rejected(field,value):
    a=architecture(); a[field]=value
    with pytest.raises(DesignError):SourceModel(a)


def test_asset_resolution_is_repo_scoped(tmp_path):
    assert resolve_asset(tmp_path,'/home/shangzeli/RGCER/artifacts/a.pt')==tmp_path/'artifacts/a.pt'
    with pytest.raises(DesignError):resolve_asset(tmp_path,'/other/a.pt')
    with pytest.raises(DesignError):resolve_asset(tmp_path,'/home/shangzeli/RGCER/../../a.pt')


def test_design_requires_external_sha_and_exact_members(tmp_path):
    names={'mechanism_asset_lock.json','probe_manifest.json','probe_eligibility.json','animal56_validation_manifest.json'}
    for n in names:write_json(tmp_path/n,{})
    write_json(tmp_path/'design_manifest.json',{'outputs':{n:sha(tmp_path/n) for n in names}})
    digest=sha(tmp_path/'design_manifest.json')
    load_design(tmp_path,digest)
    with pytest.raises(DesignError):load_design(tmp_path,'a'*64)
    (tmp_path/'probe_manifest.json').write_text('{"changed":true}')
    with pytest.raises(DesignError):load_design(tmp_path,digest)


def test_four_run_smoke_end_to_end_and_result_tamper(tmp_path,monkeypatch):
    """Synthetic weights/graphs exercise orchestration, NOT real GPU evidence."""
    import copy
    import sys
    import s4e_mechanism_smoke as module
    import toxacute_datastore
    from torch_geometric.data import Data
    from s4e_mechanism_design import read_json
    # No native graph builder or actual datastore is required for this test.
    def graph(smiles, label=0., *, sample_id, max_path_distance=8):
        return Data(x=torch.zeros((2,9),dtype=torch.long),
            in_degree=torch.ones(2,dtype=torch.long),out_degree=torch.ones(2,dtype=torch.long),
            spatial_pos=torch.tensor([[0,1],[1,0]]),
            attn_edge_type=torch.zeros((2,2,4),dtype=torch.long),edge_input=torch.zeros((2,2,2,4),dtype=torch.long),
            sample_id=sample_id, canonical_smiles=smiles, y=torch.tensor([label]),raw_smiles=smiles,
            feature_schema_version='atom_v2_bond_v1_pathavg_v1')
    monkeypatch.setitem(sys.modules,'preprocess_data',SimpleNamespace(get_graph_data_from_smiles=graph))
    def original_builder(seed,scope,device,configuration):
        a=architecture(); args=SimpleNamespace(**a,spatial_pos_clip=20)
        heads=nn.ModuleDict({t:TaskPredictionHead(8,mode='quantile',head_hidden_dim=8,dropout=0.1) for t in ANIMAL_SOURCE_TASKS})
        return Graphormer(list(ANIMAL_SOURCE_TASKS),Encoder,heads,device,args),None,list(ANIMAL_SOURCE_TASKS)
    monkeypatch.setitem(sys.modules,'scripts.d6_prep_inits',SimpleNamespace(_build_model=original_builder))
    class Store:
        metadata={'max_path_distance':8}
        sample_ids=['row_0','row_1']
        def __init__(self,path):pass
        def get_graph_data(self,i):return graph(['CC','CO'][i],sample_id=self.sample_ids[i])
        def get_label(self,i,t):return float(i+1)
        def close(self):pass
    monkeypatch.setattr(toxacute_datastore,'ToxAcuteDataStore',Store)
    monkeypatch.setattr(module,'datastore_semantic_identity',lambda p:{'test':True})
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='a'*40+'\n'))
    a=architecture(); torch.manual_seed(17); state=SourceModel(a).state_dict()
    scalers={t:{'mean':0.,'std':1.} for t in ANIMAL_SOURCE_TASKS}
    def save(name,payload,kind=None):
        p=tmp_path/name;torch.save(payload,p)
        return dict(path='/home/shangzeli/RGCER/'+name,sha256=sha(p),size_bytes=p.stat().st_size,
                    **({'kind':kind,'asset_id':kind+':'+sha(p)} if kind else {}))
    teacher=save('teacher.pt',{'model_state':state,'epoch':29,'task_scalers':scalers},'teacher')
    teacher.update(epoch=29,seed=42,architecture_config=a,configuration={},
        scalers=[dict(task=t,mean=0.,std=1.) for t in ANIMAL_SOURCE_TASKS])
    init=save('init.pt',{k:v for k,v in state.items() if k.startswith('encoder.')},'init')
    runs=[]
    for frac in (50,100):
        for method in ('B1','RPT'):
            target=copy.deepcopy(state)
            if method=='B1':target['encoder.backbone.graph_token']+=0.01
            best=save(f'{method}{frac}.pt',{'model_state':target,'epoch':3});best['epoch']=3
            runs.append(dict(run_id=f'{method}_{frac}',method=method,seed=42,declared_fraction_percent=frac,
                teacher_asset_id=teacher['asset_id'],init_asset_id=init['asset_id'],best=best))
    probe=[]
    for i,s in enumerate(('human3_test','animal56_test_only','dataset115_external_test')):
        probe.append(dict(sample_key='key'+str(i),canonical_smiles=['CC','CO','CN'][i],stratum=s,
            input_record=dict(raw_smiles=['CC','CO','CN'][i],num_nodes=2,source='toxacute' if i<2 else 'dataset115',
                              global_index=i,sample_id=f'row_{i}')))
    val=[dict(task=t,observations=[dict(sample_id=f'row_{i}',global_index=i,raw_row_index=i,
            canonical_smiles=['CC','CO'][i],label_raw=float(i+1),mask=True) for i in range(2)]) for t in ANIMAL_SOURCE_TASKS]
    design=tmp_path/'design';design.mkdir()
    documents={'mechanism_asset_lock.json':dict(runs=runs,assets=[teacher,init]),
        'probe_manifest.json':dict(ordered_samples=probe),'probe_eligibility.json':{},
        'animal56_validation_manifest.json':dict(tasks=val)}
    for n,p in documents.items():write_json(design/n,p)
    write_json(design/'design_manifest.json',dict(outputs={n:sha(design/n) for n in documents},
        datastore_semantic_identity={'test':True}))
    digest=sha(design/'design_manifest.json');out=tmp_path/'out'
    module.smoke(tmp_path,design,digest,tmp_path/'unused',out,'cpu')
    result=module.verify_smoke(design,digest,out)
    assert result['run_count']==4 and result['observations_per_run']==112
    # A coherently rehashed wrong label must still fail against the design.
    rows=read_json(out/'B1_50/predictions.json');rows[0]['label_raw']+=1
    (out/'B1_50/predictions.json').write_text(__import__('json').dumps(rows))
    checks=read_json(out/'checksums.json');checks['B1_50/predictions.json']=sha(out/'B1_50/predictions.json')
    (out/'checksums.json').write_text(__import__('json').dumps(checks))
    with pytest.raises(DesignError,match='label/identity'):
        module.verify_smoke(design,digest,out)
