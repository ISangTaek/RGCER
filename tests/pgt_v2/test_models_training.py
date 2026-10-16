import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from architecture.toxacute_tasks import TOXACUTE_TASKS
from pgt_v2.chemistry import prepare_molecule, graph_from_input
from pgt_v2.config import candidate
from pgt_v2.data import PGTCollator
from pgt_v2.models import AtomReadout, build_model, molecular_features, task_metadata
from pgt_v2.explain import additive_reference, additive_accounting, endpoint_difference, integrated_gradients
from pgt_v2.training import endpoint_loss, partition_batches, seed_everything, train_model, read_checkpoint

TASKS = tuple(TOXACUTE_TASKS[:2])


def tiny_config(method='M6'):
    return {**candidate(method,'C0'), 'hidden_dim':16, 'ffn_dim':24, 'heads':4, 'layers':1,
        'dropout':0., 'max_epochs':4, 'min_epochs':3, 'patience':2, 'effective_batch':4, 'micro_batch':2}


def graph_batch():
    graphs = [graph_from_input(prepare_molecule(smiles), sample_id=f'r{i}', raw_row_index=i)
              for i,smiles in enumerate(('CO','CC(O)C','C'))]
    return PGTCollator()(graphs)


@pytest.mark.parametrize('method',['M2','M4','M5','M6','A1','A2','A3'])
def test_graph_models_forward_backward_and_padding(method):
    seed_everything(42)
    model = build_model(method,TASKS,tiny_config(method))
    batch = graph_batch()
    prediction, aux = model(batch,return_aux=True)
    assert prediction.shape == (3,2) and torch.isfinite(prediction).all()
    endpoint_loss(prediction,torch.tensor([[1.,-1.],[float('nan'),2.],[3.,4.]]),[2,3],3).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if 'attention' in aux:
        pad = batch.padding_mask[:,None,None,:].expand_as(aux['attention'])
        assert torch.all(aux['attention'][pad] == 0)
        torch.testing.assert_close(aux['attention'].sum(-1),torch.ones_like(aux['attention'].sum(-1)))


def test_shared_query_and_task_query_counterfactual():
    batch = graph_batch()
    shared = build_model('M5',TASKS,tiny_config()).eval()
    _,aux = shared(batch,return_aux=True)
    assert torch.equal(aux['attention'][:,0],aux['attention'][:,1])
    task = build_model('M6',TASKS,tiny_config()).eval()
    pred,aux = task(batch,return_aux=True)
    modified = task.readout.query.detach().clone(); modified[0] += torch.arange(16)*.5
    other,b = task(batch,return_aux=True,query_override=modified)
    assert not torch.allclose(pred[:,0],other[:,0],atol=1e-7)
    assert torch.equal(pred[:,1],other[:,1])
    assert not torch.equal(aux['attention'][:,0],b['attention'][:,0])


@pytest.mark.parametrize('method',['A2','A3'])
def test_additive_centering_scale_and_endpoint_difference(method):
    model = build_model(method,TASKS,tiny_config()).eval()
    batch = graph_batch()
    pred,aux = model(batch,return_aux=True)
    reference = additive_reference(model,[batch],record_ids=['r0','r1','r2'],view_id='view',checkpoint_sha256='a'*64)
    raw = additive_accounting(pred,aux,mean=[10.,-4.],scale=[2.,3.])
    centered = additive_accounting(pred,aux,mean=[10.,-4.],scale=[2.,3.],reference=reference)
    torch.testing.assert_close(raw['prediction'],centered['prediction'])
    torch.testing.assert_close(centered['contributions'].sum(-1).mean(0),torch.zeros(2),atol=2e-6,rtol=1e-5)
    assert centered['max_reconstruction_error'] < 2e-6
    assert endpoint_difference(centered,0,1)['max_reconstruction_error'] < 4e-6


def test_equal_endpoint_microbatch_gradient_and_missing():
    seed_everything(45)
    x = torch.randn(7,3,dtype=torch.float64)
    y = torch.tensor([[1.,float('nan')],[2.,-1.],[3.,float('nan')],[4.,2.],[float('nan'),3.],[5.,4.],[6.,float('nan')]],dtype=torch.float64)
    count=torch.isfinite(y).sum(0)
    model=torch.nn.Linear(3,2).double()
    pred=model(x);pred.retain_grad()
    loss=endpoint_loss(pred,y,count,7);loss.backward()
    assert pred.grad[~torch.isfinite(y)].eq(0).all()
    full=[p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for ids in ([0,1,2],[3],[4,5,6]):
        (endpoint_loss(model(x[ids]),y[ids],count,7)*len(ids)/7).backward()
    for left,right in zip(full,model.parameters()):
        torch.testing.assert_close(left,right.grad,atol=1e-12,rtol=1e-12)


def test_bn_tail_merge_retains_every_record():
    assert partition_batches(list(range(9)),4,batch_norm=True)==[[0,1,2,3],[4,5,6,7,8]]
    assert partition_batches(list(range(5)),2,batch_norm=True)==[[0,1],[2,3,4]]
    with pytest.raises(ValueError): partition_batches([0],2,batch_norm=True)
    with pytest.raises(ValueError): partition_batches([0,1],1,batch_norm=True)


def test_random_prompts_full_rank_can_represent_arbitrary_queries(record_property):
    g=torch.Generator().manual_seed(42)
    p=torch.randn(59,96,generator=g,dtype=torch.float64)
    q=torch.randn(59,96,generator=g,dtype=torch.float64)
    assert torch.linalg.matrix_rank(p)==59
    weights=torch.linalg.lstsq(p,q).solution
    error=float((p@weights-q).abs().max());record_property('linear_query_max_error',error)
    assert error < 1e-12


@pytest.mark.parametrize('additive',[False,True])
def test_opposite_task_response_on_same_atom_synthetic(additive,record_property):
    # A one-atom scalar has no spatial selection ambiguity. The task head/value
    # must support opposite signs, not merely different positive attentions.
    seed_everything(71)
    atoms=torch.randn(32,1,8)
    target=torch.stack([atoms[:,0,0],-atoms[:,0,0]],dim=1)
    readout=AtomReadout(TASKS,8,2,additive=additive)
    heads=torch.nn.ModuleList([torch.nn.Linear(8,1) for _ in TASKS])
    parameters=list(readout.parameters()) + ([] if additive else list(heads.parameters()))
    optimizer=torch.optim.Adam(parameters,lr=.025)
    mask=torch.ones(32,1,dtype=torch.bool)
    for _ in range(160):
        optimizer.zero_grad();value,_=readout(atoms,mask)
        prediction=value if additive else torch.cat([h(value[:,i]) for i,h in enumerate(heads)],dim=1)
        loss=(prediction-target).square().mean();loss.backward();optimizer.step()
    record_property('synthetic_optimizer_steps',160)
    record_property('synthetic_mse',float(loss.detach()))
    assert float(loss.detach()) < .025
    for t,sign in ((0,1),(1,-1)):
        corr=np.corrcoef(prediction[:,t].detach().numpy(),atoms[:,0,0].numpy())[0,1]
        assert corr*sign > .98


def test_m3_59_channels_and_metadata_semantics():
    model=build_model('M3',TOXACUTE_TASKS,candidate('M3','C0'),adjacency=np.eye(59)).eval()
    pred=model(torch.randn(2,1024))
    assert pred.shape==(2,59) and torch.isfinite(pred).all()
    assert model.endpoint_features.shape==(59,26)
    meta=task_metadata(('man_oral_TDLo','women_oral_TDLo','human_oral_TDLo'))
    assert {v['organism'] for v in meta.values()}=={'human'}
    assert len({v['population'] for v in meta.values()})==3
    with pytest.raises(TypeError): molecular_features(['CC'],'morgan')


def test_ig_full_model_pair_and_padding(record_property):
    seed_everything(42)
    model=build_model('M6',TASKS,tiny_config()).double()
    batch=graph_batch()
    result=integrated_gradients(model,batch,task=0,other_task=1,mean=[1.,2.],scale=[2.,3.])
    assert torch.isfinite(result['attributions']).all()
    assert result['attributions'][batch.padding_mask].eq(0).all()
    residual=result['attributions'].sum(-1)-(result['prediction']-result['baseline_prediction'])
    torch.testing.assert_close(residual,result['completeness_residual'])
    assert torch.equal(result['unreliable'],result['normalized_residual'] > 1e-3)
    # A zero embedding passes near LayerNorm's singular region. A finite
    # 64-point integral may fail completeness; flag, never hide/retry it.
    record_property('ig_max_absolute_residual',float(residual.abs().max()))
    record_property('ig_unreliable_count',int(result['unreliable'].sum()))


class TinyInputs:
    def __init__(self,x):
        self.x=x
        self.records={'group_id':np.array([f'g{i}' for i in range(len(x))])}
    def __len__(self): return len(self.x)
    def batch(self,ids): return self.x[ids]


@pytest.mark.parametrize('method',['M6','M3'])
def test_exact_epoch_resume_with_dropout_and_optimizer(tmp_path,method,record_property):
    config={**tiny_config(),'micro_batch':3,'effective_batch':4,'lr':.01,'max_epochs':4,'patience':30}
    x=torch.randn(9,3,generator=torch.Generator().manual_seed(9))
    labels=np.column_stack([x[:,0].numpy(),-x[:,1].numpy()]).astype(np.float64)
    labels[0,1]=np.nan
    stats=dict(mean=np.zeros(2),scale=np.ones(2),count=np.isfinite(labels).sum(0))
    identity=dict(task_names=list(TASKS),run_id='synthetic-resume',config=config)
    def init():
        seed_everything(123)
        return torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.BatchNorm1d(8) if method=='M3' else torch.nn.Identity(),
                                   torch.nn.Dropout(.2),torch.nn.GELU(),torch.nn.Linear(8,2))
    data=TinyInputs(x)
    full=init();train_model(full,data,labels,data,labels,stats,config=config,identity=identity,output=tmp_path/'full',method=method)
    part=init();train_model(part,data,labels,data,labels,stats,config=config,identity=identity,output=tmp_path/'part',method=method,stop_after_epoch=2)
    # Simulate a fresh process consuming arbitrary randomness before restore.
    random_model=init();torch.rand(13)
    train_model(random_model,data,labels,data,labels,stats,config=config,identity=identity,output=tmp_path/'part',method=method,resume=True)
    a,b=read_checkpoint(tmp_path/'full/last.pt'),read_checkpoint(tmp_path/'part/last.pt')
    for key in a['model']: assert torch.equal(a['model'][key],b['model'][key])
    expected=8 if method=='M3' else 12
    assert a['updates']==b['updates']==expected
    record_property('synthetic_optimizer_steps',expected*2)
    assert a['best']==b['best'] and a['best_epoch']==b['best_epoch']
    for h1,h2 in zip(a['history'],b['history']):
        assert h1['train_loss']==h2['train_loss'] and h1['validation']==h2['validation']
    with pytest.raises(ValueError,match='identity'):
        train_model(random_model,data,labels,data,labels,stats,config=config,identity={**identity,'task_names':list(reversed(TASKS))},output=tmp_path/'part',method=method,resume=True)


def test_ig_linear_control_exact_original_scale_and_pair():
    class LinearFullModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=build_model('M2',TASKS,tiny_config()).backbone.double()
            self.weights=torch.nn.Parameter(torch.randn(2,16,dtype=torch.float64))
        def forward(self,batch,atom_embeddings=None):
            values=atom_embeddings.masked_fill(batch.padding_mask[...,None],0).sum(1)
            return values@self.weights.T
    model=LinearFullModel();batch=graph_batch()
    result=integrated_gradients(model,batch,task=0,other_task=1,mean=[10.,30.],scale=[2.,4.])
    assert not result['unreliable'].any()
    assert float(result['completeness_residual'].abs().max())<1e-10
