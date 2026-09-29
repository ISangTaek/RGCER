from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch

from dataset import DataCollator
from dataset115_adapter import GraphTaskView
from dataset115_model import build_human5_model
from reproducibility import state_dict_sha256
from tests.test_dataset115_adapter import args, view
from v9_srgt import DualGraph, TrainSupport, METHODS, SPEC, optimizer_for, training_records, save_smoke, load_smoke


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture():
    v=view(); ds=GraphTaskView(v,v.tasks[0]); batch=DataCollator()([ds[0],ds[1]])
    records=[dict(task=t,sample_id=s,canonical=c,group=g,split='train') for t in v.tasks
             for s,c,g in zip(v.sample_ids,v.canonical,v.groups)]
    support=TrainSupport(records)
    batch.v9_support=support.for_train_batch(v.tasks[0],batch.sample_id,batch.canonical_smiles)
    base=build_human5_model(args(),method='B0',seed=42)
    return base,batch,{t:2 for t in v.tasks},support


@pytest.mark.parametrize('method',METHODS)
def test_initial_function_equals_frozen_source_and_rng_isolated(method):
    base,batch,counts,_=fixture(); base.eval(); task=base.task_name[0]
    rng=torch.get_rng_state().clone()
    model=DualGraph(base,method,counts)
    assert torch.equal(rng,torch.get_rng_state())
    model.eval()
    torch.testing.assert_close(model(batch,task_name=task)[task],base(batch,task_name=task)[task],rtol=0,atol=0)
    model.train()
    assert not model.encoder.training and model.target.training and model.decoders.training
    a=model.encoder(batch); b=model.encoder(batch)
    assert torch.equal(a,b)
    assert not a.requires_grad
    assert model.target is not model.encoder.backbone


def train_two(method):
    from loss import QuantileRegressionLoss
    base,batch,counts,support=fixture(); task=base.task_name[0]
    model=DualGraph(base,method,counts); opt=optimizer_for(model)
    initial_source=state_dict_sha256(model.encoder); initial_target=state_dict_sha256(model.target)
    branch_gradients=[]
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        raw=model(batch,task_name=task)[task]
        loss=QuantileRegressionLoss().compute_loss(raw,batch.y)+model.regularization()
        loss.backward()
        assert all(p.grad is None for p in model.encoder.parameters())
        branch_gradients.append(sum(float(p.grad.abs().sum()) for p in model.target.parameters() if p.grad is not None))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step()
    assert branch_gradients[0]==0 and branch_gradients[1]>0
    assert state_dict_sha256(model.encoder)==initial_source
    assert state_dict_sha256(model.target)!=initial_target
    identity=dict(spec=SPEC,method=method,support=support.identity,task_updates={t:2 if t==task else 0 for t in counts})
    return base,batch,counts,model,opt,identity


@pytest.mark.parametrize('method',METHODS)
def test_raw_graph_branch_learns_without_source_drift_and_checkpoint_restores(method,tmp_path):
    base,batch,counts,model,opt,identity=train_two(method)
    path=tmp_path/'state.pt'; save_smoke(path,model,opt,identity,2)
    fresh=DualGraph(base,method,counts); newopt=optimizer_for(fresh)
    load_smoke(path,fresh,newopt,identity,2)
    assert state_dict_sha256(model)==state_dict_sha256(fresh)
    assert [float(v['step']) for v in opt.state.values()]==[float(v['step']) for v in newopt.state.values()]
    model.eval();fresh.eval();task=model.task_name[0]
    assert torch.equal(model(batch,task_name=task)[task],fresh(batch,task_name=task)[task])
    # Exact continuation with the same RNG demonstrates Adam restoration, not
    # merely equality of saved predictions. This is synthetic CPU work only.
    rng=torch.get_rng_state()
    for m,o in ((model,opt),(fresh,newopt)):
        torch.set_rng_state(rng);m.train();o.zero_grad(set_to_none=True)
        m(batch,task_name=task)[task].square().mean().backward();o.step()
    assert state_dict_sha256(model)==state_dict_sha256(fresh)


@pytest.mark.parametrize('change',['identity','source','missing_adam','adam_step','nan','task_order','unknown'])
def test_checkpoint_fail_closed_before_weight_mutation(change,tmp_path):
    base,batch,counts,model,opt,identity=train_two('SRGT')
    path=tmp_path/'state.pt';save_smoke(path,model,opt,identity,2)
    payload=torch.load(path,weights_only=True)
    if change=='identity':payload['identity']['method']='REFINE_GRAPH'
    if change=='source':payload['model_state'][next(k for k in payload['model_state'] if k.startswith('encoder.'))].add_(1)
    if change=='missing_adam':payload['optimizer_state']['state'].clear()
    if change=='adam_step':next(iter(payload['optimizer_state']['state'].values()))['step'].fill_(99)
    if change=='nan':payload['model_state']['fusion.weight'][0,0]=float('nan')
    if change=='task_order':payload['identity']['task_updates'].pop(next(iter(counts)))
    if change=='unknown':payload['extra']=True
    torch.save(payload,path)
    fresh=DualGraph(base,'SRGT',counts);before=state_dict_sha256(fresh)
    with pytest.raises(ValueError):load_smoke(path,fresh,optimizer_for(fresh),identity,2)
    assert state_dict_sha256(fresh)==before


def rec(sid,c,g,split='train',task='t'):
    return dict(task=task,sample_id=sid,canonical=c,group=g,split=split)


def test_support_excludes_canonical_and_entire_group():
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    rows=[rec('1','CC','g1'),rec('2','CCC','g1'),rec('3','c1ccccc1','g2')]
    bank=TrainSupport(rows)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    expected=DataStructs.TanimotoSimilarity(generator.GetFingerprint(Chem.MolFromSmiles('CC')),
                                           generator.GetFingerprint(Chem.MolFromSmiles('c1ccccc1')))
    assert bank.values['CC']==(expected,1.)
    assert bank.values['CC'][0] < 1
    assert TrainSupport([rec('1','CC','same'),rec('2','CCC','same')]).values['CC']==(0.,0.)
    assert bank.identity==TrainSupport(list(reversed(rows))).identity
    with pytest.raises(ValueError):bank.for_train_batch('t',['missing'],['CC'])
    with pytest.raises(ValueError):bank.for_train_batch('t',['1'],['CCC'])


@pytest.mark.parametrize('change',['test','validation','calibration','labels','duplicate','same_id','wrong_group'])
def test_support_rejects_holdouts_or_corrupt_members(change):
    rows=[rec('1','CC','g1'),rec('2','CCC','g2')]
    if change in ('test','validation','calibration'):rows[0]['split']=change
    if change=='labels':rows[0]['label']=123
    if change=='duplicate':rows.append(deepcopy(rows[0]))
    if change=='same_id':rows[1]['sample_id']='1'
    if change=='wrong_group':rows.append(rec('3','CC','g2'))
    with pytest.raises(ValueError):TrainSupport(rows)


def test_gate_is_noncompetitive_and_rare_task_shrinkage_is_stronger():
    base,batch,counts,_=fixture();counts[base.task_name[0]]=1
    model=DualGraph(base,'SRGT',counts);task=model.task_name[0]
    reference=torch.ones(2,model.hidden)
    assert torch.equal(model.gates(task,batch.v9_support,reference),torch.ones(2,2))
    with torch.no_grad():model.global_logits.fill_(2);model.task_logits[0].fill_(1)
    gates=model.gates(task,batch.v9_support,reference)
    assert torch.all(gates>1) and torch.all(gates<2)  # Both branches can increase.
    assert model.shrink_weights[0] > model.shrink_weights[1]
    assert model.regularization()>0
    bad=batch.v9_support.clone();bad[0,1]=.5
    with pytest.raises(ValueError):model.gates(task,bad,reference)


def test_target_can_change_prediction_while_source_features_are_identical():
    base,batch,counts,_=fixture();model=DualGraph(base,'REFINE_GRAPH',counts).eval(); task=model.task_name[0]
    with torch.no_grad():
        model.fusion.weight[:,model.hidden:].fill_(.05)
        _,a=model(batch,task_name=task,return_aux=True)
        model.target.final_norm.bias[0].add_(1)
        _,b=model(batch,task_name=task,return_aux=True)
    assert torch.equal(a['source_representation'],b['source_representation'])
    assert not torch.equal(a['final_representation'],b['final_representation'])


def test_training_metadata_joins_tox_rows_and_checks_split(tmp_path):
    path=tmp_path/'split_manifest.json'
    rows=[dict(row_index=17,split='train',sample_id='s17',canonical_smiles='CC',split_group='g1',num_nodes=2)]
    path.write_text(json.dumps(dict(records=rows)),encoding='utf8')
    ds=SimpleNamespace(indices=[0],split='train',get_sample_id=lambda i:'s17')
    factory=SimpleNamespace(store=SimpleNamespace(root=tmp_path,row_indices=[0],sample_ids=['s17'],
                            split_codes=[0],num_nodes=[2]),datasets={'train':{'t':ds}})
    assert training_records(factory,None,'ToxAcute')==[rec('s17','CC','g1')]
    rows[0]['split']='test';path.write_text(json.dumps(dict(records=rows)),encoding='utf8')
    with pytest.raises(ValueError):training_records(factory,None,'ToxAcute')
