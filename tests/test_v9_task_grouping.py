from copy import deepcopy
import numpy as np
import pytest
import torch

import v9_task_grouping as m
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime,one_cpu_thread,setup_factory


def row(t,s,g,split='train'):
    return dict(task=t,sample_id=s,canonical=s,group=g,split=split,label=1.)


def test_probes_are_train_only_one_per_scaffold_and_order_invariant():
    rows=[row(t,t+str(i),'scaffold'+str(i//2)) for t in ('a','b') for i in range(24)]
    x=m.probe_selection(rows,['a','b']);y=m.probe_selection(list(reversed(rows)),['a','b'])
    assert x==y and all(len(v)==8 and len({r['group'] for r in v})==8 for v in x.values())
    with pytest.raises(ValueError):m.probe_selection(rows+[row('a','new','heldout','validation')],['a','b'])
    with pytest.raises(ValueError):m.probe_selection(rows+[rows[0]],['a','b'])


def test_paper_coordinate_normalization_not_ordinary_cosine():
    x=np.array([[1.,10.,0.]],dtype=np.float32);y=np.array([[10.,1.,0.]],dtype=np.float32)
    assert m.pair_cosines(x,x)[0,0]==pytest.approx(1)
    assert m.pair_cosines(x,-x)[0,0]==pytest.approx(-1)
    assert m.pair_cosines(x,y)[0,0]==pytest.approx(20/101)
    assert m.pair_cosines(x,np.zeros_like(x))[0,0]==0
    # This example distinguishes pair-coordinate normalization from raw cosine.
    y=np.array([[2.,1.,0.]],dtype=np.float32)
    expected=np.dot([1/3,10/11],[2/3,1/11])/(np.linalg.norm([1/3,10/11])*np.linalg.norm([2/3,1/11]))
    assert m.pair_cosines(x,y)[0,0]==pytest.approx(expected)
    assert expected != pytest.approx(float(((x@y.T)/(np.linalg.norm(x)*np.linalg.norm(y))).item()))


def test_reliable_opposition_splits_and_low_sample_conflict_keeps_sharing():
    tasks=['a','b','c','d']
    def evidence(n):
        sel={t:[row(t,t+str(i),'group'+str(i)) for i in range(n)] for t in tasks}
        vectors={t:torch.ones(n,4)*(1 if t in ('a','b') else -1) for t in tasks}
        return m.similarities(vectors,sel,tasks)
    reliable=evidence(8);groups,decision=m.choose_groups('SGTA',reliable)
    assert groups==[['a','b'],['c','d']] and decision['split']
    sparse=evidence(2);assert m.choose_groups('SGTA',sparse)[0]==[tasks]
    assert len(m.choose_groups('TG_LORA_GRAPH',sparse)[0])==2
    assert m.choose_groups('SHARED_LORA',reliable)[0]==[tasks]


def test_similarity_deterministic_finite_and_conservative():
    tasks=['a','b'];sel={t:[row(t,t+str(i),str(i)) for i in range(4)] for t in tasks}
    rng=torch.Generator().manual_seed(7);vec={t:torch.randn(4,5,generator=rng) for t in tasks}
    a=m.similarities(vec,sel,tasks);assert a==m.similarities(vec,sel,tasks)
    assert a['conservative'][0][1]>=a['mean'][0][1] and np.array(a['mean']).T.tolist()==a['mean']
    vec['a'][0,0]=float('nan')
    with pytest.raises(ValueError):m.similarities(vec,sel,tasks)


def test_greedy_negative_scores_terminate_and_score_is_eq2():
    matrix=-np.ones((5,5));np.fill_diagonal(matrix,1)
    assert len(m.greedy_groups(matrix))==2
    assert m.partition_score(matrix,[list(range(5))])==pytest.approx(-5)
    matrix[0,1]=matrix[1,0]=1;matrix[2,3]=matrix[3,2]=1
    assert m.greedy_groups(matrix)==m.greedy_groups(matrix)


@pytest.mark.parametrize('sizes',[(1,108),(50,59),(1,1),(100,1),(59,)])
def test_total_rank_exactly_eight_and_no_empty_group(sizes):
    result=m.ranks_for([list(range(n)) for n in sizes]);assert sum(result)==8 and min(result)>=1


@pytest.mark.parametrize('method',m.METHODS)
def test_zero_init_capacity_and_frozen_anchor(method,joint):
    import v9_s4_screen as s4
    f,t,a,_=joint('A');model=m.GroupedGraph(t.model,a,method);before=s4.parameter_counts(model)
    batch=s4.s3.s2.batch_for(f,t,'A','train',t.model.task_name[0],[0,1],'cpu');task=t.model.task_name[0]
    t.model.eval();model.eval()
    expected=t.model(batch,task_name=task)[task]
    assert torch.equal(model(batch,task_name=task)[task],expected)
    model.configure([list(t.model.task_name),list(a.tasks)])
    assert s4.parameter_counts(model)==before and torch.equal(model(batch,task_name=task)[task],expected)
    assert not model.encoder.training and not any(p.requires_grad for p in model.encoder.parameters())
    model.train();assert not model.encoder.training


def test_inactive_deep_group_has_no_gradient_but_shallow_is_shared(joint):
    f,t,a,_=joint('A');model=m.GroupedGraph(t.model,a,'SGTA');model.configure([list(t.model.task_name),list(a.tasks)])
    import v9_s4_screen as s4
    task=t.model.task_name[0];batch=s4.s3.s2.batch_for(f,t,'A','train',task,[0,1],'cpu')
    model(batch,task_name=task)[task].sum().backward()
    for name,block in model.target.items():
        if int(name.split('__')[1])>=len(model.encoder.backbone.layers)//2:
            assert block.B[0].grad is not None and block.B[1].grad is None
        else:assert block.B[0].grad is not None


def test_repartition_retains_head_adam_and_rejects_active_adapters(joint):
    _,t,a,_=joint('A');model=m.GroupedGraph(t.model,a,'SGTA');opt=m.optimizer_for(model)
    head=next(model.decoders.parameters());head.grad=torch.ones_like(head);opt.step()
    state=deepcopy(opt.state[head]);m.repartition(model,opt,[list(t.model.task_name),list(a.tasks)])
    assert all(torch.equal(v,opt.state[head][k]) for k,v in state.items())
    p=next(model.target.parameters());p.grad=torch.ones_like(p);opt.step()
    with pytest.raises(ValueError):m.repartition(model,opt,[model.task_name])


def test_repartition_does_not_reseed_cuda_or_change_cpu_rng(joint,monkeypatch):
    _,t,a,_=joint('A');model=m.GroupedGraph(t.model,a,'SGTA');before=torch.get_rng_state().clone()
    def forbidden(*args,**kwargs):raise AssertionError('reseeded CUDA')
    monkeypatch.setattr(torch.cuda,'manual_seed_all',forbidden)
    model.configure([list(t.model.task_name),list(a.tasks)])
    assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('damage',['missing','duplicate','empty'])
def test_model_rejects_invalid_task_partitions(damage,joint):
    _,t,a,_=joint('A');model=m.GroupedGraph(t.model,a,'SGTA');tasks=model.task_name
    groups=[tasks[:-1]] if damage=='missing' else ([tasks,[tasks[0]]] if damage=='duplicate' else [tasks,[]])
    with pytest.raises(ValueError):model.configure(groups)
