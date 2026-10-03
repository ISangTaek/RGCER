from dataclasses import replace

import numpy as np
import pytest
import torch

import v10_dual_context as m
from tests.test_v10_context_models import episode, single_thread


def make(name):
    return m.build(name,6,['rat_oral_LD50','mouse_oral_LD50'],3,hidden=8)


@pytest.mark.parametrize('name',m.NEW+m.CONTROLS)
def test_trainable_masked_and_permutation_chunk_invariant(name,episode):
    model=make(name); labels=torch.tensor([[.3],[-.5],[1.2]])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    before={k:v.clone() for k,v in model.state_dict().items()}
    loss=model.loss(episode,labels); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    model.eval(); y=model(episode)
    pc=episode.context_functions.clone(); pq=episode.query_functions.clone(); pc[:,0]+=1000; pq[:,0]-=1000
    torch.testing.assert_close(model(replace(episode,context_functions=pc,query_functions=pq)),y)
    order=torch.tensor([6,0,3,1,5,2,4])
    perm=replace(episode,context_rows=[episode.context_rows[i] for i in order],
        context_features=episode.context_features[order],context_functions=episode.context_functions[order],
        context_labels=episode.context_labels[order])
    torch.testing.assert_close(model(perm),y,atol=2e-5,rtol=2e-5)
    parts=[model(replace(episode,query_rows=episode.query_rows[i:i+1],query_features=episode.query_features[i:i+1],
                         query_functions=episode.query_functions[i:i+1])) for i in range(3)]
    torch.testing.assert_close(torch.cat(parts),y,atol=2e-5,rtol=2e-5)
    reverse=replace(episode,query_rows=episode.query_rows[::-1],query_features=episode.query_features.flip(0),query_functions=episode.query_functions.flip(0))
    torch.testing.assert_close(model(reverse).flip(0),y,atol=2e-5,rtol=2e-5)
    with pytest.raises(ValueError,match='train queries only'):
        model.loss(replace(episode,query_rows=[replace(r,split='validation') for r in episode.query_rows]),labels)


@pytest.mark.parametrize('name',m.NEW+('BSA_MSE',))
def test_context_labels_have_an_effect(name,episode):
    model=make(name).eval()
    assert not torch.allclose(model(episode),model(replace(episode,context_labels=episode.context_labels+3)))


def test_capacity_control_has_no_label_context_path(episode):
    model=make('JOINT_MLP_CAP').eval()
    torch.testing.assert_close(model(episode),model(replace(episode,context_labels=episode.context_labels+3)))


@pytest.mark.parametrize('name',m.NEW)
def test_exact_distance_ties_with_more_than_top_k_are_identity_stable(name,episode):
    rows=[replace(episode.context_rows[0],sample_id=f'c{i}',canonical=f'c{i}',group=f'c{i}') for i in range(100)]
    e=replace(episode,context_rows=rows,context_features=torch.zeros(100,6),context_functions=torch.zeros(100,2),
              context_labels=torch.arange(100,dtype=torch.float32)[:,None]/100)
    model=make(name).eval(); before=model(e)
    e=replace(e,context_rows=e.context_rows[::-1],context_features=e.context_features.flip(0),
              context_functions=e.context_functions.flip(0),context_labels=e.context_labels.flip(0))
    torch.testing.assert_close(model(e),before,atol=1e-6,rtol=1e-6)


def test_retrieval_equation_with_mask_matches_numpy():
    torch.manual_seed(17); model=m.Retrieval(4)
    c,q,labels=torch.randn(5,4),torch.randn(3,4),torch.randn(5,1)
    legal=torch.tensor([[1,0,1,0,1],[0,0,0,0,0],[0,1,0,1,0]],dtype=torch.bool)
    with torch.no_grad():
        result,empty=model(c,q,labels,legal=legal)
        kc,kq=model.key(c),model.key(q)
        values=model.label(labels)[None]+model.correction(kq[:,None]-kc[None])
        logits=-((kq.numpy()[:,None]-kc.numpy()[None])**2).sum(-1)
        weights=np.zeros_like(logits)
        for i in range(3):
            mask=legal[i].numpy()
            if mask.any():
                z=np.exp(logits[i,mask]-logits[i,mask].max()); weights[i,mask]=z/z.sum()
        expected=(weights[...,None]*values.numpy()).sum(1)
    np.testing.assert_allclose(result.numpy(),expected,atol=1e-6)
    assert empty==1 and torch.equal(result[1],torch.zeros(4))


def test_retrieval_first_empty_group_is_zero_recorded_and_finite(episode):
    e=replace(episode,context_rows=[replace(r,group='one_context_group') for r in episode.context_rows])
    model=make('DCR_RETRIEVAL_FIRST')
    loss=model.loss(e,torch.ones(3,1)); loss.backward()
    assert m.diagnostics(model)['empty_context_retrieval_rows']==7
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_internal_mask_excludes_sample_canonical_and_group(episode):
    rows=episode.context_rows[:4]
    rows[1]=replace(rows[1],canonical=rows[0].canonical)
    rows[2]=replace(rows[2],group=rows[0].group)
    rows[3]=replace(rows[3],sample_id=rows[0].sample_id)
    mask=m.legal_context_pairs(rows,'cpu')
    assert not mask[0].any() and not mask.diag().any()
    assert mask[1,2]


def test_three_flows_differ_with_identical_weights(episode):
    model=make('DCR_PARALLEL'); baseline=model(episode)
    for name in ('DCR_CONTEXT_FIRST','DCR_RETRIEVAL_FIRST'):
        other=make(name); other.load_state_dict(model.state_dict())
        assert not torch.allclose(baseline,other(episode))
    loss=model.loss(episode,torch.ones(3,1)); loss.backward()
    for module in (model.project,model.retrieval,model.relation,model.decoder):
        assert any(p.grad is not None and bool((p.grad!=0).any()) for p in module.parameters())


def test_bsa_mse_has_identical_forward_and_state_schema(episode):
    old=m.base.BSATNP(6,['rat_oral_LD50','mouse_oral_LD50'],3,hidden=8)
    new=make('BSA_MSE'); new.load_state_dict(old.state_dict(),strict=True)
    torch.testing.assert_close(old(episode),new(episode))
    y=torch.ones(3,1)
    torch.testing.assert_close(new.loss(episode,y),(new(episode)-y).square().mean())
    assert not torch.isclose(old.loss(episode,y),new.loss(episode,y))


@pytest.mark.parametrize('functions,metadata',[(2,3),(56,31),(104,40)])
def test_capacity_is_formula_selected_without_consuming_extra_model_rng(functions,metadata):
    source=[str(i) for i in range(functions)]
    main=m.build('DCR_PARALLEL',128,source,metadata)
    control=m.build('JOINT_MLP_CAP',128,source,metadata)
    count=m.describe(main)['parameters']; matched=m.describe(control)
    assert count==m.parallel_parameter_count(main.input_dim,64)==matched['target_parameters']
    assert abs(matched['parameters']/count-1)<=.05
    w=matched['width']; formula=lambda k:k*k+(main.input_dim+3)*k+1
    assert abs(formula(w)-count)<=min(abs(formula(w-1)-count),abs(formula(w+1)-count))
    state=torch.get_rng_state().clone()
    m.capacity_width(main.input_dim,64)
    assert torch.equal(state,torch.get_rng_state())


def test_six_retained_plus_four_new_and_two_controls_are_distinct():
    assert len(m.CANDIDATES)==len(set(m.CANDIDATES))==10
    assert len(m.NEW)==4 and set(m.CONTROLS).isdisjoint(m.CANDIDATES)
    with pytest.raises(ValueError): make('TABR')
