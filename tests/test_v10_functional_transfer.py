from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

import v10_functional_transfer as m


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def episode(task='target', c=9, q=4, d=5, functions=3):
    g = torch.Generator().manual_seed(91)
    def rand(*shape): return torch.randn(*shape, generator=g, dtype=torch.float64)
    def rows(prefix, n):
        return tuple(m.RowIdentity(f'{prefix}{i}',f'mol-{prefix}{i}',f'group-{prefix}{i}',
                                   'train',task) for i in range(n))
    return m.Episode(task, rows('c',c), rows('q',q), rand(c,d), rand(c,functions), rand(c,1),
                     rand(q,d), rand(q,functions), rand(1,2))


@pytest.mark.parametrize('n,d', [(20,4),(5,17),(6,6)])
def test_ridge_matches_independent_numpy_objective(n,d):
    rng = np.random.default_rng(11); x=rng.normal(size=(n,d)); y=rng.normal(size=(n,2))
    expected=np.linalg.solve(x.T@x+n*.31*np.eye(d), x.T@y)
    actual=m.ridge_coefficients(torch.tensor(x),torch.tensor(y),torch.tensor(.31,dtype=torch.float64))
    np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-12,atol=1e-12)


def test_functional_ridge_matches_independent_two_stage_solution():
    e=episode(); x,f,y=e.context_features.numpy(),e.context_functions.numpy(),e.context_labels.numpy()
    avg=f.mean(1,keepdims=True); meta=f-avg; xm=x.mean(0); mm=meta.mean(0); b=(y-avg).mean(0)
    c=np.linalg.solve((meta-mm).T@(meta-mm)+len(x)*.2*np.eye(f.shape[1]),(meta-mm).T@(y-avg-b))
    residual=y-avg-b-(meta-mm)@c
    beta=np.linalg.solve((x-xm).T@(x-xm)+len(x)*.7*np.eye(x.shape[1]),(x-xm).T@residual)
    q,ff=e.query_features.numpy(),e.query_functions.numpy(); qa=ff.mean(1,keepdims=True)
    expected=qa+b+(ff-qa-mm)@c+(q-xm)@beta
    fit=m.fit_functional_ridge(e.context_features,e.context_functions,e.context_labels,
                               source_penalty=.2,residual_penalty=.7)
    np.testing.assert_allclose(fit.predict(e.query_features,e.query_functions).numpy(),expected,atol=1e-12)
    # Fitting snapshots its inputs; mutation after fitting cannot rewrite it.
    old=fit.predict(e.query_features,e.query_functions).clone()
    e.context_features.zero_(); e.context_functions.zero_(); e.context_labels.zero_()
    torch.testing.assert_close(fit.predict(e.query_features,e.query_functions),old,atol=0,rtol=0)


def model():
    torch.manual_seed(3)
    return m.FunctionalContextRidge(5,('s0','s1','s2'),2,hidden=8).double()


def test_ridge_autograd_matches_finite_differences():
    g=torch.Generator().manual_seed(51)
    x=torch.randn(4,3,generator=g,dtype=torch.float64,requires_grad=True)
    y=torch.randn(4,1,generator=g,dtype=torch.float64,requires_grad=True)
    penalty=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(m.ridge_coefficients,(x,y,penalty),eps=1e-6,atol=1e-5)


def test_episode_context_permutation_invariance_and_query_chunk_invariance():
    net=model(); e=episode(); expected=net(e)
    order=torch.tensor([4,2,6,7,1,3,0,8,5])
    reordered=replace(e,context_rows=tuple(e.context_rows[i] for i in order),
        context_features=e.context_features[order],context_functions=e.context_functions[order],
        context_labels=e.context_labels[order])
    torch.testing.assert_close(net(reordered),expected,rtol=1e-10,atol=1e-10)
    pieces=[]
    for i in range(len(e.query_rows)):
        one=replace(e,query_rows=(e.query_rows[i],),query_features=e.query_features[i:i+1],
                    query_functions=e.query_functions[i:i+1])
        pieces.append(net(one))
    torch.testing.assert_close(torch.cat(pieces),expected,rtol=1e-10,atol=1e-10)


def test_episode_label_location_scale_equivariance_and_constant_labels():
    net=model(); e=episode(); p=net(e)
    transformed=net(replace(e,context_labels=7+3*e.context_labels))
    torch.testing.assert_close(transformed,7+3*p,atol=1e-10,rtol=1e-10)
    constant=net(replace(e,context_labels=torch.full_like(e.context_labels,2.5)))
    torch.testing.assert_close(constant,torch.full_like(constant,2.5),atol=0,rtol=0)


def test_source_episode_cannot_copy_own_source_head():
    net=model(); e=episode('s1'); p=net(e)
    cf,qf=e.context_functions.clone(),e.query_functions.clone()
    cf[:,1]=1e9; qf[:,1]=-1e9
    torch.testing.assert_close(net(replace(e,context_functions=cf,query_functions=qf)),p,atol=0,rtol=0)
    cf[:,0]+=20
    assert not torch.allclose(net(replace(e,context_functions=cf,query_functions=qf)),p)


def test_context_labels_influence_predictions_but_query_labels_are_not_an_input():
    e=episode(); net=model(); p=net(e)
    shuffled=replace(e,context_labels=e.context_labels.flip(0))
    assert not torch.allclose(net(shuffled),p)
    with pytest.raises(TypeError):
        replace(e,query_labels=torch.ones(4,1))


def test_shared_model_gets_gradient_from_every_task():
    net=model()
    for task in ('s0','s1','s2','target_a','target_b'):
        e=episode(task); net.zero_grad(set_to_none=True)
        net(e).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        assert net.features[0].weight.grad.abs().sum()>0


@pytest.mark.parametrize('field', ('sample_id','canonical','group'))
def test_context_query_identity_overlap_is_rejected(field):
    e=episode(); q=replace(e.query_rows[0],**{field:getattr(e.context_rows[0],field)})
    with pytest.raises(ValueError,match='overlap'):
        model()(replace(e,query_rows=(q,*e.query_rows[1:])))


@pytest.mark.parametrize('fault',('context_validation','query_test','wrong_task','duplicate','nan','dtype','shape'))
def test_bad_context_inputs_are_rejected(fault):
    e=episode()
    if fault=='context_validation': e=replace(e,context_rows=(replace(e.context_rows[0],split='validation'),*e.context_rows[1:]))
    if fault=='query_test': e=replace(e,query_rows=(replace(e.query_rows[0],split='test'),*e.query_rows[1:]))
    if fault=='wrong_task': e=replace(e,context_rows=(replace(e.context_rows[0],task='other'),*e.context_rows[1:]))
    if fault=='duplicate': e=replace(e,context_rows=(e.context_rows[1],*e.context_rows[1:]))
    if fault=='nan': e.context_labels[0]=float('nan')
    if fault=='dtype': e=replace(e,query_features=e.query_features.float())
    if fault=='shape': e=replace(e,metadata=torch.ones(2,2,dtype=torch.float64))
    with pytest.raises(ValueError): model()(e)


@pytest.mark.parametrize('value',(0.,-1.,float('inf'),float('nan')))
def test_invalid_penalties_fail(value):
    e=episode()
    with pytest.raises(ValueError,match='penalty'):
        m.fit_functional_ridge(e.context_features,e.context_functions,e.context_labels,source_penalty=value)


def test_frozen_bank_is_detached_and_preserves_original_model():
    torch.manual_seed(12)
    encoder=nn.Sequential(nn.Linear(4,5),nn.Dropout(.5))
    heads=nn.ModuleDict({'s0':nn.Linear(5,3),'s1':nn.Linear(5,3)})
    bank=m.FrozenFunctionBank(encoder,heads); bank.train()
    x=torch.randn(7,4,requires_grad=True); h,f=bank(x)
    assert encoder.training and all(p.requires_grad for p in encoder.parameters())
    assert not h.requires_grad and not f.requires_grad
    assert all(not module.training for module in bank.modules())
    assert bank.tasks==('s0','s1')
    torch.testing.assert_close(f[:,0],bank.heads['s0'](h)[:,0])
    h2,f2=bank(x); torch.testing.assert_close(h,h2,atol=0,rtol=0); torch.testing.assert_close(f,f2,atol=0,rtol=0)
    bank.encoder[0].weight.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'): bank(x)
