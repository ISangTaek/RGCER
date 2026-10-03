from dataclasses import replace
import numpy as np
import pytest
import torch

import v10_context_models as m
from v10_functional_transfer import Episode, RowIdentity


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def episode():
    torch.manual_seed(42)
    def rows(n,offset):
        return [RowIdentity(str(i),str(i),str(i),'train','rat_oral_LD50') for i in range(offset,offset+n)]
    return Episode('rat_oral_LD50',rows(7,0),rows(3,10),torch.randn(7,6),torch.randn(7,2),
                   torch.randn(7,1),torch.randn(3,6),torch.randn(3,2),torch.tensor([[1.,0.,1.]]))


def make(name): return m.build(name,6,['rat_oral_LD50','mouse_oral_LD50'],3,hidden=8)


@pytest.mark.parametrize('name',[n for n in m.CANDIDATES+m.CONTROLS if n not in m.ANALYTIC])
def test_method_gradients_own_head_mask_and_chunk_invariance(name,episode):
    model=make(name); e=episode; labels=torch.tensor([[.3],[-.5],[1.2]])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    before={k:v.clone() for k,v in model.state_dict().items()}
    loss=model.loss(e,labels); assert torch.isfinite(loss); loss.backward()
    assert any(p.grad is not None and bool((p.grad!=0).any()) for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step(); assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    model.eval(); prediction=model(e)
    cp=e.context_functions.clone(); qp=e.query_functions.clone(); cp[:,0]+=1000; qp[:,0]-=1000
    torch.testing.assert_close(model(replace(e,context_functions=cp,query_functions=qp)),prediction)
    order=torch.tensor([6,0,3,1,5,2,4])
    perm=replace(e,context_rows=[e.context_rows[i] for i in order],context_features=e.context_features[order],
                 context_functions=e.context_functions[order],context_labels=e.context_labels[order])
    torch.testing.assert_close(model(perm),prediction,atol=2e-5,rtol=2e-5)
    chunks=[]
    for i in range(3):
        chunks.append(model(replace(e,query_rows=e.query_rows[i:i+1],query_features=e.query_features[i:i+1],query_functions=e.query_functions[i:i+1])))
    torch.testing.assert_close(torch.cat(chunks),prediction,atol=2e-5,rtol=2e-5)
    with pytest.raises(ValueError,match='train queries only'):
        model.loss(replace(e,query_rows=[replace(r,split='validation') for r in e.query_rows]),labels)


def test_tabr_equation_with_independent_numpy_attention(episode):
    model=make('TABR'); model.eval()
    with torch.no_grad():
        c,q=model.inputs(episode); hc,hq=model.encoder(c),model.encoder(q); kc,kq=model.key(hc),model.key(hq)
        logits=-((kq.numpy()[:,None]-kc.numpy()[None])**2).sum(-1)
        weights=np.exp(logits-logits.max(-1,keepdims=True)); weights/=weights.sum(-1,keepdims=True)
        values=model.label(episode.context_labels)[None]+model.correction(kq[:,None]-kc[None])
        expected=model.predictor(hq+torch.tensor((weights[...,None]*values.numpy()).sum(1)))
        torch.testing.assert_close(model(episode),expected)


def test_tabm_individual_losses_not_mean_prediction_loss(episode):
    model=make('TABM'); y=torch.randn(3,1); branches=model.branches(episode)
    assert branches.shape==(3,32)
    torch.testing.assert_close(model.loss(episode,y),(branches-y).square().mean())
    assert model.loss(episode,y) > (branches.mean(-1,keepdim=True)-y).square().mean()
    torch.testing.assert_close(model(episode),branches.mean(-1,keepdim=True))


def test_modernnca_prediction_is_explicit_label_weighted_kernel(episode):
    model=make('MODERNNCA'); c,q=model.inputs(episode)
    weights=(-m.squared_distance(model.embedding(q),model.embedding(c))).softmax(-1)
    torch.testing.assert_close(model(episode),weights@episode.context_labels)
    assert model(episode).min() >= episode.context_labels.min()
    assert model(episode).max() <= episode.context_labels.max()


def test_kernelicl_symmetric_query_embedding_for_identical_input(episode):
    model=make('KERNELICL')
    e=replace(episode,query_features=episode.context_features[:3].clone(),query_functions=episode.context_functions[:3].clone())
    c,q=model.symmetric_embeddings(e)
    torch.testing.assert_close(c[:3],q,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(model(e),(-m.squared_distance(q,c)*(torch.nn.functional.softplus(model.log_gamma)+1e-4)).softmax(-1)@e.context_labels)


def test_dnp_local_variance_literal_equation_and_kl():
    model=make('DNP'); c,q=torch.randn(5,model.input_dim),torch.randn(2,model.input_dim); y=torch.randn(5,1)
    mu,s=model.local_distribution(c,y,q)
    w=(-m.squared_distance(model.distance(q),model.distance(c)).clamp_min(1e-12).sqrt()/np.sqrt(8)).softmax(-1)
    a,b=model.local(torch.cat((c,y),-1)).chunk(2,-1)
    torch.testing.assert_close(mu,w@a)
    torch.testing.assert_close(s.square(),torch.exp(w[...,None]*b[None]).sum(1))
    pg=m.gaussian(torch.randn(2,16)); qg=m.gaussian(torch.randn(2,16))
    expected=torch.distributions.kl_divergence(torch.distributions.Normal(*qg),torch.distributions.Normal(*pg)).sum(-1).mean()
    torch.testing.assert_close(m.kl(qg,pg),expected)


def test_dimension_aggregator_variable_input_dimensions():
    model=m.DimensionAggregator(8)
    for dim in (6,56,104):
        out=model(torch.randn(3,dim),torch.randn(3,1)); assert out.shape==(3,8)
        out.square().mean().backward()
    assert model.value.weight.grad is not None


@pytest.mark.parametrize('name',('INP','DNP','DANP','BSA_TNP','KERNELICL','MODERNNCA','TABR','FCR'))
def test_context_labels_affect_predictions_without_query_labels(name,episode):
    model=make(name).eval(); other=replace(episode,context_labels=episode.context_labels+2.)
    assert not torch.allclose(model(episode),model(other))


def test_fixed_noise_does_not_consume_rng_and_is_antithetic():
    torch.manual_seed(42); state=torch.get_rng_state().clone()
    result=m.fixed_noise(['a','b'],8,torch.zeros(1))
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(result[:8],-result[8:])
    assert torch.equal(result[:,1:],m.fixed_noise(['b'],8,torch.zeros(1)))


@pytest.mark.parametrize('name',m.ANALYTIC)
def test_closed_form_finite_and_context_permutation(name,episode):
    for config in (0,1):
        y=m.closed_form(name,episode,config); assert y.shape==(3,1) and torch.isfinite(y).all()
        e=replace(episode,context_rows=episode.context_rows[::-1],context_features=episode.context_features.flip(0),
                  context_functions=episode.context_functions.flip(0),context_labels=episode.context_labels.flip(0))
        torch.testing.assert_close(y,m.closed_form(name,e,config))
