from copy import deepcopy
import math
import pytest
import torch
from torch.nn import functional as F
import v11_models as m
from tests.test_v11_features import chemical_value,data,value,single_thread


@pytest.mark.parametrize('name',m.NEW+m.CONTROLS)
def test_models_train_all_cores_and_evaluate_independently_of_query_batch(data,name):
    model=m.build(name,data,tiny=True); task=data.targets[0]; ids=data.by[task,'train'][:2]
    opt=m.optimizer(model,0)
    for step in range(2):
        opt.zero_grad(); e=data.episode(task,ids); loss=model.loss(e,data.y[ids]); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        m.schedule(opt,model,step,2); opt.step()
    model.eval(); q=data.by[task,'validation']; whole=model(data.episode(task,q,cap=None)).detach()
    single=torch.cat([model(data.episode(task,[i],cap=None)).detach() for i in q])
    assert torch.allclose(whole,single,atol=3e-5,rtol=1e-4)
    copy=m.build(name,data,tiny=True); copy.load_state_dict(model.state_dict()); copy.eval()
    assert torch.equal(whole,copy(data.episode(task,q,cap=None)).detach())


@pytest.mark.parametrize('name',m.NEW+m.CONTROLS)
def test_query_labels_are_not_visible_to_models(data,name):
    model=m.build(name,data,tiny=True).eval(); t=data.targets[0]; q=data.by[t,'validation']
    before=model(data.episode(t,q,cap=None)).detach()
    data.y[q]+=99999
    after=model(data.episode(t,q,cap=None)).detach()
    assert torch.equal(before,after)


def test_tcf_controls_match_capacity_at_formal_dimensions(data):
    for ns in (56,104):
        dims=(96+2*ns,2048,48,100)
        metadata={'vocabulary':{'population':list(range(80)),'route':list(range(10)),'endpoint':list(range(8)),'domain':[0,1]}}
        args=(96,list(map(str,range(ns))),metadata); c=m.CONFIGS['TCF']
        tcf=m.TCF(*args,config=c); graph=m.TCF(*args,config=c,graph_only=True); plain=m.ConcatMLP(*args,config=c)
        assert m.count(tcf)==m.tcf_count(dims,64,4)
        assert abs(m.count(graph)/m.count(tcf)-1)<.05 and abs(m.count(plain)/m.count(tcf)-1)<.05


def test_tokenization_covers_every_input_exactly_once(data):
    model=m.build('FT_TRANSFORMER_MV',data,tiny=True)
    indices=[i for block in model.tokens.parts for i in block]
    assert sorted(indices)==list(range(model.input_dim))


def test_machine_readable_method_manifest_matches_executable_configs():
    import json
    from pathlib import Path
    p=Path(m.__file__).parent/'configs/v11_s1_method_manifest.json'
    manifest=json.loads(p.read_text())
    assert manifest['configurations']==m.CONFIGS and manifest['configuration_multiplier']==[1.,.3]
    assert len(manifest['upstream'])==9


def test_batchensemble_forward_and_gradients_equal_explicit_members():
    torch.manual_seed(4); layer=m.BatchEnsembleLinear(5,3,4,True).double()
    x=torch.randn(2,4,5,dtype=torch.float64,requires_grad=True)
    expected=torch.stack([F.linear(x[:,k]*layer.r[k],layer.weight)*layer.s[k]+layer.bias[k] for k in range(4)],1)
    actual=layer(x)
    assert torch.allclose(actual,expected,atol=1e-12)
    args=(x,*layer.parameters())
    a=torch.autograd.grad(actual.square().sum(),args,retain_graph=True); b=torch.autograd.grad(expected.square().sum(),args)
    assert all(torch.allclose(u,v,atol=1e-10) for u,v in zip(a,b))


def test_t2g_hard_graph_has_no_self_edges_or_readout_key_and_has_gradient():
    layer=m.Attention(8,2,9,'t2g'); a=layer.adjacency()
    assert (a.diagonal(dim1=-2,dim2=-1)==0).all() and (a[:,:,0]==0).all()
    assert set(a.detach().unique().tolist())<={0.,1.}
    layer(torch.randn(3,9,8)).square().sum().backward()
    assert layer.columns.grad.abs().sum()>0 and layer.relation.grad.abs().sum()>0


def test_excel_causal_attention_cannot_read_future_tokens():
    layer=m.Attention(8,2,5,'excel').eval(); x=torch.randn(2,5,8); y=x.clone(); y[:,3:]+=100
    assert torch.equal(layer(x)[:,:3],layer(y)[:,:3])


def test_am_product_is_per_row_and_both_prompt_branches_train():
    layer=m.AMBlock(8,2,5,3).eval(); x=torch.randn(3,5,8)
    assert torch.allclose(layer(x)[:1],layer(x[:1]),atol=1e-6)
    layer(x).square().sum().backward()
    for branch in (layer.sum,layer.product):
        assert branch.q.weight.grad.abs().sum()>0 and branch.k.weight.grad.abs().sum()>0


def test_mamba_selective_recurrence_matches_closed_form_and_gradient():
    torch.manual_seed(8); layer=m.SelectiveSSM(4,3,2,2).double(); x=torch.randn(2,4,4,dtype=torch.float64,requires_grad=True)
    raw,z=layer.input(x).chunk(2,-1); v=F.silu(layer.conv(raw.transpose(1,2))[...,:4].transpose(1,2))
    dt,b,c=torch.split(layer.select(v),(layer.rank,3,3),-1); dt=F.softplus(layer.dt(dt))
    a=torch.exp(dt[:,:,:,None]*(-layer.A_log.exp())); bx=dt[:,:,:,None]*b[:,:,None,:]*v[:,:,:,None]
    ys=[]
    for t in range(4):
        h=sum(bx[:,j]*a[:,j+1:t+1].prod(1) for j in range(t+1))
        ys.append((h*c[:,t,None,:]).sum(-1)+layer.D*v[:,t])
    expected=layer.output(torch.stack(ys,1)*F.silu(z)); actual=layer(x)
    assert torch.allclose(actual,expected,atol=1e-12)
    args=(x,*layer.parameters()); ga=torch.autograd.grad(actual.square().sum(),args,retain_graph=True)
    gb=torch.autograd.grad(expected.square().sum(),args)
    assert all(torch.allclose(a,b,atol=1e-10) for a,b in zip(ga,gb))


def test_grande_hard_paths_select_one_leaf_and_train_straight_through(data):
    model=m.build('GRANDE_MV',data,tiny=True); e=data.episode(data.targets[0],data.by[data.targets[0],'train'][:2])
    model.loss(e,data.y[:2]).backward()
    assert model.selectors.grad.abs().sum()>0 and model.thresholds.grad.abs().sum()>0
    assert model.leaves.grad.abs().sum()>0 and model.weights.grad.abs().sum()>0


def test_modernnca_uses_euclidean_distance_and_train_only_bank(data):
    model=m.build('MODERNNCA_MV',data,tiny=True).eval(); t=data.targets[0]; e=data.episode(t,data.by[t,'validation'])
    c=model.post(model.encoder(model.flat(e,True))); q=model.post(model.encoder(model.flat(e)))
    expected=(-torch.cdist(q,c,p=2)).softmax(-1)@e.base.context_labels
    assert torch.allclose(model(e),expected,atol=1e-6)


def test_realmlp_author_schedule_and_parameter_group_rates(data):
    model=m.build('REALMLP_MV',data,tiny=True); opt=m.optimizer(model,0)
    assert [g['lr'] for g in opt.param_groups]==pytest.approx([.07,.007,.42])
    for step in (0,1,8,39):
        m.schedule(opt,model,step,40); expected=.5*(1-math.cos(2*math.pi*math.log2(1+15*step/40)))
        assert opt.param_groups[0]['lr']==.07*expected


@pytest.mark.parametrize('name',m.NEW+m.CONTROLS)
def test_formal_width_forward_backward_at_104_source_dimensions(data,name):
    # Synthetic tensors only; formal configuration, largest source-function bank.
    from dataclasses import replace
    from types import SimpleNamespace
    t=data.targets[0]; e=data.episode(t,data.by[t,'train'][:2]); b=e.base
    sources=[f'species{i}_oral_LD50' for i in range(104)]
    metadata={'vocabulary':{'population':list(range(105)),'route':['oral'],'endpoint':['LD50','TDLo'],'domain':['A','B']}}
    md=sum(map(len,metadata['vocabulary'].values()))
    fake=SimpleNamespace(h=torch.zeros(1,96),sources=sources,value={'metadata':metadata},device='cpu',
                         chemical={'robust_center':torch.zeros(96+208+2048+48+md),'robust_factors':torch.ones(96+208+2048+48+md)})
    b=replace(b,context_features=torch.randn(len(b.context_rows),96),query_features=torch.randn(2,96),
              context_functions=torch.randn(len(b.context_rows),104),query_functions=torch.randn(2,104),metadata=torch.zeros(1,md))
    e=replace(e,base=b); model=m.build(name,fake); loss=model.loss(e,data.y[:2]); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.eval()(e).shape==(2,1)
