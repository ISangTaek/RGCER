from copy import deepcopy
import pytest
import torch
from reproducibility import state_dict_sha256
from tests.test_v9_srgt import fixture, one_cpu_thread
from v9_ssra import AdaptedGraph, Residual, METHODS, optimizer_for, save_smoke, load_smoke
from v9_c2a_source_stats import targets, derive


def payload_for(encoder):
    stats={}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        for name,module in targets(encoder).items():
            d=module.in_features;x=torch.randn(64,d,dtype=torch.float64)+.4
            sm=torch.stack([x[:32].sum(0),x[32:].sum(0)])
            ss=torch.stack([x[:32].T@x[:32],x[32:].T@x[32:]])
            stats[name]=dict(count=[32,32],sum_mean=sm,sum_second=ss,
                            derived=[derive(sm[i],ss[i],32) for i in range(2)],
                            pooled=derive(sm.sum(0),ss.sum(0),64))
    return dict(stats=stats)


@pytest.mark.parametrize('method',METHODS)
def test_initial_function_rng_source_and_active_context(method):
    base,batch,counts,_=fixture();base.eval();p=payload_for(base.encoder)
    rng=torch.get_rng_state().clone();m=AdaptedGraph(base,method,counts,p).eval()
    assert torch.equal(rng,torch.get_rng_state())
    for t in counts:
        assert torch.equal(m(batch,task_name=t)[t],base(batch,task_name=t)[t])
        assert m._active is None
    with pytest.raises(ValueError):m(batch)
    with pytest.raises(ValueError):m.encoder(batch)
    m.train();assert not m.encoder.training and m.decoders.training


@pytest.mark.parametrize('method',METHODS)
def test_gradients_source_freezing_train_replay_and_save_restore(method,tmp_path):
    from loss import QuantileRegressionLoss
    base,batch,counts,_=fixture();p=payload_for(base.encoder);m=AdaptedGraph(base,method,counts,p)
    opt=optimizer_for(m);initial=state_dict_sha256(m.encoder);t=m.task_name[0]
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss=QuantileRegressionLoss().compute_loss(m(batch,task_name=t)[t],batch.y)+m.regularization(t)
        loss.backward()
        assert all(x.grad is None for x in m.encoder.parameters())
        assert sum(float(x.grad.abs().sum()) for x in m.target.parameters() if x.grad is not None)>0
        opt.step()
    assert state_dict_sha256(m.encoder)==initial
    ident=dict(task_updates={k:3 if k==t else 0 for k in counts},method=method)
    path=tmp_path/'state.pt';save_smoke(path,m,opt,ident,3)
    fresh=AdaptedGraph(base,method,counts,p);load_smoke(path,fresh,optimizer_for(fresh),ident,3)
    m.eval();fresh.eval();assert torch.equal(m(batch,task_name=t)[t],fresh(batch,task_name=t)[t])
    assert state_dict_sha256(m)==state_dict_sha256(fresh)
    if method=='SSRA':
        before=m(batch,task_name=t)[t].detach().clone()
        m(batch,task_name=m.task_name[1]);assert torch.equal(before,m(batch,task_name=t)[t])


def test_penalty_equals_uncentered_empirical_update_energy():
    linear=torch.nn.Linear(24,12);p=payload_for(fixture()[0].encoder)
    x=torch.randn(40,24,dtype=torch.float64)+3
    mean=x.mean(0);cov=x.T@x/len(x)-torch.outer(mean,mean)
    eig,vec=torch.linalg.eigh(cov);proj=vec[:,-16:]@vec[:,-16:].T
    stat=dict(count=[20,20],sum_second=torch.stack([x[:20].T@x[:20],x[20:].T@x[20:]]),pooled=dict(projector=proj))
    a=Residual(linear,'SSRA',stat);a.B.data.normal_();beta=torch.tensor(.3)
    w=a.delta(beta);prediction=x.float()@w.T
    expected=prediction.square().sum(1).mean()/(torch.trace(a.M)*12)
    torch.testing.assert_close(a.penalty(beta),expected)
    centered=((w@cov.float())*w).sum()/(torch.trace(a.M)*12)
    assert a.penalty(beta)>centered


def test_rank_parameter_matching_and_task_specific_release():
    base,batch,counts,_=fixture();p=payload_for(base.encoder)
    a=AdaptedGraph(base,'LORA',counts,p);b=AdaptedGraph(base,'SSRA',counts,p)
    for key in a.target:
        assert a.target[key].A.shape==b.target[key].A.shape
        assert torch.equal(a.target[key].A,b.target[key].A)
        b.target[key].B.data.normal_()
    n=sum(x.numel() for x in b.target.parameters())-sum(x.numel() for x in a.target.parameters())
    assert n==len(base.encoder.backbone.layers)*len(counts)
    b.target.beta_logits.data[:,0]=5
    assert b.beta(0,b.task_name[0])>b.beta(0,b.task_name[1])
    for name,key in b._keys.items():assert torch.linalg.matrix_rank(b.target[key].delta(b.beta(0,b.task_name[0])),atol=1e-4)<=8


@pytest.mark.parametrize('change',['P','M','encoder','adam','identity'])
def test_forged_checkpoint_rejected(change,tmp_path):
    base,batch,counts,_=fixture();p=payload_for(base.encoder);m=AdaptedGraph(base,'SSRA',counts,p);opt=optimizer_for(m);t=m.task_name[0]
    (m(batch,task_name=t)[t].sum()+m.regularization(t)).backward();opt.step()
    ident=dict(task_updates={k:1 if k==t else 0 for k in counts});path=tmp_path/'state.pt';save_smoke(path,m,opt,ident,1)
    v=torch.load(path,weights_only=True)
    if change in ('P','M','encoder'):
        key=next(k for k in v['model_state'] if k.endswith('.'+change) or (change=='encoder' and k.startswith('encoder.') and v['model_state'][k].dtype==torch.float32))
        v['model_state'][key].add_(1)
    elif change=='adam':v['optimizer_state']['state'][0]['step']+=1
    else:v['identity']['unknown']=True
    torch.save(v,path);fresh=AdaptedGraph(base,'SSRA',counts,p)
    with pytest.raises(ValueError):load_smoke(path,fresh,optimizer_for(fresh),ident,1)
