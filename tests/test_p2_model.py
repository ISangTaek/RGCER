from copy import deepcopy
import pytest
import torch
from torch import nn
from p2_contract import P2Error
from p2_model import P2Optimization,build_model

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Module();self.encoder.backbone=nn.Linear(3,3)
        self.decoders=nn.ModuleDict({'head':nn.Linear(3,1)})
    def forward(self,x):return self.decoders['head'](self.encoder.backbone(x))

def update(m,c):
    c.optimizer.zero_grad(set_to_none=True);m(torch.ones(2,3)).square().mean().backward()
    c.optimizer.step();c.check()

@pytest.mark.parametrize('method,warmup',[('SOURCE',0),('FROZEN',40),('B1_low',0),('HF_low',5)])
def test_epoch_freeze_and_updates(method,warmup):
    torch.manual_seed(4);m=Tiny();c=P2Optimization(m,method);initial=deepcopy(m.encoder.state_dict())
    for epoch in range(6):
        c.begin_epoch(epoch);update(m,c)
        if epoch<warmup:
            assert all(torch.equal(v,initial[k]) for k,v in m.encoder.state_dict().items())
        else:assert any(not torch.equal(v,initial[k]) for k,v in m.encoder.state_dict().items())
    assert all(float(c.optimizer.state[p]['step'])==6 for _,p in c.heads)
    if method=='HF_low':assert all(float(c.optimizer.state[p]['step'])==1 for _,p in c.backbone)

@pytest.mark.parametrize('epoch',[True,-1,40,1,0.0])
def test_invalid_first_epoch(epoch):
    c=P2Optimization(Tiny(),'HF_low')
    with pytest.raises(P2Error):c.begin_epoch(epoch)

def test_frozen_weight_mutation():
    c=P2Optimization(Tiny(),'FROZEN');c.begin_epoch(0)
    with torch.no_grad():c.backbone[0][1].add_(1)
    with pytest.raises(P2Error):c.check()

def test_lr_mutation():
    c=P2Optimization(Tiny(),'B1_low');c.begin_epoch(0);c.optimizer.param_groups[0]['lr']=.01
    with pytest.raises(P2Error):c.check()

@pytest.mark.parametrize('method',['FROZEN','HF_low','B1_low','SOURCE'])
def test_optimizer_restore_continuity(method):
    torch.manual_seed(7);a=Tiny();original=deepcopy(a.state_dict());ca=P2Optimization(a,method)
    for e in range(5):ca.begin_epoch(e);update(a,ca)
    b=Tiny();b.load_state_dict(original);cb=P2Optimization(b,method)
    b.load_state_dict(a.state_dict());cb.restore(ca.snapshot())
    ca.begin_epoch(5);cb.begin_epoch(5);update(a,ca);update(b,cb)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())

def test_restore_rejects_wrong_method():
    c=P2Optimization(Tiny(),'HF_low');c.begin_epoch(0);update(c.model,c)
    with pytest.raises(P2Error):P2Optimization(Tiny(),'FROZEN').restore(c.snapshot())

@pytest.mark.parametrize('mutation',['head_missing','bad_moment','bad_step','bad_lr','extra_state'])
def test_restore_rejects_corruption(mutation):
    c=P2Optimization(Tiny(),'B1_low');c.begin_epoch(0);update(c.model,c)
    s=c.snapshot();opt=s['optimizer'];pid=opt['param_groups'][1]['params'][0]
    if mutation=='head_missing':del opt['state'][pid]
    if mutation=='bad_moment':opt['state'][pid]['exp_avg']=torch.zeros(1)
    if mutation=='bad_step':opt['state'][pid]['step']=torch.tensor(float('nan'))
    if mutation=='bad_lr':opt['param_groups'][0]['lr']=.1
    if mutation=='extra_state':opt['state'][999999]={}
    with pytest.raises(P2Error):P2Optimization(Tiny(),'B1_low').restore(s)

def test_real_model_pairing_without_forward():
    before=torch.get_rng_state().clone();a=build_model('source',42);b=build_model('source',42)
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    t1=build_model('target',42,a.encoder.state_dict())
    changed=deepcopy(a.encoder.state_dict());next(iter(changed.values())).add_(.01)
    t2=build_model('target',42,changed)
    assert all(torch.equal(v,t2.decoders.state_dict()[k]) for k,v in t1.decoders.state_dict().items())
    assert all(torch.equal(v,a.encoder.state_dict()[k]) for k,v in t1.encoder.state_dict().items())
    with pytest.raises(P2Error):build_model('target',42,{})

@pytest.mark.parametrize('role,seed,source',[('target',42,None),('source',True,None),('source',47,None),('bad',42,None)])
def test_invalid_model(role,seed,source):
    with pytest.raises(P2Error):build_model(role,seed,source)
