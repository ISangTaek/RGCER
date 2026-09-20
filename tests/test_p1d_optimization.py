import copy

import pytest
import torch
from torch import nn

from p1d_optimization import ARMS, OptimizationControl, OptimizationError, OptimizationSpec, select_arms


def model():
    torch.manual_seed(42)
    m = nn.Module()
    m.encoder = nn.Module()
    m.encoder.backbone = nn.Sequential(nn.Linear(2, 3), nn.Dropout(.1))
    m.decoders = nn.Linear(3, 1)
    return m


def update(m, opt):
    opt.zero_grad(set_to_none=True)
    m.decoders(m.encoder.backbone(torch.ones(2, 2))).square().mean().backward()
    opt.step()


@pytest.mark.parametrize('arm', ARMS)
def test_boundary_and_head_continuity(arm):
    m = model(); c = OptimizationControl(m, OptimizationSpec(arm)); m.train()
    start = copy.deepcopy(m.state_dict())
    for epoch in range(6):
        rng = torch.get_rng_state().clone()
        c.begin_epoch(epoch)
        assert torch.equal(rng, torch.get_rng_state()) and m.encoder.backbone.training
        update(m, c.optimizer); r = c.record()
        assert set(r['head_optimizer_steps'].values()) == {epoch + 1}
        c.validate_optimizer_state(c.optimizer.state_dict(), epoch)
        if epoch < c.spec.warmup_epochs:
            assert r['backbone_optimizer_parameters'] == 0
            assert all(torch.equal(p, start[n]) for n,p in m.state_dict().items() if n.startswith('encoder.'))
    assert any(not torch.equal(p, start[n]) for n,p in m.state_dict().items() if n.startswith('encoder.'))


def test_original_single_group_numerical_equivalence():
    a = model(); b = copy.deepcopy(a)
    old = torch.optim.AdamW(a.parameters(), lr=.001, weight_decay=1e-5)
    new = OptimizationControl(b, OptimizationSpec('B1_high'))
    assert len(new.optimizer.param_groups) == 1
    for epoch in range(6):
        new.begin_epoch(epoch)
        rng = torch.get_rng_state().clone(); update(a, old)
        torch.set_rng_state(rng); update(b, new.optimizer)
        assert all(torch.equal(v, b.state_dict()[k]) for k,v in a.state_dict().items())


@pytest.mark.parametrize('arm', ARMS)
def test_actual_legacy_trainer_optimizer_and_freeze_match(arm):
    from types import SimpleNamespace
    from trainer import Trainer
    legacy = Trainer.__new__(Trainer)
    legacy.model = model()
    legacy.loss_balancer = nn.Module()
    spec = OptimizationSpec(arm)
    legacy.args = SimpleNamespace(backbone_lr_multiplier=spec.backbone_lr/.001,
                                  freeze_backbone_epochs=spec.warmup_epochs,
                                  trainable_last_blocks=0)
    legacy.optimizer = legacy._make_optimizer(dict(optim='adamw', lr=.001, weight_decay=1e-5))
    other = copy.deepcopy(legacy.model)
    control = OptimizationControl(other, spec)
    for epoch in range(6):
        legacy._apply_backbone_freeze(epoch)
        control.begin_epoch(epoch)
        rng = torch.get_rng_state().clone(); update(legacy.model, legacy.optimizer)
        torch.set_rng_state(rng); update(other, control.optimizer)
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in legacy.model.state_dict().items())


@pytest.mark.parametrize('epoch', [0, 4, 5])
def test_control_resume_no_rng_reset(epoch):
    a = model(); c = OptimizationControl(a, OptimizationSpec('HF_low'))
    for e in range(epoch+1):
        c.begin_epoch(e); update(a,c.optimizer)
    b = model(); d = OptimizationControl(b, OptimizationSpec('HF_low'))
    b.load_state_dict(a.state_dict()); state=copy.deepcopy(c.optimizer.state_dict())
    d.validate_optimizer_state(state,epoch); d.optimizer.load_state_dict(state)
    c.begin_epoch(epoch+1);d.begin_epoch(epoch+1)
    rng=torch.get_rng_state().clone();update(a,c.optimizer)
    torch.set_rng_state(rng);update(b,d.optimizer)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


@pytest.mark.parametrize('bad', ['readout', 'frozen', 'lr', 'frozen_state', 'frozen_tensor', 'epoch'])
def test_controller_fail_closed(bad):
    m=model()
    if bad=='readout':m.encoder.readout=nn.Linear(3,3)
    if bad=='frozen':next(m.parameters()).requires_grad_(False)
    if bad in ('readout','frozen'):
        with pytest.raises(OptimizationError):OptimizationControl(m,OptimizationSpec('HF_low'))
        return
    c=OptimizationControl(m,OptimizationSpec('HF_low'));c.begin_epoch(0)
    if bad=='lr':c.optimizer.param_groups[0]['lr']=.1
    if bad=='frozen_state':c.optimizer.state[c.backbone[0][1]]={'step':torch.tensor(1.)}
    if bad=='frozen_tensor':
        with torch.no_grad():c.backbone[0][1].add_(1)
    if bad=='epoch':
        with pytest.raises(OptimizationError):c.begin_epoch(True)
    else:
        with pytest.raises(OptimizationError):c.verify_frozen()


def histories():
    return {a:[dict(epoch=e,split='validation',endpoints={t:dict(n=2,rmse=1.) for t in ('a','b','c')})
               for e in range(40)] for a in ARMS}


def test_selector_ties_and_recompute():
    h=histories()
    for rows in h.values():
        for r in rows:r['macro_rmse']=-999  # ignored untrusted summary field
    out=select_arms(h,setting='ToxAcute',seed=42,task_names=['a','b','c'])
    assert out['selected']=={'B1':'B1_high','HF':'HF_high'}
    assert all(s['best_epoch']==0 and s['best_validation_macro_rmse']==1 for s in out['scores'].values())
    for t in ('a','b','c'):h['HF_low'][7]['endpoints'][t]['rmse']=.5
    assert select_arms(h,setting='ToxAcute',seed=42,task_names=['a','b','c'])['selected']['HF']=='HF_low'


@pytest.mark.parametrize('bad', ['missing_arm','missing_epoch','missing_task','nan','negative','bool_count','changed_count','test','wrong_epoch','seed'])
def test_selector_rejects_invalid_even_if_other_arm_wins(bad):
    h=histories();r=h['HF_low'][0]
    if bad=='missing_arm':del h['HF_low']
    if bad=='missing_epoch':h['HF_low'].pop()
    if bad=='missing_task':del r['endpoints']['a']
    if bad=='nan':r['endpoints']['a']['rmse']=float('nan')
    if bad=='negative':r['endpoints']['a']['rmse']=-1
    if bad=='bool_count':r['endpoints']['a']['n']=True
    if bad=='changed_count':r['endpoints']['a']['n']=3
    if bad=='test':r['split']='test'
    if bad=='wrong_epoch':r['epoch']=True
    with pytest.raises(OptimizationError):
        select_arms(h,setting='ToxAcute',seed=43 if bad=='seed' else 42,task_names=['a','b','c'])
