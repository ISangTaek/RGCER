import pytest
import torch
from tests.test_p1d_tox import fixture_factory,TASKS
from p1d_tox_epoch import run_epoch
from p1d_runtime import same
from p1d_optimization import OptimizationError


def test_original_full_epoch_and_controlled_b1_exact():
    f=fixture_factory();a=f.make_trainer(None);b=f.make_trainer('B1_high')
    for epoch in range(2):
        ra=run_epoch(a,f.loaders['train'],epoch)
        rb=run_epoch(b,f.loaders['train'],epoch)
        assert ra['loss']==rb['loss'] and ra['updates']==rb['updates']==3
        same(a.model.state_dict(),b.model.state_dict(),'full epoch model')
        same(a.optimizer.state_dict(),b.optimizer.state_dict(),'full epoch Adam')


def test_hf_uses_original_epoch_sampler_and_unfreezes():
    f=fixture_factory();t=f.make_trainer('HF_low')
    initial={n:p.detach().clone() for n,p in t.optimization.backbone}
    for epoch in range(6):
        r=run_epoch(t,f.loaders['train'],epoch)
        assert r['updates']==3 and r['cumulative_updates']==3*(epoch+1)
        assert r['task_samples']=={task:2 for task in TASKS}
        assert all(v==epoch+1 for v in r['optimization']['head_optimizer_steps'].values())
        if epoch<5:
            assert all(torch.equal(initial[n],p) for n,p in t.optimization.backbone)
            assert r['optimization']['backbone_optimizer_parameters']==0
    assert any(not torch.equal(initial[n],p) for n,p in t.optimization.backbone)


def test_reject_changed_sampler_before_any_updates():
    f=fixture_factory();t=f.make_trainer('HF_low');t.args.task_sampling='uniform'
    with pytest.raises(OptimizationError,match='proportional'):run_epoch(t,f.loaders['train'],0)
    assert t.optimizer_updates==0
