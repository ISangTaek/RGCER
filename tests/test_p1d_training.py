"""Synthetic Graphormer engine tests; no historical/local/server assets."""
import hashlib

import pytest
import torch

from dataset115_training import EpochConfig
from p1d_optimization import OptimizationSpec, OptimizationError
from tests.test_dataset115_training import runner


def make(arm):
    return runner(config=EpochConfig(6, 2, .001, 1e-5, 1), optimization=OptimizationSpec(arm))


def test_real_engine_original_arm_exact(tmp_path):
    old = runner(config=EpochConfig(6, 2, .001, 1e-5, 1)).run(tmp_path/'old')
    new = make('B1_high').run(tmp_path/'new')
    for a, b in zip(old['history'], new['history']):
        assert a == {k:v for k,v in b.items() if k != 'optimization'}
    a=torch.load(tmp_path/'old/epoch_005.pt',weights_only=True)
    b=torch.load(tmp_path/'new/epoch_005.pt',weights_only=True)
    assert a['model_digest']==b['model_digest']
    assert a['initial_heads']==b['initial_heads'] and a['initial_encoder']==b['initial_encoder']


@pytest.mark.parametrize('arm', ['B1_low','HF_high','HF_low'])
def test_real_engine_freeze_and_resume(tmp_path, arm):
    full=make(arm).run(tmp_path/'full')
    first=make(arm).run(tmp_path/'first',stop_after=5)
    path=tmp_path/'first/epoch_004.pt'
    resumed=make(arm).run(tmp_path/'resume',resume=path,resume_sha=hashlib.sha256(path.read_bytes()).hexdigest())
    assert full['history']==resumed['history']
    a=torch.load(tmp_path/'full/epoch_005.pt',weights_only=True)
    b=torch.load(tmp_path/'resume/epoch_005.pt',weights_only=True)
    assert a['model_digest']==b['model_digest']
    if arm.startswith('HF'):
        assert all(h['optimization']['frozen'] for h in first['history'])
        assert not full['history'][5]['optimization']['frozen']
        assert first['history'][4]['optimization']['backbone_optimizer_parameters']==0
        assert full['history'][5]['optimization']['backbone_optimizer_parameters']>0
        assert set(full['history'][5]['optimization']['head_optimizer_steps'].values())=={6}


def test_resume_rejects_optimizer_lr_change(tmp_path):
    make('HF_low').run(tmp_path/'first',stop_after=5)
    payload=torch.load(tmp_path/'first/epoch_004.pt',weights_only=True)
    payload['optimizer_state']['param_groups'][0]['lr']=.01
    bad=tmp_path/'bad.pt';torch.save(payload,bad)
    with pytest.raises(OptimizationError,match='group contract'):
        make('HF_low').run(tmp_path/'resume',resume=bad,resume_sha=hashlib.sha256(bad.read_bytes()).hexdigest())


def test_resume_unfreeze_rejects_self_consistent_frozen_drift(tmp_path):
    from dataset115_smoke import state_digest
    make('HF_low').run(tmp_path/'first',stop_after=5)
    payload=torch.load(tmp_path/'first/epoch_004.pt',weights_only=True)
    key=next(k for k in payload['model_state'] if k.startswith('encoder.backbone.'))
    payload['model_state'][key]=payload['model_state'][key]+1
    payload['model_digest']=state_digest(payload['model_state'])
    bad=tmp_path/'bad.pt';torch.save(payload,bad)
    with pytest.raises(OptimizationError,match='restored frozen tensor'):
        make('HF_low').run(tmp_path/'resume',resume=bad,resume_sha=hashlib.sha256(bad.read_bytes()).hexdigest())
