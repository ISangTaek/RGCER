from copy import deepcopy
from itertools import product
import pytest
from p1d_schedule import ARMS, SETTINGS, job, resolve_plan, screening_jobs


def selections(bits):
    result={}
    for i,s in enumerate(SETTINGS):
        scores={a:dict(best_epoch=0,best_validation_macro_rmse=2.) for a in ARMS}
        winners={f:f+('_low' if bits[2*i+j] else '_high') for j,f in enumerate(('B1','HF'))}
        for a in winners.values():scores[a]['best_validation_macro_rmse']=1.
        result[s]=dict(setting=s,seed=42,scores=scores,selected=winners,
                       scope='DEVELOPMENT_VALIDATION_ONLY',acceptance_status='PENDING_REVIEW')
    return result


def test_screen_has_eight_new_and_no_implicit_retry():
    rows=screening_jobs()
    assert len(rows)==12 and sum(r['action']=='TRAIN' for r in rows)==8
    assert sum(r['new_updates'] for r in rows)==3360
    for r in rows:
        if r['action']=='REUSE':assert r['missing_reuse_action']=='STOP_NOT_RETRAIN'


@pytest.mark.parametrize('bits',list(product((False,True),repeat=6)))
def test_every_selection_branch(bits):
    p=resolve_plan(selections(bits)); rows=p['jobs']; b=p['budget']
    expected=20 + 4*sum(bits[i] for i in (0,2,4)) - 2*bits[1]
    assert b['new_trajectories']==expected
    assert b['model_epochs']==40*expected
    assert b['optimizer_updates']==sum(r['new_updates'] for r in rows)
    assert b['new_trajectories']<=32 and b['model_epochs']<=1280 and b['optimizer_updates']<=12960
    historical=[r for r in rows if r['alias'] and r['alias'].startswith('s3_')]
    assert {r['seed'] for r in historical}==({42,44,46} if bits[1] else {42})
    assert not p['execution_authorized'] and not p['test_authorized']
    assert not p['source_pretraining_authorized'] and not b['saved_budget_reallocation']


def test_exact_tie_uses_high_lr():
    s=selections((False,)*6)
    for v in s.values():
        for score in v['scores'].values():score['best_validation_macro_rmse']=1.
    assert resolve_plan(s)['budget']['new_trajectories']==20
    s['ToxAcute']['selected']['HF']='HF_low'
    with pytest.raises(ValueError,match='tie rule'):resolve_plan(s)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,True,'1'])
def test_invalid_score_never_drops_an_arm(value):
    s=selections((False,)*6)
    s['A']['scores']['HF_low']['best_validation_macro_rmse']=value
    with pytest.raises(ValueError,match='selection score'):resolve_plan(s)


@pytest.mark.parametrize('mutation', ['missing_setting','missing_arm','test','wrong_seed','pass','unknown'])
def test_fail_closed(mutation):
    s=selections((False,)*6)
    if mutation=='missing_setting':del s['B']
    if mutation=='missing_arm':del s['A']['scores']['HF_low']
    if mutation=='test':s['A']['scope']='test'
    if mutation=='wrong_seed':s['A']['seed']=43
    if mutation=='pass':s['A']['acceptance_status']='PASS'
    if mutation=='unknown':s['A']['extra']=True
    with pytest.raises(ValueError):resolve_plan(s)


def test_inputs_not_mutated():
    s=selections((True,)*6);saved=deepcopy(s)
    resolve_plan(s)
    assert s==saved


@pytest.mark.parametrize('args',[('A','B0',42,'screen'),('ToxAcute','HF_low',True,'screen'),
                               ('ToxAcute','HF_low',43,'screen'),('A','HF_low',42,'replication')])
def test_job_rejects_out_of_scope(args):
    with pytest.raises(ValueError):job(*args)


def test_frozen_reuse_lock_and_request_correspond():
    import json
    from pathlib import Path
    from p1d_schedule import S3_SEEDS
    config=Path(__file__).resolve().parents[1]/'configs'
    lock=json.loads((config/'p1d_accepted_s3_reuse.json').read_bytes())
    request=json.loads((config/'p1d2_reuse_asset_request.json').read_bytes())
    assert lock['schema']=='p1d_accepted_s3_reuse_v1'
    assert lock['decision']=='REUSE_TRAINING_ASSET'
    assert lock['initial_model_pt_role']=='PRE_OVERLAY_SCRATCH_EVIDENCE_ONLY'
    assert not lock['execution_authorized'] and not lock['test_authorized']
    assert tuple(r['seed'] for r in lock['runs'])==S3_SEEDS
    for r,q in zip(lock['runs'],request['runs']):
        assert r['run_id']==q['run_id'] and r['relative_path']==q['relative_path']
        assert r['best_epoch']==q['best_epoch']
        assert abs(r['best_validation_macro_rmse']-q['best_validation_macro_rmse'])<1e-12
        assert r['initial_training_tensor_sha256']==q['initial_model_sha256']
        assert r['training_init_file_sha256']==q['init_asset_sha256']
        assert set(r['weight_sha256'])=={w['name'] for w in q['weights']}
        assert all(len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in r['weight_sha256'].values())
