from copy import deepcopy
from pathlib import Path
import pytest
import torch

import v9_s6_screen as m
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory


def test_worker_ids_do_not_collide_when_log_suffix_is_applied():
    names = [name for _, name, _ in m.sequence()]
    assert all(Path(name).suffix == '' for name in names)
    assert len({str(Path(n).with_suffix('.log')) for n in names}) == len(names)
    assert all(str(Path(n).with_suffix('.command.json')) == n+'.command.json' for n in names)


@pytest.mark.parametrize('setting', m.s3.s2.c0.SETTINGS)
def test_all_fifteen_paths_internal_selection_full_refit(setting, joint, tmp_path):
    f, t, source, ref = joint(setting); out = tmp_path/'case'; out.mkdir()
    job = next(j for j in m.jobs() if j['setting'] == setting)
    result = m.train_case(f, t, source, job, out, 'a'*40, 'cpu', ref, dict(m.SPEC, max_epochs=2))
    assert len(result['results']) == 15 and not result['test_evaluated']
    chosen = m.read(out/'frozen_selection.json')
    assert chosen['outer_labels_used'] is False
    assert result['inner_updates_per_engine'] == result['target_updates_per_engine']
    policies = m.read(out/'frozen_policies.json')
    for epoch, policy in enumerate(policies):
        full = m.s3.s2.epoch_plan(*m.s3.s2.c0.tasks_and_counts(setting), m.s3.s2.c0.BATCH_SIZES[setting], epoch)
        assert [r['full_target'] for r in policy['steps']] == full
    for name, r in result['results'].items():
        assert r['chosen'] == chosen['selected'][name]
        assert r['checkpoint_sha256'] == m.sha(out/(name+'.pt'))
    f, t, source, ref = joint(setting)
    checked = m.verify_case(f, t, source, job, out, 'a'*40, ref, dict(m.SPEC, max_epochs=2))
    assert checked == result


def test_lr_selection_cannot_accept_outer_scores():
    records = []
    for lr in m.SPEC['lrs']:
        records.append(dict(job=dict(id=str(lr), lr=lr, setting='A'),
                            selected={n: dict(inner_macro_rmse=1.) for n in m.method.METHODS}))
    assert set(m.choose_lrs(records).values()) == {str(.0001)}
    records[1]['selected']['PROTECTED_GATE']['inner_macro_rmse'] = .5
    assert m.choose_lrs(records)['PROTECTED_GATE'] == str(.001)


@pytest.fixture
def completed_case(joint, tmp_path, monkeypatch):
    f, t, source, ref = joint('A'); out = tmp_path/'case'; out.mkdir()
    job = next(j for j in m.jobs() if j['setting'] == 'A'); spec = dict(m.SPEC, max_epochs=1)
    original = m.method.learn_weights
    def guarded(*args, **kwargs):
        assert not (out/'frozen_selection.json').exists(), 'controller called during full refit'
        return original(*args, **kwargs)
    monkeypatch.setattr(m.method, 'learn_weights', guarded)
    m.train_case(f, t, source, job, out, 'a'*40, 'cpu', ref, spec, smoke=True)
    return job, out, spec


@pytest.mark.parametrize('damage', ('metric', 'checkpoint', 'policy', 'selection', 'optimizer_steps', 'prediction_forgery', 'source_identity', 'inner_scaler'))
def test_fail_closed_against_result_and_training_forgery(damage, completed_case, joint):
    job, out, spec = completed_case; receipt = m.read(out/'receipt.json')
    name = 'TARGET_FULL'
    if damage == 'metric':
        receipt['results'][name]['selected']['macro_rmse'] += .1
    elif damage == 'checkpoint':
        checkpoint = torch.load(out/(name+'.pt'), weights_only=True)
        checkpoint['identity']['source']['heads_sha256'] = '0'*64
        torch.save(checkpoint, out/(name+'.pt')); receipt['results'][name]['checkpoint_sha256'] = m.sha(out/(name+'.pt'))
    elif damage == 'policy':
        p = m.read(out/'frozen_policies.json'); p[0]['steps'][0]['weights']['TARGET_FULL'][0] = .1
        (out/'frozen_policies.json').write_text(__import__('json').dumps(p), encoding='utf-8')
        receipt['policy_sha256'] = m.sha(out/'frozen_policies.json')
    elif damage == 'selection':
        s = m.read(out/'frozen_selection.json'); s['selected'][name]['inner_macro_rmse'] = -1.
        (out/'frozen_selection.json').write_text(__import__('json').dumps(s), encoding='utf-8')
        receipt['selection_sha256'] = m.sha(out/'frozen_selection.json')
    elif damage == 'optimizer_steps':
        p = m.read(out/'refit_epoch_001.json'); p['optimizer_steps'][name] = {}
        (out/'refit_epoch_001.json').write_text(__import__('json').dumps(p), encoding='utf-8')
    elif damage == 'prediction_forgery':
        p = m.read(out/(name+'_validation.json')); p[0]['prediction'] += 1.
        (out/(name+'_validation.json')).write_text(__import__('json').dumps(p), encoding='utf-8')
        receipt['results'][name]['selected'] = m.metric(p)
        receipt['results'][name]['prediction_sha256'] = m.sha(out/(name+'_validation.json'))
    elif damage == 'source_identity':
        receipt['identity']['source']['heads_sha256'] = '0'*64
    else:
        p = m.read(out/'inner_scalers.json'); p[next(iter(p))]['mean'] += 1.
        (out/'inner_scalers.json').write_text(__import__('json').dumps(p), encoding='utf-8')
    (out/'receipt.json').write_text(__import__('json').dumps(receipt), encoding='utf-8')
    f, t, source, ref = joint('A')
    with pytest.raises(ValueError): m.verify_case(f, t, source, job, out, 'a'*40, ref, spec, smoke=True)


def test_zero_learned_policy_equals_full_target_in_complete_refit(joint, tmp_path, monkeypatch):
    f, t, source, ref = joint('A'); job = next(j for j in m.jobs() if j['setting'] == 'A')
    monkeypatch.setattr(m.method, 'learn_weights', lambda *args, **kwargs: [0.]*len(args[9]))
    result = m.train_case(f, t, source, job, tmp_path, 'a'*40, 'cpu', ref, dict(m.SPEC, max_epochs=1), smoke=True)
    baseline = m.read(tmp_path/'TARGET_FULL_validation.json')
    for name in m.method.GRADIENT_METHODS:
        assert m.read(tmp_path/(name+'_validation.json')) == baseline
        assert result['results'][name]['paired_control'] == result['results'][name]['selected']


def test_parent_verification_uses_fresh_children_without_opening_datastores(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'check_launch', lambda *a: {})
    monkeypatch.setattr(m, 'gate_check', lambda *a: None)
    monkeypatch.setattr(m.s3, 'references', lambda *a: {})
    monkeypatch.setattr(m, 'compare', lambda *a: {})
    monkeypatch.setattr(m.s3, 'context', lambda *a: pytest.fail('parent opened LMDB'))
    visited = []
    def call(argv, prefix, env):
        assert env['CUDA_VISIBLE_DEVICES'] == ''
        name = argv[argv.index('--job')+1]; visited.append(name)
        m.write(tmp_path/name/'verification.json', dict(task=m.TASK, commit='a'*40, job=name, pid=len(visited), device='cpu',
                content_status='PASS', raw_prediction_replayed=True, new_optimizer_updates=0,
                receipt_sha256=m.sha(tmp_path/name/'receipt.json')))
    monkeypatch.setattr(m.s5, 'call_worker', call)
    for _, name, _ in m.sequence():
        (tmp_path/name).mkdir(); m.write(tmp_path/(name+'.command.json'), dict(exit_code=0))
        m.write(tmp_path/name/'receipt.json', dict(model_training_runs=22, model_epochs=880))
    result = m.verify(tmp_path, 'a'*40, tmp_path/'split', tmp_path/'source')
    assert len(visited) == 9 and result['model_training_runs'] == 132 and result['model_epochs'] == 5280
