from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random
import zipfile

import numpy as np
import pytest
import torch

import v9_s6r1_screen as m
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory


COMMIT = 'a'*40


def replace(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def source_case(joint, tmp_path, job, epochs=2):
    old = tmp_path/'old'; old.mkdir()
    f, t, source, ref = joint(job['setting'])
    m.s6.train_case(f, t, source, job, old, m.SOURCE_COMMIT, 'cpu', ref, dict(m.s6.SPEC, max_epochs=epochs))
    return old, dict(m.SPEC, max_epochs=epochs)


@pytest.mark.parametrize('job', m.jobs(), ids=lambda j: j['id'])
def test_refit_reproduces_original_and_records_every_epoch_without_calibration(job, joint, tmp_path, monkeypatch):
    old, spec = source_case(joint, tmp_path, job)
    hashes = {p.name: m.sha(p) for p in old.iterdir()}
    def prohibited(*args, **kwargs): pytest.fail('new calibration called')
    monkeypatch.setattr(m.method, 'learn_weights', prohibited)
    monkeypatch.setattr(m.method, 'gate_coefficients', prohibited)
    monkeypatch.setattr(m.s6, 'snapshots_and_choices', prohibited)
    f, t, source, ref = joint(job['setting']); out = tmp_path/'replay'
    receipt = m.refit_case(f, t, source, job, old, out, COMMIT, 'cpu', ref, spec)
    assert receipt['model_training_runs'] == 11 and receipt['model_epochs'] == 22
    assert receipt['calibration_updates'] == receipt['extra_smoke_updates'] == 0
    assert receipt['total_updates'] == 11*receipt['target_updates_per_engine']
    for epoch in (1, 2):
        predictions = m.read(out/f'validation_epoch_{epoch:03d}.json')
        assert set(predictions['methods']) == set(m.method.METHODS)
        for name in m.method.METHODS:
            saved = torch.load(out/'development'/(name+'.pt'), weights_only=True)
            assert saved['frozen_chosen'] == m.read(old/'frozen_selection.json')['selected'][name]
    f, t, source, ref = joint(job['setting'])
    assert m.verify_case(f, t, source, job, old, out, COMMIT, ref, spec) == receipt
    assert hashes == {p.name: m.sha(p) for p in old.iterdir()}


@pytest.mark.parametrize('fail', (False, True))
def test_diagnostic_forward_preserves_rng_and_modes_even_on_exception(fail, joint, monkeypatch):
    f, t, source, _ = joint('A'); job = next(j for j in m.jobs() if j['setting'] == 'A')
    original = t.model; model = m.method.build(t.model, source, 'TARGET_FULL', job['lr'], 'cpu')[0]
    list(model.modules())[1].eval(); modes = [x.training for x in model.modules()]
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    def consume(*args, **kwargs):
        random.random(); np.random.rand(); torch.rand(7); model.eval()
        if fail: raise RuntimeError('forward failed')
        return [{'prediction': 1.}]
    monkeypatch.setattr(m.s6, 'evaluate', consume)
    evaluate = m.prediction_evaluator(f, t, job, model, [], {}, 'cpu')
    if fail:
        with pytest.raises(RuntimeError, match='forward failed'): evaluate(m.method.state(model))
    else:
        assert evaluate(m.method.state(model)) == [{'prediction': 1.}]
    assert t.model is original and modes == [x.training for x in model.modules()]
    assert before[0] == random.getstate() and torch.equal(before[2], torch.get_rng_state())
    after = np.random.get_state()
    assert before[1][0] == after[0] and np.array_equal(before[1][1], after[1]) and before[1][2:] == after[2:]


def test_extra_random_consuming_evaluation_cannot_change_refit(joint, tmp_path, monkeypatch):
    job = next(j for j in m.jobs() if j['setting'] == 'A')
    old, spec = source_case(joint, tmp_path, job)
    original = m.s6.evaluate
    def consume(*args, **kwargs):
        result = original(*args, **kwargs)
        random.random(); np.random.rand(); torch.rand(19)
        return result
    monkeypatch.setattr(m.s6, 'evaluate', consume)
    f, t, source, ref = joint('A')
    receipt = m.refit_case(f, t, source, job, old, tmp_path/'replay', COMMIT, 'cpu', ref, spec)
    assert all(v['exact_original_state'] for v in receipt['original'].values())


@pytest.fixture
def replay_case(joint, tmp_path):
    job = next(j for j in m.jobs() if j['setting'] == 'A')
    old, spec = source_case(joint, tmp_path, job, epochs=1)
    f, t, source, ref = joint('A'); out = tmp_path/'replay'
    m.refit_case(f, t, source, job, old, out, COMMIT, 'cpu', ref, spec)
    return job, old, out, spec


@pytest.mark.parametrize('damage', ('metric', 'raw_prediction', 'checkpoint_identity', 'recipe', 'optimizer',
                                    'source_copy', 'missing_epoch', 'receipt_scope', 'original_tensor'))
def test_verifier_rejects_forgery_and_incomplete_replay(damage, replay_case, joint):
    job, old, out, spec = replay_case; r = m.read(out/'receipt.json'); name = 'TARGET_FULL'
    history = m.read(out/'epoch_001.json')
    if damage == 'metric': r['development'][name]['selected']['macro_rmse'] += .1
    elif damage == 'raw_prediction':
        path = out/'validation_epoch_001.json'; data = m.read(path)
        data['methods'][name][0]['prediction'] += 1.
        replace(path, data); history['predictions_sha256'] = m.sha(path)
        history['results'][name]['selected'] = m.s6.metric(data['methods'][name])
        for channel in ('original', 'development'): r[channel][name]['selected'] = history['results'][name]['selected']
        replace(out/'epoch_001.json', history); r['history_sha256'] = m.digest([history])
    elif damage in ('checkpoint_identity', 'recipe', 'original_tensor'):
        channel = 'original' if damage == 'original_tensor' else 'development'
        path = out/channel/(name+'.pt'); payload = torch.load(path, weights_only=True)
        if damage == 'checkpoint_identity': payload['identity']['source']['heads_sha256'] = '0'*64
        elif damage == 'recipe': payload['frozen_chosen']['recipe']['engine'] = 'JOINT_FULL'
        else: next(iter(payload['models']['main'].values())).add_(.001)
        torch.save(payload, path); r[channel][name]['checkpoint_sha256'] = m.sha(path)
    elif damage == 'optimizer':
        history['optimizer_steps'][name] = {}; replace(out/'epoch_001.json', history)
        r['history_sha256'] = m.digest([history])
    elif damage == 'source_copy': replace(out/'source_evidence'/'frozen_policies.json', [])
    elif damage == 'missing_epoch': (out/'validation_epoch_001.json').unlink()
    else: r['extra_smoke_updates'] = 1
    replace(out/'receipt.json', r)
    f, t, source, ref = joint('A')
    with pytest.raises((ValueError, FileNotFoundError)):
        m.verify_case(f, t, source, job, old, out, COMMIT, ref, spec)


def test_changed_training_backend_stops_before_new_output(joint, tmp_path):
    job = next(j for j in m.jobs() if j['setting'] == 'A'); old, spec = source_case(joint, tmp_path, job, 1)
    receipt = m.read(old/'receipt.json'); receipt['environment']['torch'] = 'different'
    replace(old/'receipt.json', receipt); f, t, source, ref = joint('A'); out = tmp_path/'new'
    with pytest.raises(ValueError, match='backend'):
        m.refit_case(f, t, source, job, old, out, COMMIT, 'cpu', ref, spec)
    assert not out.exists()


@pytest.fixture
def locked_source(tmp_path, monkeypatch):
    root = tmp_path/'source'; root.mkdir()
    m.write(root/'verification.json', dict(task=m.s6.TASK, commit=m.SOURCE_COMMIT, content_status='PASS',
                                         model_training_runs=132, model_epochs=5280))
    (root/'payload').write_bytes(b'locked')
    raw = ''.join(m.sha(p)+'  '+p.name+'\n' for p in sorted(root.iterdir())).encode()
    (root/'checksums.sha256').write_bytes(raw)
    monkeypatch.setattr(m, 'SOURCE_FILE_COUNT', 2)
    monkeypatch.setattr(m, 'SOURCE_MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())
    return root, raw


def test_source_archive_fallback_checks_pinned_archive_and_manifest(locked_source, monkeypatch):
    root, raw = locked_source
    assert m.verify_source(root)['files_checked'] == 2
    archive = root.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x') as z: z.writestr('checksums.sha256', raw)
    monkeypatch.setattr(m, 'SOURCE_ZIP_SHA256', m.sha(archive))
    (root/'checksums.sha256').unlink()
    assert m.verify_source(root)['files_checked'] == 2
    with archive.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='archive'): m.verify_source(root)


@pytest.mark.parametrize('damage', ('missing', 'changed', 'extra', 'rewritten_manifest', 'wrong_commit'))
def test_source_lock_is_external_to_self_consistent_manifest(damage, locked_source, monkeypatch):
    root, raw = locked_source
    if damage == 'missing': (root/'payload').unlink()
    elif damage == 'extra': (root/'unlisted').write_bytes(b'bad')
    elif damage == 'wrong_commit':
        data = m.read(root/'verification.json'); data['commit'] = 'b'*40; replace(root/'verification.json', data)
        raw = ''.join(m.sha(root/n)+'  '+n+'\n' for n in ('payload', 'verification.json')).encode()
        (root/'checksums.sha256').write_bytes(raw)
        monkeypatch.setattr(m, 'SOURCE_MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())
    else:
        (root/'payload').write_bytes(b'bad')
        if damage == 'rewritten_manifest':
            (root/'checksums.sha256').write_bytes(raw.replace(hashlib.sha256(b'locked').hexdigest().encode(), m.sha(root/'payload').encode()))
    with pytest.raises(ValueError): m.verify_source(root)


@pytest.mark.parametrize('name', ('../escape', '/absolute', 'x/../escape', 'a\\b', 'C:/escape', './alias', 'a//b', ''))
def test_manifest_rejects_unsafe_paths(name, monkeypatch):
    monkeypatch.setattr(m, 'SOURCE_FILE_COUNT', 1)
    with pytest.raises(ValueError): m.parse_manifest(('a'*64+'  '+name+'\n').encode())


def test_manifest_rejects_case_aliases(monkeypatch):
    monkeypatch.setattr(m, 'SOURCE_FILE_COUNT', 2)
    with pytest.raises(ValueError): m.parse_manifest(('a'*64+'  A\n'+'b'*64+'  a\n').encode())


def comparison_fixture(tmp_path):
    receipts = []; references = {}
    def metrics(score): return dict(macro_rmse=score, endpoints={'endpoint': {'rmse': score}})
    for job in m.jobs():
        old = tmp_path/job['id']; old.mkdir()
        m.write(old/'frozen_selection.json', dict(job=job, selected={n: dict(inner_macro_rmse=1. if job['lr'] == .001 else 2.) for n in m.method.METHODS}))
        values = {n: dict(epoch=4, selected=metrics(.8), paired_control=metrics(.9)) for n in m.method.METHODS}
        receipts.append(dict(job=job, development=deepcopy(values), original=deepcopy(values)))
        references[job['setting']+'_strong'] = dict(selected=metrics(1.1))
    return receipts, references


def test_two_selection_channels_and_target_control_have_deterministic_ties(tmp_path):
    receipts, references = comparison_fixture(tmp_path)
    for r in receipts:
        if r['job']['lr'] == .001:
            for v in r['development'].values(): v['epoch'] = 1
    result = m.compare(receipts, tmp_path, references)
    assert all(v['method'] == 'TARGET_FULL' and v['job']['lr'] == .0001 for v in result['target_best_dev'].values())
    assert all(r['development_job']['lr'] == .0001 and r['original_job']['lr'] == .001 for r in result['comparisons'])
    assert result['all_scene_candidate_signals'] == []  # Equal target-only scores are not transfer gains.
    assert result['unified_superiority_confirmed'] is False


@pytest.mark.parametrize('blocked', ('none', 'one_scene', 'paired', 'target', 'tox066'))
def test_promotion_requires_same_candidate_and_all_strong_controls(blocked, tmp_path):
    receipts, references = comparison_fixture(tmp_path); name = m.method.CANDIDATES[0]
    for r in receipts:
        r['development'][name]['selected']['macro_rmse'] = .7
        if blocked == 'one_scene' and r['job']['setting'] == 'B': r['development'][name]['selected']['macro_rmse'] = 1.2
        if blocked == 'paired': r['development'][name]['paired_control']['macro_rmse'] = .7
        if blocked == 'target': r['development']['TARGET_LAST']['selected']['macro_rmse'] = .6
        if blocked == 'tox066' and r['job']['setting'] == 'ToxAcute':
            for v in r['development'].values(): v['selected']['macro_rmse'] = 1.07; v['paired_control']['macro_rmse'] = 1.08
            r['development'][name]['selected']['macro_rmse'] = 1.04
    result = m.compare(receipts, tmp_path, references)
    assert result['all_scene_candidate_signals'] == ([name] if blocked == 'none' else [])


def test_parent_verification_uses_six_fresh_cpu_children_without_opening_datastores(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'check_launch', lambda *a: {})
    monkeypatch.setattr(m, 'gate_check', lambda *a: None)
    monkeypatch.setattr(m, 'verify_source', lambda *a, **kw: {})
    monkeypatch.setattr(m.s3, 'references', lambda *a: {})
    monkeypatch.setattr(m, 'compare', lambda *a: {})
    monkeypatch.setattr(m.s3, 'context', lambda *a: pytest.fail('parent opened LMDB'))
    (tmp_path/'source_manifest.sha256').write_bytes(b'locked'); visited = []
    def call(argv, prefix, env):
        assert env['CUDA_VISIBLE_DEVICES'] == ''
        name = argv[argv.index('--job')+1]; visited.append(name)
        m.write(tmp_path/name/'verification.json', dict(task=m.TASK, commit=COMMIT, job=name, pid=len(visited), device='cpu',
                content_status='PASS', selected_raw_prediction_replayed=True, new_optimizer_updates=0,
                receipt_sha256=m.sha(tmp_path/name/'receipt.json')))
    monkeypatch.setattr(m.s5, 'call_worker', call)
    for job in m.jobs():
        name = job['id']; (tmp_path/name).mkdir(); m.write(tmp_path/(name+'.command.json'), dict(exit_code=0))
        m.write(tmp_path/name/'receipt.json', dict(total_updates=40*11*{'ToxAcute': 6, 'A': 15, 'B': 9}[job['setting']]))
    result = m.verify(tmp_path, tmp_path/'old', COMMIT, tmp_path/'split', tmp_path/'source')
    assert len(visited) == 6 and result['total_updates'] == 26400 and result['model_epochs'] == 2640
