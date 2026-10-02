from copy import deepcopy
from pathlib import Path
import threading
import time
import json
import pytest
import torch

import v9_s5_screen as mod
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory


def corrupt(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.mark.parametrize('job', mod.jobs(), ids=lambda j: j['id'])
def test_all_33_paths_train_select_and_raw_replay(job, tmp_path, joint):
    spec = dict(mod.SPEC, max_epochs=2); f, t, a, ref = joint(job['setting'])
    r = mod.train_one(f, t, a, job, tmp_path, 'a'*40, 'cpu', ref, spec)
    f, t, a, ref = joint(job['setting'])
    checked = mod.verify_job(f, t, a, job, tmp_path, 'a'*40, ref, spec)
    assert checked == r and len(r['history']) == 2 and r['checkpoint_validation_replay'] is True


@pytest.mark.parametrize('pair', mod.SMOKE)
def test_every_method_real_smoke_code_path(pair, tmp_path, joint):
    setting, method = pair; f, t, a, _ = joint(setting)
    mod.smoke(f, t, a, setting, method, tmp_path, 'a'*40, 'cpu')
    receipt = mod.read(tmp_path/'receipt.json')
    assert receipt['updates'] == 2 and receipt['content_status'] == 'PASS'


@pytest.mark.parametrize('damage', ['metric', 'steps', 'checkpoint', 'source', 'private_route', 'truncated'])
def test_independent_verifier_rejects_corruption(damage, tmp_path, joint):
    job = mod.jobs()[0]; spec = dict(mod.SPEC, max_epochs=2)
    f, t, a, ref = joint(job['setting']); r = mod.train_one(f, t, a, job, tmp_path, 'a'*40, 'cpu', ref, spec)
    if damage == 'metric':
        values = mod.read(tmp_path/'validation_epoch_001.json'); values[0]['prediction'] += 1
        corrupt(tmp_path/'validation_epoch_001.json', values)
    elif damage == 'steps':
        row = r['history'][0]; name = next(iter(row['optimizer_steps'])); row['optimizer_steps'][name] += 1
        corrupt(tmp_path/'epoch_001.json', row); corrupt(tmp_path/'receipt.json', r)
    elif damage == 'checkpoint':
        payload = torch.load(tmp_path/'best.pt', weights_only=True)
        name = next(iter(payload['model_state'])); payload['model_state'][name].fill_(float('nan'))
        torch.save(payload, tmp_path/'best.pt'); r['checkpoint_sha256'] = mod.sha(tmp_path/'best.pt'); corrupt(tmp_path/'receipt.json', r)
    elif damage == 'source':
        rows = mod.read(tmp_path/'source_train_observations.json'); rows[0]['label'] += 1
        corrupt(tmp_path/'source_train_observations.json', rows)
    elif damage == 'private_route':
        r['history'][0]['losses'][0]['rule'] = 'OTHER'
        corrupt(tmp_path/'epoch_001.json', r['history'][0]); corrupt(tmp_path/'receipt.json', r)
    else:
        r['history'] = r['history'][:1]; corrupt(tmp_path/'receipt.json', r)
    f, t, a, ref = joint(job['setting'])
    with pytest.raises(ValueError):
        mod.verify_job(f, t, a, job, tmp_path, 'a'*40, ref, spec)


def test_dispatch_one_per_gpu_and_completes_all():
    active = set(); seen = []; lock = threading.Lock(); peak = []
    def execute(item, gpu):
        with lock:
            assert gpu not in active; active.add(gpu); peak.append(len(active))
        time.sleep(.01)
        with lock:
            active.remove(gpu); seen.append(item)
    mod.dispatch(list(range(11)), [0, 1, 2, 3], execute)
    assert sorted(seen) == list(range(11)) and max(peak) <= 4 and not active


def test_dispatch_failure_stops_new_jobs_and_preserves_inflight():
    started = []; finished = []
    def execute(item, gpu):
        started.append(item)
        if item == 0:
            raise ValueError('fixture failure')
        time.sleep(.02); finished.append(item)
    with pytest.raises(RuntimeError, match='in-flight'):
        mod.dispatch(list(range(20)), [0, 1], execute)
    assert set(started) <= {0, 1} and 0 in started
    assert finished == [1]


def test_compare_never_combines_scene_specific_winners():
    reference = {}
    for s in mod.s3.s2.c0.SETTINGS:
        for m in mod.s3.s2.METHODS:
            reference[f'{s}_{m}_s42'] = dict(selected=dict(macro_rmse=1., endpoints={'x': dict(rmse=1.)}))
    results = []
    for j in mod.jobs():
        score = .9 if j['setting'] == 'ToxAcute' else 1.1
        results.append(dict(job=j, best_epoch=40, selected=dict(macro_rmse=score, endpoints={'x': dict(rmse=score)})))
    assert mod.compare(results, reference)['all_scene_candidate_signals'] == []
    for r in results:
        if r['job']['method'] == 'TPRS':
            r['selected'] = dict(macro_rmse=.8, endpoints={'x': dict(rmse=.8)})
    report = mod.compare(results, reference)
    assert report['all_scene_candidate_signals'] == report['private_architecture_candidate_signals'] == ['TPRS']
    assert report['unified_superiority_confirmed'] is False


def test_matrix_has_ten_methods_excluding_controls_and_no_lr_duplicates():
    assert len(mod.method.CANDIDATES) == len(set(mod.method.CANDIDATES)) == 10
    assert len(mod.jobs()) == 33 and len(mod.SMOKE) == 11
    assert mod.SPEC['max_epochs'] == 40 and mod.SPEC['warmup_epochs'] == 0
    assert 'TPO_FT' not in mod.method.CANDIDATES
    assert len({j['encoder_lr'] for j in mod.jobs()}) == 1


def test_cpu_verification_argv_has_no_train_or_gpu_override(tmp_path):
    argv = mod.worker_command('verify', mod.jobs()[0]['id'], tmp_path, 'a'*40, Path('split'), Path('lock'))
    assert argv[argv.index('--mode')+1] == 'verify' and '--gpus' not in argv
    assert '_worker' in argv


def test_self_consistent_wrong_predictions_rejected_by_raw_checkpoint_replay(tmp_path, joint):
    job = mod.jobs()[0]; spec = dict(mod.SPEC, max_epochs=1)
    f, t, a, ref = joint(job['setting']); r = mod.train_one(f, t, a, job, tmp_path, 'a'*40, 'cpu', ref, spec)
    rows = mod.read(tmp_path/'selected_validation.json')
    for row in rows:
        row['prediction'] += .1
    score = mod.s3.s2.metrics(rows, mod.read(tmp_path/'validation_observations.json'))
    r['history'][0]['validation'] = score; r['selected'] = score
    for name, value in [('selected_validation', rows), ('validation_epoch_001', rows), ('epoch_001', r['history'][0]), ('receipt', r)]:
        corrupt(tmp_path/(name+'.json'), value)
    f, t, a, ref = joint(job['setting'])
    with pytest.raises(ValueError, match='raw-data prediction replay'):
        mod.verify_job(f, t, a, job, tmp_path, 'a'*40, ref, spec)


def test_parent_verifier_never_opens_graph_context(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'check_launch', lambda *args: {})
    monkeypatch.setattr(mod, 'gate_check', lambda *args: None)
    monkeypatch.setattr(mod.s3, 'context', lambda *args: pytest.fail('parent opened graph context'))
    monkeypatch.setattr(mod.s3, 'references', lambda *args: {})
    monkeypatch.setattr(mod, 'compare', lambda *args: {})
    for setting, kind in mod.SMOKE:
        name = 'smoke_'+setting+'_'+kind; (tmp_path/name).mkdir()
        mod.write(tmp_path/(name+'.command.json'), dict(exit_code=0))
        mod.write(tmp_path/name/'receipt.json', dict(task=mod.TASK, commit='a'*40, setting=setting, method=kind,
                  updates=2, content_status='PASS', private_changed=kind == 'TPRS', rows=[{}, {}]))
    for j in mod.jobs():
        (tmp_path/j['id']).mkdir(); mod.write(tmp_path/(j['id']+'.command.json'), dict(exit_code=0))
        mod.write(tmp_path/j['id']/'receipt.json', dict(job=j, updates={'ToxAcute': 240, 'A': 600, 'B': 360}[j['setting']]))
    calls = []
    def child(argv, prefix, env):
        assert env['CUDA_VISIBLE_DEVICES'] == '' and argv[argv.index('--mode')+1] == 'verify'
        job = next(j for j in mod.jobs() if j['id'] == argv[argv.index('--job')+1]); calls.append(job['id'])
        mod.write(tmp_path/job['id']/'verification.json', dict(task=mod.TASK, commit='a'*40, job=job,
                  content_status='PASS', raw_prediction_replayed=True, device='cpu', new_optimizer_updates=0, pid=len(calls),
                  receipt_sha256=mod.sha(tmp_path/job['id']/'receipt.json')))
    monkeypatch.setattr(mod, 'call_worker', child)
    checked = mod.verify(tmp_path, 'a'*40, Path('split'), Path('lock'))
    assert len(calls) == len(set(calls)) == 33 and checked['total_updates'] == 13200


def test_claim_consumed_or_mismatched_cannot_relaunch(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path); monkeypatch.setattr(mod, 'REGISTRY', 'registry')
    (tmp_path/'registry').mkdir(); output = tmp_path/'output'; output.mkdir()
    claim = mod.claim_value(output, 'a'*40)
    mod.write(tmp_path/'registry'/'attempt.json', claim)
    mod.write(output/'launch.json', dict(task=mod.TASK, commit='a'*40, claim=claim))
    mod.check_launch(output, 'a'*40)
    with pytest.raises(ValueError, match='one-shot'):
        mod.check_launch(output, 'b'*40)
