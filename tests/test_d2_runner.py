from copy import deepcopy
import json

import pytest
import torch

import d2_control as core
import d2_runner as mod
from tests.test_d2_control import data_factory, one_thread, small_spec


@pytest.mark.parametrize('job', core.jobs(), ids=lambda j: j['id'])
def test_all_27_graph_trajectories_and_independent_replay(job, tmp_path, data_factory):
    data = data_factory(job['condition'], job['setting'])
    out = tmp_path/'run'
    result = mod.train_one(data, job, out, 'a'*40, 'cpu', small_spec())
    fresh = data_factory(job['condition'], job['setting'])
    checked = mod.verify_one(fresh, job, out, 'a'*40, 'cpu', small_spec())
    assert checked['status'] == 'CONTENT_PASS' and checked['selected'] == result['selected']
    assert not checked['scientific_acceptance'] and not checked['test_accessed']
    assert result['completed_updates'] == 8 and result['source_passes_completed'] == (2 if data.source else 0)


@pytest.mark.parametrize('damage', ['label', 'best_step', 'schedule', 'coverage', 'checkpoint_identity',
                                  'encoder', 'optimizer_step', 'optimizer_config', 'optimizer_nan', 'truncated', 'source_pass', 'smoke'])
def test_independent_verifier_rejects_self_consistent_corruption(damage, tmp_path, data_factory):
    job = next(j for j in core.jobs() if j['condition'] == ('FJ' if damage == 'encoder' else 'PJ'))
    root = tmp_path/'run'
    data = data_factory(job['condition'], job['setting'])
    mod.train_one(data, job, root, 'a'*40, 'cpu', small_spec())
    r = mod.read(root/'training.json')
    h = mod.read(root/'history.json')
    best = r['selected']['step']
    if damage == 'label':
        p = root/r['selected']['predictions']
        rows = mod.read(p)
        rows[0]['label'] += 1
        p.write_text(json.dumps(rows))
        for row in h:
            if row['step'] == best:
                row['predictions_sha256'] = mod.sha(p)
                (root/f'validation_{best:07d}.receipt.json').write_text(json.dumps(row))
    elif damage == 'best_step':
        r['selected'] = h[-1] if r['selected'] != h[-1] else h[0]
    elif damage == 'schedule':
        h[0]['schedule_sha256'] = '0'*64
        (root/f'validation_{h[0]["step"]:07d}.receipt.json').write_text(json.dumps(h[0]))
    elif damage == 'coverage':
        r['exposure']['target'][data.target_tasks[0]] += 1
    elif damage == 'truncated':
        h.pop()
    elif damage == 'source_pass':
        p = root/'source_pass_history.json'
        rows = mod.read(p)
        rows[0]['step'] += 1
        p.write_text(json.dumps(rows))
    elif damage == 'smoke':
        p = root/'first_two_updates.json'
        rows = mod.read(p)
        rows['updates'][0]['gradient_l1']['source'] = 0
        p.write_text(json.dumps(rows))
    else:
        p = root/'best.pt'
        q = torch.load(p, weights_only=True)
        if damage == 'checkpoint_identity':
            q['identity']['data']['condition'] = 'RT'
        elif damage == 'encoder':
            q['model_state'][next(k for k in q['model_state'] if k.startswith('encoder.'))].add_(1)
        elif damage == 'optimizer_config':
            q['optimizer_state']['param_groups'][0]['eps'] *= 2
        elif damage == 'optimizer_nan':
            next(iter(q['optimizer_state']['state'].values()))['exp_avg'].fill_(float('nan'))
        else:
            next(iter(q['optimizer_state']['state'].values()))['step'].add_(1)
        torch.save(q, p)
        r['best_sha256'] = mod.sha(p)
    (root/'training.json').write_text(json.dumps(r))
    (root/'history.json').write_text(json.dumps(h))
    with pytest.raises(ValueError):
        mod.verify_one(data_factory(job['condition'], job['setting']), job, root, 'a'*40, 'cpu', small_spec())


def test_target_only_never_fetches_source_labels_during_run(data_factory, tmp_path):
    job = next(j for j in core.jobs() if j['condition'] == 'RT')
    data = data_factory('RT', job['setting'])
    original = data.batch
    roles = []
    def batch(role, *args):
        roles.append(role)
        assert role == 'target'
        return original(role, *args)
    data.batch = batch
    mod.train_one(data, job, tmp_path/'run', 'a'*40, 'cpu', small_spec())
    assert set(roles) == {'target'}


def test_run_output_cannot_overwrite(data_factory, tmp_path):
    job = core.jobs()[0]
    with pytest.raises(FileExistsError):
        mod.train_one(data_factory('FJ', job['setting']), job, tmp_path, 'a'*40, 'cpu', small_spec())
