"""Small synthetic tests. No private fixtures, numeric Human6 data or GPU."""
from dataclasses import replace
import csv
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

import human6 as h
from dataset115_contract import PRIMARY, digest


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def views():
    a = h.View('train', ('dataset115:row_0','dataset115:row_1'), ('CC','CCC'),
               ('CC','CCC'), ('CC','CCC'), np.arange(12).reshape(2,6), 'a'*64)
    b = h.View('validation', ('dataset115:row_2','dataset115:row_3'), ('CO','CN'),
               ('CO','CN'), ('CO','CN'), np.arange(12).reshape(2,6)+.5, 'a'*64)
    return dict(train=a, validation=b)


def trainer(method):
    from tests.test_dataset115_adapter import args
    from dataset115_model import build_human5_model
    a = vars(args())
    source = build_human5_model(SimpleNamespace(**a), method='B0', seed=123).encoder.state_dict()
    return h.Trainer(views(), source, a, method=method, seed=42, device='cpu', policy_sha='a'*64, source_sha='b'*64)


def test_schedule_and_approved_budget():
    p = h.policy()
    counts = p['counts']['train']
    sched = h.schedule(counts, 42, 0)
    assert len(sched) == 9 and sum(len(ids) for _,ids in sched) == 183
    assert sched == h.schedule(counts,42,0) and sched != h.schedule(counts,43,0)
    for t in h.TASKS:
        assert sorted(i for task, ids in sched if task==t for i in ids) == list(range(counts[t]))
    assert len(p['runs']) == 15 and sum(r['optimizer_updates'] for r in p['runs']) == 5400
    assert p['budget']['total_updates_cap'] == 5562 and not p['test_authorized']
    assert all(r['backbone_active_updates'] == {'FROZEN':0,'B1_low':360,'HF_low':315}[r['method']] for r in p['runs'])


def test_view_rejects_old_tasks_and_holdout():
    for split in ('test','calibration'):
        with pytest.raises(ValueError, match='train/validation'):
            replace(views()['train'], split=split)
    with pytest.raises(ValueError, match='train/validation'):
        replace(views()['train'], tasks=PRIMARY)


def test_metadata_and_numeric_scope(tmp_path, monkeypatch):
    tasks = list(h.TASKS)+['human_oral_TDLo','man_oral_TDLo','women_oral_TDLo','child_oral_TDLo','human_oral_LDLo']
    tasks += [f'animal{i}_oral_LD50' for i in range(104)]
    fields = ['cid','smiles']+tasks+[str(i) for i in range(1024)]
    cp, sp = tmp_path/'input.csv', tmp_path/'split.json'
    refs = []
    with cp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, s in enumerate(('train','validation','calibration','test')):
            row = dict.fromkeys(fields, 'DO_NOT_PARSE')
            row.update(cid=str(i), smiles=f'C{i}')
            if s in ('train','validation'):
                row.update({t:'0' for t in h.TASKS})
            w.writerow(row)
            refs.append(dict(row_index=i,sample_id=f'row_{i}',raw_smiles=f'C{i}',canonical_smiles=f'C{i}',split_group=f'g{i}',split=s))
    sp.write_text(json.dumps(dict(records=refs,split_algorithm='constrained_scaffold_v3',seed=42,source_csv_sha256=digest(cp))))
    p = dict(data=dict(csv_sha256=digest(cp),split_sha256=digest(sp)),
             counts={s:dict.fromkeys(h.TASKS,1) for s in ('train','validation')})
    called = []
    original = h.label
    def guarded(v):
        called.append(v)
        return original(v)
    monkeypatch.setattr(h,'label',guarded)
    got = h.load_views(cp,sp,p)
    assert len(called) == 12 and set(called) == {'0'}
    assert got['train'].labels.shape == (1,6) and not got['train'].labels.flags.writeable
    assert PRIMARY == ('child_oral_TDLo','human_oral_LDLo','human_oral_TDLo','man_oral_TDLo','women_oral_TDLo')


def test_metrics_missing_invalid_and_true_population():
    v = views()['validation']
    rows = [dict(task=t,split='validation',sample_id=sid,canonical=v.canonical[i],group=v.groups[i],
                 label=float(v.labels[i,j]),prediction=float(v.labels[i,j]+j))
            for j,t in enumerate(h.TASKS) for i,sid in enumerate(v.sample_ids)]
    assert h.metrics(rows,v)['macro_rmse'] == 2.5
    for bad in (rows[:-1], rows+rows[:1], [dict(r,prediction=float('nan')) for r in rows],
                [dict(r,label=r['label']+1) for r in rows]):
        with pytest.raises(ValueError): h.metrics(bad,v)


@pytest.mark.parametrize('method', h.METHODS)
def test_freeze_best_and_exact_resume(tmp_path, method):
    full = trainer(method); full_result = full.run(tmp_path/'full',epochs=6)
    first = trainer(method); first.run(tmp_path/'first',epochs=5)
    resumed = trainer(method)
    resumed.run(tmp_path/'resumed',epochs=6,resume=tmp_path/'first/last.pt',resume_sha=digest(tmp_path/'first/last.pt'))
    a = torch.load(tmp_path/'full/last.pt', weights_only=True)
    b = torch.load(tmp_path/'resumed/last.pt', weights_only=True)
    assert a['history'] == b['history'] and a['model_state_sha'] == b['model_state_sha']
    assert full_result['updates'] == 36 and full_result['validation_replayed']
    assert full_result['best_epoch'] == min(range(6),key=lambda e:a['history'][e]['validation']['macro_rmse'])
    if method == 'HF_low':
        assert all(x['optimization']['frozen'] for x in a['history'][:5])
        assert not a['history'][5]['optimization']['frozen']
        assert a['history'][4]['optimization']['backbone_optimizer_parameters'] == 0
        assert a['history'][5]['optimization']['backbone_optimizer_parameters'] > 0
    if method == 'FROZEN':
        assert h.state_digest({k[8:]:v for k,v in a['model_state'].items() if k.startswith('encoder.')}) == full.initial_encoder
    assert sorted(p.name for p in (tmp_path/'full').glob('*.pt')) == ['best.pt','last.pt']
    with pytest.raises(FileExistsError): trainer(method).run(tmp_path/'full',epochs=6)


def test_paired_initialization_and_reject_bad_resume(tmp_path):
    ts = [trainer(m) for m in h.METHODS]
    assert len({t.initial_heads for t in ts}) == len({t.initial_encoder for t in ts}) == 1
    ts[2].run(tmp_path/'first',epochs=5)
    p = torch.load(tmp_path/'first/last.pt', weights_only=True)
    p['optimizer']['param_groups'][0]['lr'] = .1
    torch.save(p,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='optimizer group'):
        trainer('HF_low').run(tmp_path/'badresume',epochs=6,resume=tmp_path/'bad.pt',resume_sha=digest(tmp_path/'bad.pt'))
