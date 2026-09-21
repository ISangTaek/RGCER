"""Synthetic-only tests of fixed-best inference, permissions and no early test access."""
from copy import deepcopy
import csv
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import human6 as h
from scripts import run_human6_test as x
from tests.test_human6 import views, trainer


def rows(view):
    return [dict(task=t,split=view.split,sample_id=sid,canonical=view.canonical[i],group=view.groups[i],
                 label=float(view.labels[i,j]),prediction=float(view.labels[i,j]+.2))
            for j,t in enumerate(h.TASKS) for i,sid in enumerate(view.sample_ids)]


@pytest.mark.parametrize('kind',['prediction','label','duplicate','missing'])
def test_replay_rejects_bad(kind):
    good=rows(views()['validation']); bad=deepcopy(good)
    if kind in ('prediction','label'): bad[0][kind]+=.01
    elif kind=='duplicate': bad.append(bad[0])
    else: bad.pop()
    with pytest.raises(ValueError): x.compare_replay(bad,good)
    x.compare_replay(list(reversed(good)),good)


def test_inference_only_matches_existing_evaluate(tmp_path):
    torch.set_num_threads(1)
    t=trainer('FROZEN')
    original,_=t.evaluate()
    predicted=x.predict(t.model,t.views['validation'],t.scaler,'cpu')
    x.compare_replay(predicted,original)
    state=t.model.state_dict(); path=tmp_path/'best.pt'
    torch.save(dict(identity=t.identity,epoch=2,model_state=state,model_state_sha=h.state_digest(state)),path)
    r=dict(identity=t.identity,best_epoch=2,files={'best.pt':dict(size_bytes=path.stat().st_size,sha256=h.digest(path))})
    assert h.state_digest(x.load_best(path,r))==h.state_digest(state)
    with pytest.raises(ValueError): x.load_best(path,dict(r,best_epoch=3))
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError): x.load_best(path,r)


def test_test_metrics_truth_and_old_permissions():
    v=SimpleNamespace(**vars(views()['validation'])); v.split='test'
    rr=rows(v); m=x.test_metrics(rr,v)
    assert m['macro_rmse']==pytest.approx(.2) and m['known_route_macro_rmse']==pytest.approx(.2)
    with pytest.raises(ValueError): x.test_metrics([dict(r,label=r['label']+1) for r in rr],v)
    with pytest.raises(ValueError): x.test_metrics([dict(r,prediction=float('nan')) for r in rr],v)
    with pytest.raises(ValueError): h.View(**vars(v))


def test_heldout_reader_parses_only_test_human6(tmp_path,monkeypatch):
    fields=['cid','smiles']+list(h.TASKS)+[f'other{i}' for i in range(109)]+[str(i) for i in range(1024)]
    # column schema separately covered by existing tests; numeric boundary is focus.
    monkeypatch.setattr(h,'task_columns',lambda names:None)
    cp=tmp_path/'data.csv'; sp=tmp_path/'split.json'; refs=[]
    with cp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(51):
            s='test' if i<48 else ('train','validation','calibration')[i-48]
            r=dict.fromkeys(fields,'DO_NOT_PARSE'); r.update(cid=str(i),smiles=f'C{i}')
            if s=='test': r.update(dict.fromkeys(h.TASKS,'2'))
            w.writerow(r)
            refs.append(dict(row_index=i,sample_id=f'row_{i}',raw_smiles=f'C{i}',canonical_smiles=f'C{i}',split_group=f'g{i%35}' if s=='test' else f'other{i}',split=s))
    sp.write_text(json.dumps(dict(records=refs,split_algorithm='constrained_scaffold_v3',seed=42,source_csv_sha256=h.digest(cp))))
    p=dict(data=dict(csv_sha256=h.digest(cp),split_sha256=h.digest(sp)),counts={'test':dict.fromkeys(h.TASKS,48)})
    calls=[]; orig=h.label
    def guarded(v): calls.append(v); return orig(v)
    monkeypatch.setattr(h,'label',guarded)
    v=x.load_test(cp,sp,p)
    assert len(calls)==48*6 and set(calls)=={'2'} and not v.labels.flags.writeable
    assert len(v.sample_ids)==48 and len(set(v.groups))==35


def test_wrong_archive_rejected(tmp_path):
    p=tmp_path/'wrong.zip'; p.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='accepted'): x.accepted_runs(p,h.policy())


@pytest.mark.parametrize('bad_replay',[True,False])
def test_all_replays_precede_test(tmp_path,monkeypatch,bad_replay):
    import subprocess
    from scripts import run_human6 as dispatch
    vv=views(); expected=rows(vv['validation']); scaler=h.TaskScaler.fit(vv['train'].labels,h.TASKS,allow_empty=False).to_dict()
    rr=[dict(identity=dict(method=m,seed=s,architecture={},scaler=scaler),files={'best.pt':{}},
             best_epoch=2,run_directory=f'/formal/{m}_s{s}',accepted_validation=expected)
        for m in h.METHODS for s in range(42,47)]
    monkeypatch.setattr(dispatch,'gpu_free',lambda g:None)
    monkeypatch.setattr(x,'accepted_runs',lambda *a:rr)
    monkeypatch.setattr(h,'load_views',lambda *a:vv)
    monkeypatch.setattr(x,'load_best',lambda *a:{})
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'a'*40)
    class Model:
        def load_state_dict(self,*a,**k): pass
        def to(self,*a): return self
        def cpu(self): return self
    monkeypatch.setattr(h,'model_from_source',lambda *a:Model())
    count=[0,0]
    tv=SimpleNamespace(**vars(vv['validation'])); tv.split='test'; tv.groups=('test1','test2')
    def predict(model,v,*a):
        if v.split=='validation':
            count[0]+=1
            return [dict(r,prediction=100) for r in expected] if bad_replay and count[0]==2 else expected
        return rows(v)
    def load_test(*a):
        assert count[0]==15; count[1]+=1; return tv
    monkeypatch.setattr(x,'predict',predict); monkeypatch.setattr(x,'load_test',load_test)
    args=SimpleNamespace(gpu=0,accepted_zip='unused',output=tmp_path/'out',csv=None,split_manifest='unused',formal_root=None)
    if bad_replay:
        with pytest.raises(ValueError,match='replay'): x.run(args)
        assert count==[2,0] and not (args.output/'summary.json').exists()
    else:
        x.run(args); assert count==[15,1]
        report=json.loads((args.output/'summary.json').read_text()); assert report['training_updates']==0 and len(report['runs'])==15
        with pytest.raises(ValueError,match='output exists'): x.run(args)
        ev=tmp_path/'evidence'; ev.mkdir()
        with pytest.raises(FileNotFoundError): x.package(args.output,ev)
        for n in ('wsl_commands.log','wsl_tests.xml','server_commands.log','server_tests.xml'): (ev/n).write_text('fixture')
        x.package(args.output,ev)
        assert (tmp_path/'out_review.zip.sha256').exists()
