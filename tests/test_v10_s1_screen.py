from copy import deepcopy
from pathlib import Path
import hashlib
import json
import zipfile

import pytest
import torch

import v10_s1_screen as s
from tests.test_v10_context_models import single_thread


@pytest.fixture
def value():
    torch.manual_seed(123)
    targets=['human_oral_TDLo','man_oral_TDLo']; sources=['rat_oral_LD50','mouse_oral_LD50']
    rows=[]; scalers={}
    for task in targets+sources:
        train=[]
        for split,n in (('train',12),('validation',3 if task in targets else 0)):
            for i in range(n):
                row=dict(task=task,split=split,sample_id=split+str(i),canonical=split+str(i),group=split+str(i//2),label=float(i%5)+.2*i)
                rows.append(row)
                if split=='train': train.append(row['label'])
        labels=torch.tensor(train,dtype=torch.float64)
        scalers[task]=dict(mean=float(labels.mean()),std=float(labels.std(unbiased=False)),count=len(train))
    # Canonical order is all target train, all target validation, then source train.
    rows=[r for t in targets for r in rows if r['task']==t and r['split']=='train']+[
          r for t in targets for r in rows if r['task']==t and r['split']=='validation']+[
          r for t in sources for r in rows if r['task']==t]
    raw=dict(identity={'fixture':True},rows=rows,h=torch.randn(len(rows),6),functions=torch.randn(len(rows),2),
             scalers=scalers,targets=targets,sources=sources,metadata=s.cache.task_metadata(targets+sources,sources,'A'))
    return s.cache.pack(raw,'a'*40)


def spec(): return dict(s.SPEC,epochs=2,hidden=8,query_batch=4,context_cap=6)


@pytest.mark.parametrize('name',s.models.CANDIDATES+s.models.CONTROLS)
def test_complete_train_selection_checkpoint_replay_all_methods(name,value,tmp_path):
    data=s.cache.Data(value); job=next(j for j in s.jobs() if j['method']==name)
    r=s.train_case(data,job,tmp_path/'run','a'*40,spec())
    assert s.verify_case(s.cache.Data(deepcopy(value)),job,tmp_path/'run','a'*40,spec())==r
    if name not in s.models.ANALYTIC:
        assert all(r['coverage'][t]['unique_query_observations']==12 for t in data.targets)
        assert all(r['coverage'][t]['query_observations']==24 for t in data.targets)
        if name not in s.models.TARGET_ONLY: assert all(r['coverage'][t]['updates']>=2 for t in data.sources)
    assert r['test_evaluated'] is False


def test_full_target_schedule_all_source_tasks_and_no_group_leak(value):
    data=s.cache.Data(value)
    for epoch in range(3):
        schedule=s.cache.epoch_schedule(data,epoch,batch_size=16)
        targets=[i for b in schedule for i in b['target'][1]]
        assert sorted(targets)==sorted(i for t in data.targets for i in data.by[t,'train'])
        assert {t for b in schedule for t,q in b['source']}==set(data.sources)
        for b in schedule:
            for t,q in [b['target']]+b['source']:
                e=data.episode(t,q)
                assert not {r.group for r in e.context_rows}&{r.group for r in e.query_rows}


def test_validation_labels_never_enter_features_or_predictions(value):
    changed=deepcopy(value)
    for r in changed['rows']:
        if r['split']=='validation': r['label']+=10000
    for name in s.models.CANDIDATES:
        a,b=s.cache.Data(value),s.cache.Data(changed)
        model=None if name in s.models.ANALYTIC else s.make_model(a,name,spec()).eval()
        pa,_=s.predict(a,name,0,model,spec()); pb,_=s.predict(b,name,0,model,spec())
        assert [r['prediction'] for r in pa]==[r['prediction'] for r in pb]


@pytest.mark.parametrize('what',('feature','manifest','rows','commit'))
def test_cache_tampering_rejected(value,tmp_path,what):
    folder=tmp_path/'cache'; s.cache.save_cache(value,folder,'a'*40)
    if what=='feature':
        with (folder/'cache.pt').open('ab') as f: f.write(b'bad')
    else:
        m=s.read(folder/'manifest.json')
        m[{'manifest':'identity','rows':'rows_sha256','commit':'commit'}[what]]='wrong'
        (folder/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError): s.cache.load_cache(folder,'a'*40)


@pytest.mark.parametrize('what',('test','duplicate','scaler','nan','source_validation'))
def test_bad_population_fails_closed(value,what):
    v=deepcopy(value)
    if what=='test': v['rows'][0]['split']='test'
    elif what=='duplicate': v['rows'][1]=v['rows'][0]
    elif what=='scaler': v['scalers'][v['targets'][0]]['mean']+=1
    elif what=='nan': v['functions'][0,0]=float('nan')
    else: v['rows'][-1]['split']='validation'
    with pytest.raises(ValueError): s.cache.Data(v)


@pytest.mark.parametrize('what',('metric','prediction','epoch','checkpoint','context','coverage'))
def test_verifier_rejects_forged_result(value,tmp_path,what):
    job=next(j for j in s.jobs() if j['method']=='TARGET_MLP'); data=s.cache.Data(value); out=tmp_path/'run'
    r=s.train_case(data,job,out,'a'*40,spec())
    if what=='metric':
        p=out/'receipt.json'; v=s.read(p); v['selected']['macro_rmse']+=1
    elif what=='prediction':
        p=out/'selected_validation.json'; v=s.read(p); v[0]['prediction']+=1
    elif what=='epoch':
        p=out/'receipt.json'; v=s.read(p); v['best_epoch']=3
    elif what=='checkpoint':
        with (out/'best.pt').open('ab') as f: f.write(b'bad')
        p=None
    elif what=='coverage':
        p=out/'receipt.json'; v=s.read(p); v['coverage'][data.targets[0]]['unique_query_observations']=0
    else:
        p=out/'epoch_001.json'; v=s.read(p); v['trace'][0]['episodes'][0]['context']=['illegal']
    if p is not None: p.write_text(json.dumps(v))
    with pytest.raises(ValueError): s.verify_case(data,job,out,'a'*40,spec())


def test_same_architecture_must_pass_every_scene_and_all_controls(value,tmp_path,monkeypatch):
    # Independent fabricated scores exercise selection policy, not model equations.
    expected=[r for r in value['rows'] if r['split']=='validation']; refs={}
    for setting in s.SETTINGS:
        old=[dict(r,prediction=r['label']+1) for r in expected]
        refs[setting]=dict(best=dict(name='old',rows=old,score=s.s3.s2.metrics(old,expected)))
    for job in s.jobs():
        out=tmp_path/job['id']; out.mkdir()
        error=.7 if job['method']=='FCR' else 1.2
        if job['method'] in s.models.CONTROLS: error=.9
        if job['method']=='FCR' and job['setting']=='B': error=1.1
        if job['config']==1: error+=.02
        rows=[dict(r,prediction=r['label']+error) for r in expected]
        s.write(out/'selected_validation.json',rows)
        s.write(out/'receipt.json',dict(identity=dict(job=job),selected=s.s3.s2.metrics(rows,expected),best_epoch=2))
    monkeypatch.setattr(s,'grouped_difference',lambda a,b:{'test_fixture':True})
    r=s.compare(tmp_path,refs)
    assert r['development_candidates']==[] and r['decision']=='STOP_NO_UNIFIED_DEVELOPMENT_WINNER'
    assert r['settings']['A']['candidates']['FCR']['passes_point_gate'] is True
    assert r['settings']['B']['candidates']['FCR']['passes_point_gate'] is False


def test_grouped_bootstrap_preserves_pairing(value):
    expected=[r for r in value['rows'] if r['split']=='validation']
    a=[dict(r,prediction=r['label']+.5) for r in expected]; b=[dict(r,prediction=r['label']+1.) for r in expected]
    r=s.grouped_difference(a,b,draws=100)
    assert r['delta_ci95']==[-.5,-.5] and r['draws']==100


def test_metadata_does_not_invent_species_or_use_labels():
    tasks=['mammal_(species_unspecified)_oral_LD50','human_oral_TDLo']
    meta=s.cache.task_metadata(tasks,tasks[:1],'B')
    assert meta['fields'][tasks[0]]['population']=='mammal_(species_unspecified)'
    assert meta['fields'][tasks[0]]['domain']=='ToxAcute'
    assert meta['fields'][tasks[1]]['domain']=='PubChem'


def test_066_tox_and_pubchem_legacy_schema_preserve_population(value):
    expected=[r for r in value['rows'] if r['split']=='validation']
    minimal=[dict({k:r[k] for k in ('task','sample_id','split','label')},prediction=0.) for r in expected]
    extended=[dict(r,prediction=0.,lower=-1.,upper=1.) for r in expected]
    assert s.normalize_066(minimal,expected)==s.normalize_066(extended,expected)
    extended[0]['group']='wrong'
    with pytest.raises(ValueError,match='population'): s.normalize_066(extended,expected)


def test_package_integrity_and_no_overwrite(tmp_path):
    root=tmp_path/'run'; root.mkdir(); s.write(root/'launch.json',dict(commit='a'*40))
    (root/'fixture.pt').write_bytes(b'fixture')
    r=s.package(root,'a'*40)
    with zipfile.ZipFile(r['archive']) as z:
        assert hashlib.sha256(z.read('fixture.pt')).hexdigest() in z.read('checksums.sha256').decode()
    with pytest.raises(ValueError): s.package(root,'a'*40)
