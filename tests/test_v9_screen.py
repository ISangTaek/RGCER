from copy import deepcopy
from pathlib import Path
import json

import pytest
import torch

import v9_screen as mod
from tests.test_p1d_routes import setup_factory
from tests.test_p1d_tox import fixture_factory
from tests.test_v9_ssra import payload_for
from tests.test_dataset115_training import validation_view
from dataset115_adapter import GraphTaskView


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def runtime(tmp_path,monkeypatch,setup_factory):
    def make(setting):
        if setting == 'ToxAcute':
            f=fixture_factory();v=validation_view()
            f.datasets['validation']={t:GraphTaskView(v,v.tasks[0]) for t in f.datasets['train']}
            return f,f.make_trainer()
        f=setup_factory[0](setting)[0]
        return f,f.make_trainer(original=True)
    contexts={s:make(s) for s in mod.c0.SETTINGS}
    def counts(setting):
        f,t=contexts[setting];ds=mod.datasets_for(f,t,setting)['train']
        return list(ds),{k:len(v) for k,v in ds.items()}
    original=mod.observations
    def obs(f,t,s,split):
        if s != 'ToxAcute':return original(f,t,s,split)
        rows=[]
        for task,ds in f.datasets[split].items():
            for i in range(len(ds)):
                graph=ds[i]
                rows.append(dict(task=task,sample_id=str(ds.get_sample_id(i)),split=split,
                    canonical=graph.canonical_smiles,group=graph.canonical_smiles,label=float(graph.y.reshape(-1)[0])))
        return rows
    def records(f,t,s):
        return [{k:v for k,v in r.items() if k != 'label'} for r in obs(f,t,s,'train')]
    lock=tmp_path/'lock.json'
    mod.write(lock,dict(initial_encoders={s:mod.state_dict_sha256(t.model.encoder) for s,(f,t) in contexts.items()}))
    monkeypatch.setattr(mod.c2b,'LOCK',lock)
    monkeypatch.setattr(mod.c0,'tasks_and_counts',counts)
    monkeypatch.setattr(mod.c0,'contract',lambda s:dict(synthetic=True,setting=s))
    monkeypatch.setattr(mod,'observations',obs)
    monkeypatch.setattr(mod.srgt,'training_records',records)
    monkeypatch.setattr(mod.c2b,'load_stats',lambda root,setting,encoder:payload_for(encoder))
    return make


@pytest.mark.parametrize('job',mod.jobs(),ids=lambda j:j['id'])
def test_complete_epochs_validation_reload_and_independent_verification(job,tmp_path,runtime):
    spec=dict(mod.SPEC,max_epochs=2,patience=1)
    f,t=runtime(job['setting']);out=tmp_path/job['id'];out.mkdir()
    r=mod.train_one(f,t,job,out,'a'*40,'cpu',tmp_path,spec)
    f,t=runtime(job['setting'])
    checked=mod.verify_job(f,t,job,out,'a'*40,tmp_path,spec)
    assert checked['selected'] == r['selected']
    assert r['updates'] > 0 and len(r['history']) == 2
    assert r['checkpoint_validation_replay'] and not r['test_evaluated']
    assert set(r['selected']['endpoints']) == set(mod.c0.tasks_and_counts(job['setting'])[0])


def test_complete_epoch_sampling_no_duplicates_and_shared_plan():
    tasks=['one','two'];counts=dict(one=65,two=17)
    plan=mod.epoch_plan(tasks,counts,32,0)
    assert plan == mod.epoch_plan(tasks,counts,32,0)
    assert plan != mod.epoch_plan(tasks,counts,32,1)
    for t in tasks:
        assert sorted(i for b in plan if b['task'] == t for i in b['indices']) == list(range(counts[t]))


def test_early_stop_strict_first_tie_and_no_later_epochs():
    hist=[dict(epoch=i+1,validation=dict(macro_rmse=v)) for i,v in enumerate([3.,2.,2.,2.])]
    spec=dict(mod.SPEC,patience=2)
    assert mod.select(hist,spec) == (2,True)
    hist.append(dict(epoch=5,validation=dict(macro_rmse=1.)))
    with pytest.raises(ValueError,match='after early stop'):mod.select(hist,spec)


def test_validation_support_immutable_and_excludes_same_group():
    records=[dict(task='t',sample_id=str(i),canonical=c,group=g,split='train')
             for i,(c,g) in enumerate([('CC','a'),('CCC','a'),('CO','b')])]
    bank=mod.srgt.TrainSupport(records);before=deepcopy(bank.identity)
    a=bank.for_inference(['CC'],['a'])
    only=mod.srgt.TrainSupport([records[-1]])
    assert torch.equal(a,only.for_inference(['CC'],['a']))
    bank.for_inference(['CN'],['new'])
    assert bank.identity == before and 'CN' not in bank.values and len(bank.records) == 3


@pytest.mark.parametrize('change',['label','duplicate','nan','missing'])
def test_metric_population_rejects_forgery(change):
    expected=[dict(task='t',sample_id=str(i),split='validation',label=float(i),canonical='CC',group='a') for i in range(2)]
    rows=[dict(r,prediction=r['label']) for r in expected]
    if change == 'label':rows[0]['label']+=1
    if change == 'duplicate':rows[1]=deepcopy(rows[0])
    if change == 'nan':rows[0]['prediction']=float('nan')
    if change == 'missing':rows.pop()
    with pytest.raises(ValueError):mod.metrics(rows,expected)


def test_ranking_penalizes_route_failure_and_requires_all_jobs():
    results=[]
    for job in mod.jobs():
        value=1.
        if job['method'] == 'SRGT':value=.8 if job['setting'] != 'B' else 1.1
        if job['method'] == 'SSRA':value=.95
        results.append(dict(job=job,selected=dict(macro_rmse=value)))
    r=mod.rank(results)
    assert r['recommended_for_confirmation'] == 'SSRA'
    assert r['ranking'][0]['better_than_all_controls_in_all_scenes']
    assert not r['unified_superiority_confirmed']
    with pytest.raises(ValueError):mod.rank(results[:-1])


@pytest.mark.parametrize('change',['checkpoint','labels','truncation','best_epoch'])
def test_verifier_rejects_self_consistent_modified_evidence(change,tmp_path,runtime):
    job=mod.jobs()[0];spec=dict(mod.SPEC,max_epochs=2,patience=1)
    f,t=runtime(job['setting']);out=tmp_path/'job';out.mkdir()
    r=mod.train_one(f,t,job,out,'a'*40,'cpu',tmp_path,spec)
    if change == 'checkpoint':
        p=out/'best.pt';payload=torch.load(p,weights_only=True)
        k=next(k for k in payload['model_state'] if k.startswith('encoder.'))
        payload['model_state'][k].add_(1);torch.save(payload,p);r['checkpoint_sha256']=mod.sha(p)
    if change == 'labels':
        p=out/'validation_epoch_001.json';rows=mod.read(p);rows[0]['label']+=1
        p.write_text(json.dumps(rows),encoding='utf8')
    if change == 'truncation':r['history']=r['history'][:1]
    if change == 'best_epoch':r['best_epoch']=999
    (out/'receipt.json').write_text(json.dumps(r),encoding='utf8')
    f,t=runtime(job['setting'])
    with pytest.raises(ValueError):mod.verify_job(f,t,job,out,'a'*40,tmp_path,spec)


def test_gate_rejects_small_report_even_with_all_classnames(tmp_path):
    import xml.etree.ElementTree as ET
    root=ET.Element('testsuite')
    for name in mod.TESTS:ET.SubElement(root,'testcase',classname='tests.'+Path(name).stem,name='fake')
    ET.ElementTree(root).write(tmp_path/'tests.xml')
    mod.write(tmp_path/'gate.json',dict(task=mod.TASK,commit='a'*40,role='wsl',tests=list(mod.TESTS),junit_sha256=mod.sha(tmp_path/'tests.xml')))
    mod.write(tmp_path/'command.json',dict(exit_code=0))
    with pytest.raises(ValueError,match='incomplete'):mod.gate_check(tmp_path,'a'*40,'wsl')


def test_dispatch_failure_stops_matrix_and_packages_partial_facts(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import p1d4_batch
    import zipfile
    monkeypatch.setattr(mod,'REPO',tmp_path)
    monkeypatch.setattr(mod.c0,'check_code',lambda *a:None)
    monkeypatch.setattr(mod,'gate_check',lambda *a:None)
    monkeypatch.setattr(mod.c2b,'prior_gate',lambda *a:None)
    monkeypatch.setattr(p1d4_batch,'free_gpus',lambda a:a)
    monkeypatch.setattr(mod.c0,'gpu_uuid',lambda a:'GPU-synthetic')
    calls=[]
    def failed(argv,**kw):calls.append(argv);return SimpleNamespace(returncode=17)
    monkeypatch.setattr(mod.subprocess,'run',failed)
    for role in ('wsl','server'):(tmp_path/role).mkdir()
    a=SimpleNamespace(commit='a'*40,gpu=0,output=tmp_path/'screen',wsl_evidence=tmp_path/'wsl',
        server_evidence=tmp_path/'server',stats_root=tmp_path/'stats',split_manifest=tmp_path/'split',source_lock=tmp_path/'source')
    with pytest.raises(ValueError,match='job failed'):mod.run(a)
    assert len(calls) == 1 and (a.output/'failed.json').exists()
    assert not (a.output/'verification.json').exists()
    artifact=mod.package(a.output,a.commit)
    with zipfile.ZipFile(artifact['path']) as z:assert 'failed.json' in z.namelist()
