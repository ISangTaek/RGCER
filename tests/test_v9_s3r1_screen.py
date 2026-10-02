from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

import v9_s3r1_screen as mod
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime,one_cpu_thread,setup_factory


def populations():
    def row(c,g,split):return dict(task='t',sample_id=c,canonical=c,group=g,split=split,label=1.)
    return [row('human-train','ht','train')],[row('human-val','shared','validation')],[row('animal','shared','train')]


def test_b_cross_database_same_scaffold_is_legal():
    train,valid,source=populations()
    result=mod.s3.check_populations('B',train,valid,source,excluded_canonical={'animal','tox-held-out'})
    assert result['canonical_overlap']==[] and result['shared_scaffold_groups']==['shared']
    assert result['validation_observations_in_shared_scaffolds']==valid


@pytest.mark.parametrize('setting',['ToxAcute','A'])
def test_same_database_scaffold_guard_stays_strict(setting):
    with pytest.raises(ValueError,match='same-dataset'):
        mod.s3.check_populations(setting,*populations())


@pytest.mark.parametrize('location',['target_train','target_validation','auxiliary_validation','target_scaffold','missing_reference'])
def test_disallowed_population_cannot_be_rescued_by_route_b(location):
    train,valid,source=populations();excluded={'animal','tox-held-out'}
    if location=='target_train':train[0]['canonical']='tox-held-out'
    if location=='target_validation':valid[0]['canonical']='tox-held-out'
    if location=='auxiliary_validation':valid[0]['canonical']='animal'
    if location=='target_scaffold':train[0]['group']='shared'
    if location=='missing_reference':excluded=None
    with pytest.raises(ValueError):mod.s3.check_populations('B',train,valid,source,excluded_canonical=excluded)


def share_scaffold(f,t,source):
    valid=mod.s3.s2.observations(f,t,'B','validation')
    source.rows[0]['group']=valid[0]['group']
    source.identity['observations_sha256']=mod.digest(source.rows)


@pytest.mark.parametrize('method',mod.s3.METHODS)
def test_real_synthetic_b_training_and_verifier_allow_shared_scaffold(method,tmp_path,joint):
    job=next(j for j in mod.jobs() if j['method']==method)
    f,t,source,reference=joint('B');share_scaffold(f,t,source)
    spec=dict(mod.s3.SPEC,max_epochs=3,warmup_epochs=1)
    result=mod.s3.train_one(f,t,source,job,tmp_path,'a'*40,'cpu',reference,spec,task_id=mod.TASK)
    f,t,source,reference=joint('B');share_scaffold(f,t,source)
    checked=mod.s3.verify_job(f,t,source,job,tmp_path,'a'*40,reference,spec,task_id=mod.TASK)
    assert checked==result and checked['task']==mod.TASK and checked['identity']['task']==mod.TASK
    # A full-Tox overlap not present in the auxiliary train rows must still fail.
    f._table.overlaps=frozenset([mod.s3.s2.observations(f,t,'B','validation')[0]['canonical']])
    with pytest.raises(ValueError,match='all-partition'):
        mod.s3.verify_job(f,t,source,job,tmp_path,'a'*40,reference,spec,task_id=mod.TASK)


def test_training_rejects_all_partition_overlap_before_any_update(tmp_path,joint):
    f,t,source,reference=joint('B')
    f._table.overlaps=frozenset([mod.s3.s2.observations(f,t,'B','train')[0]['canonical']])
    out=tmp_path/'rejected';out.mkdir()
    with pytest.raises(ValueError,match='all-partition'):
        mod.s3.train_one(f,t,source,mod.jobs()[0],out,'a'*40,'cpu',reference)
    assert list(out.iterdir())==[]


def test_three_job_preflight_uses_metadata_without_forward_or_updates(joint,monkeypatch):
    f,t,source,ref=joint('B');share_scaffold(f,t,source)
    reference={j['id']:ref for j in mod.jobs()};accepted=[dict(identity=dict(source=source.identity))]
    def forbidden(*a,**kw):raise AssertionError('preflight must not run model forward')
    monkeypatch.setattr(t.model,'forward',forbidden)
    result=mod.preflight_context(f,t,source,'a'*40,reference,accepted)
    assert result['model_forward_calls']==result['optimizer_updates']==0 and set(result['identities'])==set(reference)
    assert result['population_check']['shared_scaffold_groups']
    bad=deepcopy(reference);bad[mod.jobs()[1]['id']]['identity']['train_sha256']='bad'
    with pytest.raises(ValueError,match='identity pairing'):
        mod.preflight_context(f,t,source,'a'*40,bad,accepted)


@pytest.fixture
def accepted_fixture(tmp_path,monkeypatch):
    root=tmp_path/'078';root.mkdir();reference={}
    mod.write(root/'launch.json',dict(task=mod.s3.TASK,commit=mod.OLD_COMMIT,jobs=mod.s3.jobs(),spec=mod.s3.SPEC))
    (root/'B_B1_1E4_s42').mkdir();mod.write(root/'B_B1_1E4_s42/started.json',dict(started=True))
    for job in mod.inherited_jobs():
        out=root/job['id'];out.mkdir()
        valid=[dict(task='t',sample_id='v',canonical='CC',group='g',split='validation',label=1.)]
        rows=[dict(valid[0],prediction=2.)];score=mod.s3.s2.metrics(rows,valid)
        identity=dict(task=mod.s3.TASK,commit=mod.OLD_COMMIT,job=job,spec=mod.s3.SPEC)
        history=[dict(epoch=e,validation=score) for e in range(1,41)]
        r=dict(task=mod.s3.TASK,commit=mod.OLD_COMMIT,job=job,identity=identity,history=history,
               updates=240 if job['setting']=='ToxAcute' else 600,best_epoch=1,selected=score)
        for name,value in [('identity',identity),('receipt',r),('validation_observations',valid),('selected_validation',rows)]:
            mod.write(out/(name+'.json'),value)
        for h in history:
            mod.write(out/f'epoch_{h["epoch"]:03d}.json',h)
            mod.write(out/f'validation_epoch_{h["epoch"]:03d}.json',rows)
        (out/'best.pt').write_bytes(b'immutable synthetic checkpoint')
        reference[job['id']]=dict(identity=dict(validation_sha256=mod.digest(valid)))
    lock=dict(schema='v9_s3r1_inherited_v1',commit=mod.OLD_COMMIT,task=mod.s3.TASK,spec=mod.s3.SPEC,
              completed_jobs=mod.inherited_jobs(),files={p.relative_to(root).as_posix():mod.sha(p) for p in root.rglob('*') if p.is_file()})
    path=tmp_path/'lock.json';mod.write(path,lock);monkeypatch.setattr(mod,'LOCK',path)
    monkeypatch.setattr(mod.s3,'gate_check',lambda *a:None)
    monkeypatch.setattr(mod.s3,'references',lambda *a:reference)
    return root,lock


@pytest.mark.parametrize('damage',['none','checkpoint','missing','extra','wrong_commit_rehashed','metric_rehashed','matrix'])
def test_immutable_completed_six_and_contract_validation(accepted_fixture,damage):
    root,lock=accepted_fixture;folder=root/mod.inherited_jobs()[0]['id']
    if damage=='none':
        results,_=mod.inherited(root);assert len(results)==6 and sum(r['updates'] for r in results)==2520
        return
    if damage=='checkpoint':(folder/'best.pt').write_bytes(b'different checkpoint')
    if damage=='missing':(folder/'selected_validation.json').unlink()
    if damage=='extra':mod.write(root/'unexpected.json',{})
    if damage in ('wrong_commit_rehashed','metric_rehashed'):
        path=folder/'receipt.json';r=mod.read(path)
        if damage=='wrong_commit_rehashed':r['commit']='b'*40
        else:r['selected']['macro_rmse']=0.
        path.write_text(json.dumps(r),encoding='utf8');lock['files'][path.relative_to(root).as_posix()]=mod.sha(path)
    if damage=='matrix':lock['completed_jobs']=mod.inherited_jobs()[:-1]
    mod.LOCK.write_text(json.dumps(lock),encoding='utf8')
    with pytest.raises(ValueError):mod.inherited(root)


@pytest.fixture
def dispatch(tmp_path,monkeypatch):
    import p1d4_batch
    monkeypatch.setattr(mod,'REPO',tmp_path);lock=tmp_path/'lock.json';mod.write(lock,{})
    monkeypatch.setattr(mod,'LOCK',lock);monkeypatch.setattr(mod.s3.s2.c0,'check_code',lambda *a:None)
    monkeypatch.setattr(mod,'gate_check',lambda *a:None);monkeypatch.setattr(mod,'inherited',lambda *a:([],{}))
    monkeypatch.setattr(mod.s3,'copy_references',lambda *a:None)
    monkeypatch.setattr(p1d4_batch,'free_gpus',lambda a:a);monkeypatch.setattr(mod.s3.s2.c0,'gpu_uuid',lambda *a:'GPU-synthetic')
    monkeypatch.setattr(mod,'preflight',lambda *a:dict(optimizer_updates=0));monkeypatch.setattr(mod,'verify',lambda *a:dict(synthetic=True))
    for name in ('wsl','server','078'):(tmp_path/name).mkdir()
    a=SimpleNamespace(commit='a'*40,gpu=0,output=tmp_path/'new',inherited_root=tmp_path/'078',
                      wsl_evidence=tmp_path/'wsl',server_evidence=tmp_path/'server',split_manifest=tmp_path/'split',source_lock=tmp_path/'source')
    return a


@pytest.mark.parametrize('failure',['none','preflight','first_worker'])
def test_dispatch_preflight_first_only_three_jobs_and_persistent_attempt(dispatch,monkeypatch,failure):
    a=dispatch;calls=[]
    def child(argv,**kw):
        assert (a.output/'preflight.json').is_file();calls.append(argv)
        return SimpleNamespace(returncode=17 if failure=='first_worker' else 0)
    monkeypatch.setattr(mod.subprocess,'run',child)
    if failure=='preflight':
        def bad(*a):raise ValueError('synthetic invalid metadata')
        monkeypatch.setattr(mod,'preflight',bad)
    if failure=='none':
        mod.run(a);assert [v[v.index('--job')+1] for v in calls]==[j['id'] for j in mod.jobs()]
        assert (a.output/'verification.json').is_file()
    else:
        with pytest.raises(ValueError):mod.run(a)
        assert len(calls)==(0 if failure=='preflight' else 1)
        assert (a.output/'failed.json').is_file() and not (a.output/'verification.json').exists()
    # Changing only the output directory does not grant another attempt.
    a.output=a.output.with_name('another-output')
    with pytest.raises(ValueError,match='attempt already consumed'):mod.run(a)
    assert not a.output.exists()


def test_worker_rejects_out_of_scope_before_gpu_or_data_access():
    with pytest.raises(ValueError,match='only three B'):
        mod.worker(SimpleNamespace(job='ToxAcute_B1_1E4_s42'))


@pytest.mark.parametrize('damage',['commit','job','spec','output'])
def test_worker_requires_exact_launch_and_claim(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(mod,'REPO',tmp_path)
    root=tmp_path/'new';root.mkdir();previous=tmp_path/'old';(tmp_path/mod.REGISTRY).mkdir(parents=True)
    value=mod.claim_value('a'*40,root,previous);mod.write(tmp_path/mod.REGISTRY/'attempt.json',value)
    launch=dict(task=mod.TASK,commit='a'*40,jobs=mod.jobs(),spec=deepcopy(mod.s3.SPEC),claim=value)
    if damage=='commit':launch['commit']='b'*40
    if damage=='job':launch['jobs']=mod.s3.jobs()
    if damage=='spec':launch['spec']['max_epochs']=80
    if damage=='output':launch['claim']['output']=str(tmp_path/'other')
    mod.write(root/'launch.json',launch)
    with pytest.raises(ValueError):mod.check_launch(root,'a'*40,previous)


@pytest.mark.parametrize('damage',['none','inherited_summary','worker_exit'])
def test_combined_verifier_keeps_old_and_new_provenance(tmp_path,monkeypatch,damage):
    lock=tmp_path/'lock.json';mod.write(lock,dict(synthetic=True));monkeypatch.setattr(mod,'LOCK',lock)
    mod.write(tmp_path/'inherited_reference.json',mod.read(lock))
    score=dict(macro_rmse=1.,endpoints={'t':dict(rmse=1.,n=1)})
    def result(j,commit,task):
        return dict(job=j,commit=commit,task=task,updates={'ToxAcute':240,'A':600,'B':360}[j['setting']],
                    best_epoch=1,history=[{}]*40,selected=deepcopy(score),checkpoint_sha256='f'*64)
    accepted=[result(j,mod.OLD_COMMIT,mod.s3.TASK) for j in mod.inherited_jobs()]
    reference={j['id']:dict(selected=deepcopy(score)) for j in mod.s3.s2.jobs()}
    monkeypatch.setattr(mod,'check_launch',lambda *a:None);monkeypatch.setattr(mod,'gate_check',lambda *a:None)
    monkeypatch.setattr(mod,'inherited',lambda *a:(accepted,reference));monkeypatch.setattr(mod.s3,'references',lambda *a:reference)
    monkeypatch.setattr(mod.s3,'context',lambda *a:(None,None,None));monkeypatch.setattr(mod,'preflight_context',lambda *a:{})
    monkeypatch.setattr(mod.s3,'verify_job',lambda f,t,s,j,out,c,ref,**kw:result(j,c,kw['task_id']))
    mod.write(tmp_path/'preflight.json',{})
    summary=mod.inherited_summary(accepted)
    if damage=='inherited_summary':summary[0]['commit']='a'*40
    mod.write(tmp_path/'inherited_results.json',summary)
    for j in mod.jobs():mod.write(tmp_path/(j['id']+'.command.json'),dict(exit_code=3 if damage=='worker_exit' else 0))
    if damage!='none':
        with pytest.raises(ValueError):mod.verify(tmp_path,'a'*40,None,None,tmp_path/'old')
        return
    r=mod.verify(tmp_path,'a'*40,None,None,tmp_path/'old')
    assert r['new_updates']==1080 and r['inherited_updates']==2520 and r['total_epochs']==360
    assert {v['produced_by_commit'] for v in r['provenance'][:6]}=={mod.OLD_COMMIT}
    assert {v['produced_by_commit'] for v in r['provenance'][6:]}=={'a'*40}
    assert len(r['comparisons'])==9 and r['new_smoke_updates']==0 and not r['unified_superiority_confirmed']
