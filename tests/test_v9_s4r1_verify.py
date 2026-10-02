from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import lmdb
import pytest

import v9_s4r1_verify as m


def test_real_lmdb_duplicate_isolated_by_completed_fresh_processes(tmp_path):
    db=tmp_path/'db';env=lmdb.open(str(db),map_size=1024*1024)
    with env.begin(write=True) as tx:tx.put(b'key',b'original')
    env.close();owner=lmdb.open(str(db),readonly=True,lock=False,readahead=False)
    try:
        with pytest.raises(lmdb.Error,match='already open'):
            lmdb.open(str(db),readonly=True,lock=False,readahead=False)
        script=tmp_path/'verify.py'
        script.write_text("import lmdb,os,sys,json\nassert os.environ['CUDA_VISIBLE_DEVICES']==''\ne=lmdb.open(sys.argv[1],readonly=True,lock=False,readahead=False)\nwith e.begin() as t:assert t.get(b'key')==b'original'\nprint(json.dumps(dict(pid=os.getpid(),device='cpu')))\n",encoding='utf8')
        pids=[]
        for i in range(2):
            prefix=tmp_path/f'child{i}';m.run_child([sys.executable,str(script),str(db)],prefix)
            pids.append(json.loads(Path(str(prefix)+'.log').read_text())['pid'])
        assert len(set(pids))==2 and os.getpid() not in pids
        with owner.begin() as tx:assert tx.get(b'key')==b'original'
    finally:owner.close()


def test_failed_real_child_retains_exit_and_log(tmp_path):
    prefix=tmp_path/'failed'
    with pytest.raises(ValueError,match='readonly child failed'):
        m.run_child([sys.executable,'-c',"print('fixture failure');raise SystemExit(19)"],prefix)
    assert m.read(tmp_path/'failed.command.json')['exit_code']==19 and 'fixture failure' in (tmp_path/'failed.log').read_text()


@pytest.mark.parametrize('damage',['none','changed','missing','extra','sidecar'])
def test_input_snapshot_checks_raw_members_and_sidecar(tmp_path,damage):
    root=tmp_path/'080';root.mkdir();m.write(root/'failed.json',dict(original=True));m.write(root/'receipt.json',dict(commit=m.PRODUCTION_COMMIT))
    expected={p.name:m.sha(p) for p in root.iterdir()};before=m.locked_files(root,expected)
    if damage=='none':
        assert m.locked_files(root,expected)==before;return
    if damage=='changed':(root/'receipt.json').write_text('{}')
    elif damage=='missing':(root/'receipt.json').unlink()
    elif damage=='extra':m.write(root/'verification.json',{})
    else:(root/'checksums.sha256').write_text('0'*64+'  receipt.json\n')
    with pytest.raises(ValueError):m.locked_files(root,expected)


@pytest.mark.parametrize('paths',[('old','old'),('old/child','old'),('old','old/child')])
def test_reject_input_output_overlap(tmp_path,paths):
    with pytest.raises(ValueError,match='overlaps'):m.separate_output(tmp_path/paths[0],tmp_path/paths[1])


def test_production_code_change_is_not_allowed(monkeypatch):
    monkeypatch.setattr(m.s4.s3.s2.c0,'check_code',lambda *a:None)
    good='\n'.join('A\t'+p for p in m.ADDITIONS)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:good)
    m.check_code('a'*40)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:good+'\nM\tv9_s4_screen.py')
    with pytest.raises(ValueError,match='production code'):m.check_code('a'*40)


def test_worker_scope_and_cpu_environment_fail_before_data_access(monkeypatch):
    with pytest.raises(ValueError,match='nine original'):m.worker(SimpleNamespace(job='new_training'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CPU-only'):m.worker(SimpleNamespace(job=m.s4.jobs()[0]['id']))


@pytest.fixture
def dispatch(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'REPO',tmp_path);monkeypatch.setattr(m,'check_code',lambda *a:None)
    monkeypatch.setattr(m,'gate_check',lambda *a:None);monkeypatch.setattr(m,'lock_value',lambda:dict(fixture=True))
    monkeypatch.setattr(m.s4.s3,'context',lambda *a:pytest.fail('parent must not create a data context'))
    for name in ('080','wsl','server'):(tmp_path/name).mkdir()
    a=SimpleNamespace(commit='a'*40,output=tmp_path/'new',input_root=tmp_path/'080',wsl_evidence=tmp_path/'wsl',
                      server_evidence=tmp_path/'server',split_manifest=tmp_path/'split',source_lock=tmp_path/'source')
    results=[]
    for j in m.s4.jobs():
        d=a.input_root/j['id'];d.mkdir();r=dict(job=j,commit=m.PRODUCTION_COMMIT,checkpoint_sha256='f'*64,
              updates={'ToxAcute':240,'A':600,'B':360}[j['setting']],history=[{}]*40)
        m.write(d/'receipt.json',r);results.append(r)
    monkeypatch.setattr(m,'inputs',lambda *a:('immutable',results,{}))
    monkeypatch.setattr(m.s4,'compare',lambda *a:dict(all_scene_candidate_signals=[],unified_superiority_confirmed=False))
    return a


def fixture_child(a,argv,prefix):
    job=next(j for j in m.s4.jobs() if j['id']==argv[argv.index('--job')+1]);folder=a.output/job['id'];folder.mkdir()
    r=dict(task=m.TASK,job=job,verification_commit=a.commit,production_commit=m.PRODUCTION_COMMIT,input_lock_digest=m.digest(m.lock_value()),
        original_receipt_sha256=m.sha(a.input_root/job['id']/'receipt.json'),checkpoint_sha256='f'*64,content_status='PASS',
        raw_train_gradient_replayed=True,new_training_updates=0,device='cpu',pid=1000+m.s4.jobs().index(job),torch_version='fixture')
    m.write(folder/'verification.json',r);m.write(folder/'started.json',dict(task=m.TASK,job=job,verification_commit=a.commit,pid=r['pid']))
    m.write(Path(str(prefix)+'.command.json'),dict(argv=argv,exit_code=0,device='cpu'))


@pytest.mark.parametrize('failure',['none','first_child','input_mutated'])
def test_dispatch_only_nine_cpu_verifiers_no_context_or_training_in_parent(dispatch,monkeypatch,failure):
    a=dispatch;calls=[];before={p.relative_to(a.input_root).as_posix():m.sha(p) for p in a.input_root.rglob('*') if p.is_file()}
    def child(argv,prefix):
        calls.append(argv);assert argv[2]=='_worker' and '--gpu' not in argv and '--mode' not in argv
        if failure=='first_child':raise ValueError('fixture child failure')
        fixture_child(a,argv,prefix)
    monkeypatch.setattr(m,'run_child',child)
    if failure=='input_mutated':
        original=m.inputs;count=[0]
        def changed(*args):
            count[0]+=1;s,r,ref=original(*args);return (s if count[0]==1 else 'modified'),r,ref
        monkeypatch.setattr(m,'inputs',changed)
    if failure=='none':
        m.run(a);v=m.read(a.output/'verification.json')
        assert v['production_commit']==m.PRODUCTION_COMMIT and v['verification_commit']==a.commit
        assert v['original_training_updates']==3600 and v['new_training_updates']==v['new_smoke_updates']==0 and len(v['workers'])==9
        assert [v[v.index('--job')+1] for v in calls]==[j['id'] for j in m.s4.jobs()]
    else:
        with pytest.raises(ValueError):m.run(a)
        assert (a.output/'failed.json').exists() and not (a.output/'verification.json').exists()
        assert len(calls)==(1 if failure=='first_child' else 9)
    assert before=={p.relative_to(a.input_root).as_posix():m.sha(p) for p in a.input_root.rglob('*') if p.is_file()}
    a.output=a.output.with_name('second-attempt')
    with pytest.raises(ValueError,match='attempt already consumed'):m.run(a)
    assert not a.output.exists()


@pytest.mark.parametrize('damage',['production_commit','receipt_hash','bool_updates','raw_replay','device','extra'])
def test_worker_receipt_cannot_relabel_or_forge_success(dispatch,damage):
    a=dispatch;a.output.mkdir();j=m.s4.jobs()[0];fixture_child(a,['--job',j['id']],a.output/j['id'])
    p=a.output/j['id']/'verification.json';r=m.read(p)
    if damage=='production_commit':r[damage]=a.commit
    elif damage=='receipt_hash':r['original_receipt_sha256']='0'*64
    elif damage=='bool_updates':r['new_training_updates']=False
    elif damage=='raw_replay':r['raw_train_gradient_replayed']=False
    elif damage=='device':r['device']='cuda:0'
    else:r['ignored']=True
    p.write_text(json.dumps(r),encoding='utf8')
    with pytest.raises(ValueError):m.check_worker(a,j)


def test_wrong_persistent_claim_rejected(dispatch):
    a=dispatch;a.output.mkdir();(m.REPO/m.REGISTRY).mkdir(parents=True)
    expected=m.claim_value(a);m.write(a.output/'launch.json',dict(task=m.TASK,commit=a.commit,claim=expected))
    m.write(m.REPO/m.REGISTRY/'attempt.json',dict(expected,production_commit=a.commit))
    with pytest.raises(ValueError,match='persistent claim'):m.check_launch(a)
