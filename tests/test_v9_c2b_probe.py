from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess

import pytest
import torch

import v9_c2b_probe as mod
from tests.test_v9_ssra import payload_for
from reproducibility import state_dict_sha256
from tests.test_p1d_routes import setup_factory
from tests.test_p1d_tox import fixture_factory


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def synthetic_runtime(tmp_path,monkeypatch,setup_factory):
    import p1d4_runtime
    def make(setting):
        if setting=='ToxAcute':
            f=fixture_factory();t=f.make_trainer()
        else:
            f=setup_factory[0](setting)[0];t=f.make_trainer(original=True)
        return f,t
    contexts={s:make(s) for s in mod.c0.SETTINGS}
    def counts(setting):
        f,t=contexts[setting];ds=f.datasets['train'] if setting=='ToxAcute' else t.datasets['train']
        return list(ds),{k:len(v) for k,v in ds.items()}
    def records(f,t,setting):
        ds=f.datasets['train'] if setting=='ToxAcute' else t.datasets['train']
        out=[]
        for task,view in ds.items():
            for i in range(len(view)):
                graph=view[i]
                out.append(dict(task=task,sample_id=view.get_sample_id(i),canonical=graph.canonical_smiles,
                                group=graph.canonical_smiles,split='train'))
        return out
    lock=tmp_path/'lock.json'
    lock.write_text(json.dumps(dict(initial_encoders={s:state_dict_sha256(t.model.encoder) for s,(f,t) in contexts.items()})),encoding='utf8')
    monkeypatch.setattr(mod,'LOCK',lock)
    monkeypatch.setattr(mod.c0,'tasks_and_counts',counts)
    monkeypatch.setattr(mod.c0,'contract',lambda s:dict(synthetic=True,setting=s))
    monkeypatch.setattr(mod,'training_records',records)
    monkeypatch.setattr(p1d4_runtime,'factory_for',lambda repo,*,setting,**kw:make(setting)[0])
    stats=tmp_path/'stats'
    for source in ('Animal56','Nonhuman104'):
        (stats/source).mkdir(parents=True);(stats/source/'moments.pt').write_bytes(b'synthetic')
    monkeypatch.setattr(mod,'TEST_STATS',stats,raising=False)
    monkeypatch.setattr(mod,'load_stats',lambda root,setting,encoder:payload_for(encoder))
    return make


@pytest.mark.parametrize('job',mod.jobs(),ids=lambda j:j['id'])
def test_full_twelve_update_candidate_path_on_real_synthetic_graphs(job,tmp_path,synthetic_runtime):
    f,t=synthetic_runtime(job['setting']);out=tmp_path/job['id'];out.mkdir()
    result=mod.one_job(f,t,job,out,'a'*40,'cpu',mod.TEST_STATS)
    assert len(result['rows'])==12
    assert result['initial_encoder']==result['final_encoder']
    assert result['initial_target']!=result['final_target']
    assert result['train_replay_identical'] is True and result['restore_updates']==0
    assert result['scientific_pass'] is False and result['validation_evaluated'] is False
    assert result['checkpoint_bytes']>0
    assert 'candidate_overhead' not in result['summary']['excludes']
    assert 'measured_candidate_forward_backward_update' in result['summary']['includes']
    assert not list(out.glob('*validation*')) and not list(out.glob('*test*'))


def full_suite(root,make):
    root.mkdir()
    mod.write(root/'launch.json',dict(task=mod.TASK,commit='a'*40,jobs=mod.jobs()))
    mod.write(root/'metadata_preflight.json',mod.metadata_preflight('a'*40,root/'split',root/'source'))
    for job in mod.jobs():
        f,t=make(job['setting']);out=root/job['id'];out.mkdir()
        r=mod.one_job(f,t,job,out,'a'*40,'cpu',mod.TEST_STATS)
        # Deliberate synthetic timing/footprint evidence for validator tests.
        r.update(peak_allocated_bytes=1,peak_reserved_bytes=2)
        mod.write(out/'receipt.json',r)
    return root


def test_independent_verifier_rebuilds_models_optimizer_and_train_members(tmp_path,synthetic_runtime):
    root=full_suite(tmp_path/'suite',synthetic_runtime)
    result=mod.verify(root,'a'*40,tmp_path/'split',tmp_path/'source',mod.TEST_STATS)
    assert result['optimizer_updates']==72 and result['remaining_cost_updates']==144
    assert result['formal_training_authorized'] is False
    assert result['content_status']=='PASS' and result['scientific_acceptance']=='NOT_ASSESSED'
    assert len(result['results'])==6


@pytest.mark.parametrize('change',['members','sample_digest','adam_state','source','summary','target_identity','parameter_counts'])
def test_forged_self_consistent_evidence_rejected(tmp_path,synthetic_runtime,change):
    # Verification rejects on the first job; later outputs deliberately absent.
    root=tmp_path/'suite';root.mkdir()
    mod.write(root/'launch.json',dict(task=mod.TASK,commit='a'*40,jobs=mod.jobs()))
    mod.write(root/'metadata_preflight.json',mod.metadata_preflight('a'*40,root/'split',root/'source'))
    job=mod.jobs()[0];out=root/job['id'];out.mkdir();f,t=synthetic_runtime(job['setting'])
    r=mod.one_job(f,t,job,out,'a'*40,'cpu',mod.TEST_STATS);r.update(peak_allocated_bytes=1,peak_reserved_bytes=2)
    if change=='members':
        records=mod.read(out/'support_records.json');records[0]['sample_id']='forged'
        (out/'support_records.json').write_text(json.dumps(records),encoding='utf8')
        bank=mod.TrainSupport(records)
        (out/'support_identity.json').write_text(json.dumps(bank.identity),encoding='utf8')
        r['identity']['support']=bank.identity
    if change=='sample_digest':
        r['rows'][0]['sample_ids_sha256']='0'*64
        (out/'step_00.json').write_text(json.dumps(r['rows'][0]),encoding='utf8')
    if change in ('adam_state','source'):
        path=out/'smoke_state.pt';payload=torch.load(path,weights_only=True)
        if change=='adam_state':payload['optimizer_state']['state'].clear()
        else:payload['model_state'][next(k for k in payload['model_state'] if k.startswith('encoder.'))].add_(1)
        torch.save(payload,path);r['checkpoint_sha256']=mod.sha(path);r['checkpoint_bytes']=path.stat().st_size
    if change=='summary':r['summary']['train_only_40_epoch_hours_rough']=999
    if change=='target_identity':r['initial_target']='0'*64
    if change=='parameter_counts':r['trainable_parameters']+=1
    mod.write(out/'receipt.json',r)
    with pytest.raises(ValueError):mod.verify(root,'a'*40,tmp_path/'split',tmp_path/'source',mod.TEST_STATS)



def test_prior_evidence_exact_bytes_and_claim(tmp_path,monkeypatch):
    root=tmp_path/'stats';root.mkdir();(root/'receipt.json').write_text('{}')
    claim=tmp_path/'.tmp/v9_c2a_attempts_20260929/attempt.json';claim.parent.mkdir(parents=True);claim.write_text('{}')
    lock=tmp_path/'lock.json';mod.write(lock,dict(accepted_074_files={'receipt.json':mod.sha(root/'receipt.json')},accepted_074_claim_sha256=mod.sha(claim),accepted_074_zip_sha256='a'*64))
    monkeypatch.setattr(mod,'LOCK',lock);monkeypatch.setattr(mod,'REPO',tmp_path)
    assert mod.prior_gate(root)['canonical_sha256']=='a'*64
    (root/'receipt.json').write_text('{"forged":true}')
    with pytest.raises(ValueError):mod.prior_gate(root)


def test_gate_rejects_incomplete_or_skipped_suite(tmp_path):
    import xml.etree.ElementTree as ET
    root=tmp_path/'gate';root.mkdir();suite=ET.Element('testsuite')
    names=['tests.'+Path(t).stem for t in mod.TESTS]
    for i in range(mod.TEST_COUNT):ET.SubElement(suite,'testcase',classname=names[i%len(names)],name=str(i))
    def write_gate():
        ET.ElementTree(suite).write(root/'tests.xml')
        data=dict(task=mod.TASK,role='wsl',commit='a'*40,tests=list(mod.TESTS),junit_sha256=mod.sha(root/'tests.xml'),real_data_optimizer_updates=0)
        (root/'gate.json').write_text(json.dumps(data))
    write_gate();mod.gate(root,'a'*40,'wsl')
    suite.remove(suite[-1]);write_gate()
    with pytest.raises(ValueError):mod.gate(root,'a'*40,'wsl')
    ET.SubElement(suite,'testcase',classname=names[0],name='last');ET.SubElement(suite[0],'skipped');write_gate()
    with pytest.raises(ValueError):mod.gate(root,'a'*40,'wsl')


def test_claim_and_closed_matrix(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'REPO',tmp_path);mod.claim(tmp_path/'one','a'*40)
    with pytest.raises(FileExistsError):mod.claim(tmp_path/'two','a'*40)
    assert len(mod.jobs())==6
    for extra in (dict(seed=43),dict(method='CorDA'),dict(setting='Human6'),dict(unlimited=True)):
        with pytest.raises(ValueError):mod.checked_job(dict(mod.jobs()[0],**extra))


@pytest.mark.parametrize('change',['none','moment_bytes','encoder','wrong_source','extra_buffer'])
def test_locked_stats_loader_rejects_wrong_bytes_or_actual_encoder(tmp_path,monkeypatch,change):
    from tests.test_v9_srgt import fixture
    base=fixture()[0];encoder=base.encoder;payload=payload_for(encoder)
    ident=dict(source='Animal56',encoder_sha256=state_dict_sha256(encoder),selected=list(range(64)))
    payload['identity']=ident
    root=tmp_path/'stats';folder=root/'Animal56';folder.mkdir(parents=True)
    torch.save(payload,folder/'moments.pt');mod.write(folder/'selection.json',ident);mod.write(folder/'receipt.json',{})
    lock=tmp_path/'lock.json';mod.write(lock,dict(initial_encoders={'B':ident['encoder_sha256'],'A':ident['encoder_sha256']},
        accepted_074_files={'Animal56/'+p.name:mod.sha(p) for p in folder.iterdir()}))
    monkeypatch.setattr(mod,'LOCK',lock)
    if change=='none':assert mod.load_stats(root,'B',encoder)['identity']==ident;return
    if change=='moment_bytes':
        with (folder/'moments.pt').open('ab') as f:f.write(b'forged')
    if change=='encoder':next(encoder.parameters()).data.add_(1)
    if change=='extra_buffer':encoder.register_buffer('unknown',torch.tensor(1.))
    with pytest.raises((ValueError,FileNotFoundError)):
        mod.load_stats(root,'A' if change=='wrong_source' else 'B',encoder)


@pytest.mark.parametrize('failure',['exit','timeout'])
def test_supervisor_failure_preserves_claim_stops(tmp_path,monkeypatch,failure):
    import p1d4_batch
    monkeypatch.setattr(mod,'REPO',tmp_path);monkeypatch.setattr(mod.c0,'check_code',lambda *a:None)
    monkeypatch.setattr(mod,'gate',lambda *a:None);monkeypatch.setattr(mod,'prior_gate',lambda *a:{})
    monkeypatch.setattr(mod,'metadata_preflight',lambda *a:{})
    monkeypatch.setattr(mod.c0,'gpu_uuid',lambda *a:'GPU-test');monkeypatch.setattr(p1d4_batch,'free_gpus',lambda x:x)
    for role in ('wsl','server'):(tmp_path/role).mkdir()
    calls=[]
    def child(argv,**kw):
        calls.append(argv)
        if failure=='timeout':raise subprocess.TimeoutExpired(argv,kw['timeout'])
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(subprocess,'run',child)
    with pytest.raises((ValueError,subprocess.TimeoutExpired)):
        mod.run(tmp_path/'run','a'*40,0,tmp_path/'split',tmp_path/'source',tmp_path/'wsl',tmp_path/'server',tmp_path/'stats')
    assert len(calls)==1 and (tmp_path/mod.REGISTRY/'attempt.json').exists() and (tmp_path/'run/failed.json').exists()
