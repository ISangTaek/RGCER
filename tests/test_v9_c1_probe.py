from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess

import pytest
import torch

import v9_c1_probe as mod
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
    return make


@pytest.mark.parametrize('job',mod.jobs(),ids=lambda j:j['id'])
def test_full_twelve_update_candidate_path_on_real_synthetic_graphs(job,tmp_path,synthetic_runtime):
    f,t=synthetic_runtime(job['setting']);out=tmp_path/job['id'];out.mkdir()
    result=mod.one_job(f,t,job,out,'a'*40,'cpu')
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
    for job in mod.jobs():
        f,t=make(job['setting']);out=root/job['id'];out.mkdir()
        r=mod.one_job(f,t,job,out,'a'*40,'cpu')
        # Deliberate synthetic timing/footprint evidence for validator tests.
        r.update(peak_allocated_bytes=1,peak_reserved_bytes=2)
        mod.write(out/'receipt.json',r)
    return root


def test_independent_verifier_rebuilds_models_optimizer_and_train_members(tmp_path,synthetic_runtime):
    root=full_suite(tmp_path/'suite',synthetic_runtime)
    result=mod.verify(root,'a'*40,tmp_path/'split',tmp_path/'source')
    assert result['optimizer_updates']==72 and result['remaining_cost_updates']==216
    assert result['formal_training_authorized'] is False
    assert result['content_status']=='PASS' and result['scientific_acceptance']=='NOT_ASSESSED'
    assert len(result['results'])==6


@pytest.mark.parametrize('change',['members','sample_digest','adam_state','source','summary','target_identity','parameter_counts'])
def test_forged_self_consistent_evidence_rejected(tmp_path,synthetic_runtime,change):
    # Verification rejects on the first job; later outputs deliberately absent.
    root=tmp_path/'suite';root.mkdir()
    mod.write(root/'launch.json',dict(task=mod.TASK,commit='a'*40,jobs=mod.jobs()))
    job=mod.jobs()[0];out=root/job['id'];out.mkdir();f,t=synthetic_runtime(job['setting'])
    r=mod.one_job(f,t,job,out,'a'*40,'cpu');r.update(peak_allocated_bytes=1,peak_reserved_bytes=2)
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
    with pytest.raises(ValueError):mod.verify(root,'a'*40,tmp_path/'split',tmp_path/'source')


def test_prior_gate_requires_exact_accepted_c0_bytes(tmp_path,monkeypatch):
    lock=json.loads(Path(mod.LOCK).read_bytes())
    prior=tmp_path/'c0';prior.mkdir()
    (prior/'verification.json').write_text('{}')
    with pytest.raises(ValueError,match='accepted C0 verification'):mod.prior_gate(prior)
    lock['verification_sha256']=mod.sha(prior/'verification.json')
    path=tmp_path/'fake_lock.json';path.write_text(json.dumps(lock),encoding='utf8')
    monkeypatch.setattr(mod,'LOCK',path)
    with pytest.raises(FileNotFoundError):mod.prior_gate(prior)


def test_new_claim_cannot_be_reset_with_a_new_output_path(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,'REPO',tmp_path)
    mod.claim(tmp_path/'one','a'*40)
    with pytest.raises(FileExistsError):mod.claim(tmp_path/'two','a'*40)


def test_method_and_seed_matrix_is_closed():
    assert len(mod.jobs())==6
    for job in mod.jobs():mod.checked_job(job)
    for extra in [dict(seed=43),dict(method='SSRA'),dict(setting='Human6'),dict(unlimited=True)]:
        with pytest.raises(ValueError):mod.checked_job(dict(mod.jobs()[0],**extra))


@pytest.mark.parametrize('failure',['exit','timeout'])
def test_supervisor_failure_stops_and_consumes_attempt(tmp_path,monkeypatch,failure):
    import p1d4_batch
    monkeypatch.setattr(mod,'REPO',tmp_path)
    monkeypatch.setattr(mod.c0,'check_code',lambda *a:None)
    monkeypatch.setattr(mod,'gate',lambda *a:None)
    monkeypatch.setattr(mod,'prior_gate',lambda *a:dict(accepted=True))
    monkeypatch.setattr(mod.c0,'gpu_uuid',lambda *a:'GPU-synthetic')
    monkeypatch.setattr(p1d4_batch,'free_gpus',lambda x:x)
    for role in ('wsl','server'):(tmp_path/role).mkdir()
    calls=[]
    def child(argv,**kw):
        calls.append(argv)
        assert 0<kw['timeout']<=1800 and kw['env']['CUDA_VISIBLE_DEVICES']=='GPU-synthetic'
        if failure=='timeout':raise subprocess.TimeoutExpired(argv,1)
        return SimpleNamespace(returncode=3)
    monkeypatch.setattr(subprocess,'run',child)
    with pytest.raises((ValueError,subprocess.TimeoutExpired)):
        mod.run(tmp_path/'run','a'*40,0,tmp_path/'split',tmp_path/'source',tmp_path/'wsl',tmp_path/'server',tmp_path/'prior')
    assert len(calls)==1 and (tmp_path/'run/failed.json').exists()
    assert (tmp_path/mod.REGISTRY/'attempt.json').exists()


def test_failure_package_preserves_checkpoint_and_refuses_overwrite(tmp_path):
    root=tmp_path/'run';root.mkdir()
    mod.write(root/'launch.json',dict(task=mod.TASK))
    mod.write(root/'failed.json',dict(error='synthetic'))
    torch.save(dict(x=torch.ones(2)),root/'smoke_state.pt')
    result=mod.package(root)
    assert result['scientific_acceptance']=='NOT_ASSESSED'
    assert mod.sha(tmp_path/'run.zip')==result['sha256']
    with pytest.raises(ValueError):mod.package(root)
