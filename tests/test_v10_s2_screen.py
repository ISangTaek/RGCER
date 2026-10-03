from copy import deepcopy
import json
import zipfile

import pytest
import torch

import v10_s2_screen as s
from tests.test_v10_s1_screen import value
from tests.test_v10_context_models import single_thread
from tests.test_v10_cache_integration import named_source
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory


def spec(): return dict(s.SPEC,epochs=2,hidden=8,query_batch=4,context_cap=6)


@pytest.mark.parametrize('name',s.models.NEW+s.models.CONTROLS)
def test_complete_train_and_replay_every_new_structure(name,value,tmp_path):
    data=s.cache.Data(value); job=next(j for j in s.jobs() if j['method']==name)
    receipt=s.train_case(data,job,tmp_path/'run','a'*40,spec())
    assert s.verify_case(s.cache.Data(deepcopy(value)),job,tmp_path/'run','a'*40,spec())==receipt
    assert all(receipt['coverage'][t]['query_observations']==24 for t in data.targets)
    assert all(receipt['coverage'][t]['updates']>=2 for t in data.sources)
    assert receipt['test_evaluated'] is False


@pytest.mark.parametrize('what',('identity','prediction','context','diagnostics','coverage','epoch','optimizer_lr','optimizer_steps'))
def test_reject_modified_results_including_self_consistent_checkpoint_hash(what,value,tmp_path):
    job=s.jobs()[0]; data=s.cache.Data(value); root=tmp_path/'run'
    s.train_case(data,job,root,'a'*40,spec())
    p=root/'receipt.json'; v=s.read(p)
    if what=='identity': v['identity']['reference_lock_sha256']='0'*64
    elif what=='prediction':
        p=root/'selected_validation.json'; v=s.read(p); v[0]['prediction']+=1
    elif what in ('context','diagnostics'):
        p=root/'epoch_001.json'; v=s.read(p); e=v['trace'][0]['episodes'][0]
        if what=='context': e['context']=['wrong']
        else: e['diagnostics']['empty_context_retrieval_rows']=5
    elif what=='coverage': v['coverage'][data.targets[0]]['query_observations']+=1
    elif what=='epoch': v['best_epoch']=99
    else:
        cp=torch.load(root/'best.pt',weights_only=True)
        if what=='optimizer_lr': cp['optimizer']['param_groups'][0]['lr']=.5
        else:
            for state in cp['optimizer']['state'].values(): state['step']+=1
        torch.save(cp,root/'best.pt'); v['checkpoint_sha256']=s.sha(root/'best.pt')
    p.write_text(json.dumps(v),encoding='utf8')
    with pytest.raises(ValueError): s.verify_case(data,job,root,'a'*40,spec())


def fabricated_comparison(value,tmp_path,monkeypatch,errors):
    expected=[r for r in value['rows'] if r['split']=='validation']; refs={}
    def result(error,name):
        rows=[dict(r,prediction=r['label']+error) for r in expected]
        return dict(name=name,rows=rows,score=s.s3.s2.metrics(rows,expected))
    for setting in s.SETTINGS:
        refs[setting]=dict(best=result(.8,'strongest'),selected={m:result(.95,m) for m in s.models.REUSED},
                           mixtures={},historical={'scoreboard':[]})
    for job in s.jobs():
        folder=tmp_path/job['id']; folder.mkdir()
        error=errors.get((job['setting'],job['method']),1.)+.01*job['config']
        x=result(error,job['method'])
        s.write(folder/'selected_validation.json',x['rows'])
        s.write(folder/'receipt.json',dict(identity=dict(job=job),best_epoch=1,selected=x['score'],architecture={'parameters':100}))
    monkeypatch.setattr(s.s1,'grouped_difference',lambda a,b:{'synthetic':True})
    return s.compare(tmp_path,refs)


def test_uniform_gate_rejects_a_per_scene_winner_collage(value,tmp_path,monkeypatch):
    errors={(scene,s.models.NEW[i]):.6 for i,scene in enumerate(s.SETTINGS)}
    r=fabricated_comparison(value,tmp_path,monkeypatch,errors)
    assert r['development_candidates']==[] and r['selected_candidate'] is None
    assert r['decision']=='STOP_DUAL_CONTEXT_NO_UNIFIED_WINNER'


def test_uniform_gate_respects_new_control_and_stable_min_gain_ranking(value,tmp_path,monkeypatch):
    errors={}
    for scene in s.SETTINGS:
        errors[scene,'DCR_PARALLEL']=.6
        errors[scene,'DCR_CONTEXT_FIRST']=.5
        errors[scene,'DCR_RETRIEVAL_FIRST']=.75
        errors[scene,'BSA_MSE']=.7
    r=fabricated_comparison(value,tmp_path,monkeypatch,errors)
    assert r['development_candidates']==['DCR_CONTEXT_FIRST','DCR_PARALLEL']
    assert r['selected_candidate']=='DCR_CONTEXT_FIRST'
    assert r['uniform_superiority_established'] is False


def test_matrix_has_no_retraining_of_retained_methods_or_extra_seeds():
    jobs=s.jobs(); assert len(jobs)==len({j['id'] for j in jobs})==36
    assert {j['method'] for j in jobs}==set(s.models.NEW+s.models.CONTROLS)
    assert {j['seed'] for j in jobs}=={42} and {j['config'] for j in jobs}=={0,1}
    lock=s.refs.lock()
    assert sum(lock['trajectory_updates'][j['setting']] for j in jobs)==s.SPEC['new_optimizer_updates']==30240


def test_package_keeps_inner_reference_checksum_and_partial_failure(tmp_path):
    root=tmp_path/'run'; (root/'reference_085').mkdir(parents=True)
    s.write(root/'launch.json',dict(task=s.TASK,commit='a'*40))
    s.write(root/'failed.json',dict(error='synthetic failure'))
    (root/'reference_085/checksums.sha256').write_text('reference manifest fixture\n')
    package=s.package(root,'a'*40)
    with zipfile.ZipFile(package['archive']) as z:
        assert 'reference_085/checksums.sha256' in z.namelist() and 'failed.json' in z.namelist()
        assert z.testzip() is None
    with pytest.raises(ValueError): s.package(root,'a'*40)


def test_package_rejects_unknown_payload(tmp_path):
    s.write(tmp_path/'launch.json',dict(task=s.TASK,commit='a'*40))
    (tmp_path/'unexpected.bin').write_bytes(b'not an approved artifact')
    with pytest.raises(ValueError,match='unexpected'): s.package(tmp_path,'a'*40)


@pytest.mark.parametrize('setting',s.SETTINGS)
def test_all_new_structures_on_real_graphormer_synthetic_interface(setting,joint):
    factory,trainer,source,_=joint(setting)
    raw=s.cache.extract(factory,trainer,named_source(source),setting,'cpu')
    data=s.cache.Data(s.cache.pack(raw,s.refs.SOURCE_COMMIT))
    task=data.targets[0]
    # This graph fixture has only two train rows. Use its disjoint validation
    # query for an interface/backprop probe; actual train-loss roundtrips above
    # use larger synthetic populations and exercise the train-only loss guard.
    q=data.by[task,'validation']
    e=s.episode(data,task,q,42,s.SPEC,evaluation=True)
    for name in s.models.NEW+s.models.CONTROLS:
        model=s.make_model(data,name,s.SPEC)
        prediction=model(e)
        loss=prediction.square().mean(); loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
