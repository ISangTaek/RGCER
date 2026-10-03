from copy import deepcopy
import json
import zipfile
import pytest
import torch
import v11_s1_screen as s
from tests.test_v11_features import value,chemical_value,data,single_thread


def spec(): return dict(s.SPEC,epochs=2,query_batch=4,context_cap=6,tiny=True)


@pytest.mark.parametrize('name',s.models.NEW+s.models.CONTROLS)
def test_full_training_roundtrip_with_shared_source_target_coverage(data,tmp_path,name):
    job=next(j for j in s.jobs() if j['method']==name)
    r=s.train_case(data,job,tmp_path/'run','a'*40,spec())
    assert s.verify_case(data,job,tmp_path/'run','a'*40,spec())==r
    assert all(r['coverage'][t]['unique_query_observations']==12 and r['coverage'][t]['query_observations']==24 for t in data.targets)
    assert all(r['coverage'][t]['updates']>=2 for t in data.sources)
    assert r['scientific_acceptance']=='PENDING_REVIEW' and not r['test_evaluated']


@pytest.mark.parametrize('kind',('prediction','chemical','optimizer','checkpoint','context','metric','coverage'))
def test_independent_verifier_rejects_self_consistent_forgery(data,tmp_path,kind):
    job=next(j for j in s.jobs() if j['method']=='TCF'); out=tmp_path/'run'
    r=s.train_case(data,job,out,'a'*40,spec())
    if kind=='chemical': data.chemical['robust_factors'][0]+=1
    elif kind in ('optimizer','checkpoint'):
        payload=torch.load(out/'best.pt',weights_only=True)
        if kind=='optimizer': payload['optimizer']['param_groups'][0]['lr']*=2
        else: payload['state']['readout.2.bias']+=1
        torch.save(payload,out/'best.pt'); r['checkpoint_sha256']=s.sha(out/'best.pt')
        (out/'receipt.json').write_text(json.dumps(r))
    elif kind=='coverage':
        r['coverage'][data.targets[0]]['unique_query_observations']-=1
        (out/'receipt.json').write_text(json.dumps(r))
    elif kind=='prediction':
        p=s.read(out/'selected_validation.json'); p[0]['prediction']+=1
        (out/'selected_validation.json').write_text(json.dumps(p))
    else:
        h=s.read(out/'epoch_001.json')
        if kind=='metric': h['score']['macro_rmse']+=1
        else: h['trace'][0]['episodes'][0]['context'][0]='invalid'
        (out/'epoch_001.json').write_text(json.dumps(h))
    with pytest.raises(ValueError): s.verify_case(data,job,out,'a'*40,spec())


def test_matrix_and_author_lr_configurations_are_frozen():
    assert len(s.jobs())==72 and len(s.models.NEW)==10 and len(s.models.CONTROLS)==2
    assert {j['seed'] for j in s.jobs()}=={42} and s.SPEC['epochs']==40
    assert s.SPEC['new_optimizer_updates']==60480 and s.SPEC['tiny'] is False
    assert s.refs.lock()['trajectory_updates']=={'ToxAcute':720,'A':1120,'B':680}


def mock_comparison(data,tmp_path):
    reference={}; expected=[r for r in data.rows if r['split']=='validation']
    for setting in s.SETTINGS:
        rows=[dict(r,prediction=r['label']+1) for r in expected]
        fixed=dict(name='frozen',score=s.s3.s2.metrics(rows,expected),rows=rows)
        reference[setting]=dict(best=fixed,selected086={},selected={},mixtures={},historical={'scoreboard':[]})
        for j in s.jobs():
            if j['setting']!=setting: continue
            e=1.1 if j['method'] in s.models.CONTROLS else .95
            if j['method']=='TCF': e=.9
            pred=[dict(r,prediction=r['label']+e) for r in expected]; folder=tmp_path/j['id']; folder.mkdir()
            s.write(folder/'receipt.json',dict(identity={'job':j},selected=s.s3.s2.metrics(pred,expected),best_epoch=1,architecture={'parameters':100}))
            s.write(folder/'selected_validation.json',pred)
    return reference


def test_gate_requires_same_family_in_all_three_settings_and_beats_both_controls(data,tmp_path):
    refs=mock_comparison(data,tmp_path); got=s.compare(tmp_path,refs)
    assert got['selected_candidate']=='TCF' and not got['uniform_superiority_established']
    for j in s.jobs():
        if j['setting']=='B' and j['method'] in s.models.NEW:
            p=tmp_path/j['id']/'receipt.json'; r=s.read(p); r['selected']['macro_rmse']=1.2; p.write_text(json.dumps(r))
    assert s.compare(tmp_path,refs)['selected_candidate'] is None


def test_package_includes_nested_reference_manifests_but_rejects_unknown_files(tmp_path):
    root=tmp_path/'run'; root.mkdir(); s.write(root/'launch.json',dict(task=s.TASK,commit='a'*40))
    for sub in ('reference_086','reference_086/reference_085'):
        p=root/sub; p.mkdir(exist_ok=True,parents=True); (p/'checksums.sha256').write_text('test')
    (root/'secret.env').write_text('do not ship')
    with pytest.raises(ValueError): s.package(root,'a'*40)
    (root/'secret.env').unlink()
    result=s.package(root,'a'*40)
    with zipfile.ZipFile(result['archive']) as z: assert len(z.namelist())==4
