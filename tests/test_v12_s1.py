from copy import deepcopy
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import v12_s1 as s
import v12_s1_data as d
import v12_relation as relation


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def scene():
    rng = np.random.default_rng(52)
    canonical = ['m'+str(i) for i in range(48)]
    descriptors = rng.normal(size=(48,24))
    names = d.read(d.LOCK)['feature_spec']['descriptors']
    descriptors[:,names.index('MolWt')] = 150+20*descriptors[:,names.index('MolWt')]
    packed = rng.integers(0,256,size=(48,256),dtype=np.uint8)
    sources, targets = ['mouse_oral_LD50','rat_oral_LD50'], ['human_oral_TDLo']
    rows, fns, chem = [], [], []
    for task in sources+targets:
        members = range(36) if task in sources else list(range(18))+[43,44,45]
        for i in members:
            group = 'g'+str(i//2)
            label = float(descriptors[i,1]*(2. if task.startswith('rat') else 1.)+.02*i)
            rows.append(dict(task=task,sample_id=str(i),canonical=canonical[i],group=group,
                             split='train' if i<36 else 'validation',label=label))
            fns.append([descriptors[i,1],descriptors[i,2]])
            chem.append(i)
    return d.validate_scene(d.Scene('ToxAcute',rows,sources,targets,np.array(fns),canonical,np.array(chem),
                                   packed,descriptors,np.zeros((48,24),dtype=bool),names,dict(fixture=True)))


def test_cv_qualification_is_not_mislabeled_strict_oof(scene):
    q = scene.qualification()
    assert q['source_teacher'] == 'FULL_SOURCE_TRAIN_NOT_STRICT_GROUP_OOF'
    assert q['target_train_shared_source_molecules'] == 18
    assert all(n>0 for n in q['target_fold_counts'][scene.targets[0]].values())


@pytest.mark.parametrize('mutation', ['test','duplicate','canonical','group_overlap','nonfinite','missing_task'])
def test_bad_population_fails_closed(scene,mutation):
    item = deepcopy(scene)
    if mutation=='test': item.rows[-1]['split']='test'
    elif mutation=='duplicate': item.rows[-1]=dict(item.rows[-2])
    elif mutation=='canonical': item.row_chemical[-1]=0
    elif mutation=='group_overlap': item.rows[-1]['group']=item.rows[0]['group']
    elif mutation=='nonfinite': item.functions[0,0]=np.nan
    else: item.targets.append('missing_oral_TDLo')
    with pytest.raises(ValueError): d.validate_scene(item)


def test_target_validation_and_target_labels_do_not_fit_shared_preprocessing(scene):
    changed = deepcopy(scene)
    for r in changed.rows:
        if r['task'] in changed.targets: r['label']+=1e5
    valid_ids = changed.indices(changed.targets,'validation')
    changed.descriptors[changed.row_chemical[valid_ids]] += 10000
    changed.functions[valid_ids] += 10000
    _,_,before=s.prepare_analytic(scene)
    _,_,after=s.prepare_analytic(changed)
    assert before == after


def test_fold_labels_change_scores_not_fitted_models(scene):
    z,q,shared=s.prepare_analytic(scene)
    case=next(iter(s.case_plan(scene)))
    before=s.target_case(scene,case,z,q,shared['prior'])
    altered=deepcopy(scene)
    for row in altered.rows:
        if row['task'] in altered.targets and row['split']=='train' and d.fold_id(row['group'])==case['fold']:
            row['label']+=50
    after=s.target_case(altered,case,z,q,shared['prior'])
    assert before['models']==after['models']
    assert before['score']!=after['score']


def test_validation_scores_cannot_choose_configuration(scene,tmp_path):
    s.run_analytic(scene,tmp_path/'analytic')
    results={case['id']:d.read(tmp_path/'analytic'/(case['id']+'.json')) for case in s.case_plan(scene)}
    before=s.select_cases(scene,results)
    for result in results.values():
        if result['case']['fold'] is None:
            result['score']['macro_rmse']=-1e6 if result['case']['configuration']==1 else 1e6
    after=s.select_cases(scene,results)
    assert {m:v['configuration'] for m,v in before.items()}=={m:v['configuration'] for m,v in after.items()}


def test_four_cells_both_ends_have_disjoint_chemical_groups(scene):
    selected=list(relation.cases(scene))
    assert selected and len(selected)==relation.qualification(scene)['directed_fold_cases']
    for case in selected:
        for row in case['predictions']:
            assert d.fold_id(row['left_group'])==d.fold_id(row['right_group'])==case['fold']
        assert case['train_unique_molecules']<=relation.SPEC['max_train_molecules']
        assert case['train_likelihood_weight']==pytest.approx(case['train_unique_groups'])


def test_wrong_route_endpoint_not_eligible_for_relation(scene):
    altered=deepcopy(scene)
    original=altered.sources[1]; altered.sources[1]='rat_skin_LDLo'
    for row in altered.rows:
        if row['task']==original: row['task']=altered.sources[1]
    q=relation.qualification(altered)
    assert q['eligible_task_pairs']==0 and q['unrepresented_source_tasks']==sorted(altered.sources)


def test_reused_bank_requires_equal_labels_and_raw_chemistry(scene):
    assert s.source_bank_equivalence(scene,deepcopy(scene))['relation_reused_for_B']
    changed=deepcopy(scene); changed.rows[0]['label']+=.1
    with pytest.raises(ValueError): s.source_bank_equivalence(scene,changed)
    changed=deepcopy(scene); changed.fp_packed[0,0]^=1
    with pytest.raises(ValueError): s.source_bank_equivalence(scene,changed)


def test_external_row_metrics_reject_self_consistent_wrong_labels(scene):
    expected=[scene.rows[i] for i in scene.indices(scene.targets,'validation')]
    rows=[dict(r,prediction=r['label']) for r in expected]
    assert d.metrics(rows,expected)['macro_rmse']==0
    rows[0]['label']+=1; rows[0]['prediction']+=1
    with pytest.raises(ValueError): d.metrics(rows,expected)


@pytest.mark.parametrize('name',['../a','/a','C:/a','a\\b','a/../b'])
def test_input_paths_cannot_escape_root(tmp_path,name):
    with pytest.raises(ValueError): d.safe_file(tmp_path,name)


def test_forged_trusted_input_hash_fails_before_tensor_loading(tmp_path,monkeypatch):
    (tmp_path/'asset').write_text('wrong',encoding='utf-8')
    lock=tmp_path/'lock.json'
    d.write(lock,dict(schema='v12_s1_input_lock_v1',reference_commit=d.CHEMICAL_COMMIT,files={'asset':'0'*64}))
    monkeypatch.setattr(d,'LOCK',lock)
    with pytest.raises(ValueError,match='SHA'): d.check_inputs(tmp_path)


def test_full_synthetic_run_and_independent_replay_rejects_forgery(scene,tmp_path,monkeypatch):
    cases={name:deepcopy(scene) for name in d.SCENES}
    for name,item in cases.items(): item.name=name
    identity=dict(commit='a'*40,spec=s.SPEC)
    fake_preflight=dict(identity=identity)
    monkeypatch.setattr(s,'preflight',lambda *args:fake_preflight)
    monkeypatch.setattr(d,'load_scene',lambda root,name:cases[name])
    monkeypatch.setattr(s,'historical_reference',lambda root,item:dict(fixture=True,scene=item.name))
    monkeypatch.setitem(s.SPEC,'target_endpoint_model_fits',120)
    monkeypatch.setitem(s.SPEC,'source_task_prior_fits',6)
    out=tmp_path/'run'
    s.run(tmp_path,out,'a'*40)
    assert s.verify(tmp_path,out,'a'*40)['target_endpoint_fits']==120
    saved=out/'analytic/ToxAcute/C3_c0_full.json'
    value=d.read(saved); value['models'][scene.targets[0]]['linear']['coefficient'][0]+=1
    saved.write_text(json.dumps(value),encoding='utf-8')
    # An attacker also updating the file manifest still cannot pass scientific replay.
    checksum=d.read(out/'checksums.json'); checksum[saved.relative_to(out).as_posix()]=d.sha(saved)
    (out/'checksums.json').write_text(json.dumps(checksum),encoding='utf-8')
    with pytest.raises(ValueError,match='replay'): s.verify(tmp_path,out,'a'*40)
