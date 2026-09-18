from __future__ import annotations
import json
import hashlib
import numpy as np
import pytest
import torch

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS
from dataset115_adapter import LabelView
from dataset115_contract import ContractError, PRIMARY
from baselines.b115_data import JointTrain, assemble, check_bytes, prepare, verify_permissions
from baselines.b115_training import (configuration, epoch_candidates, is_improvement,
    masked_loss, train_engine, validation_metrics)
from baselines.models.toxacol import ToxACoLNet, endpoint_feature_matrix, task_adjacency


@pytest.fixture(autouse=True)
def small_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def target(route='B', split='train'):
    return LabelView(route, 'target', split, PRIMARY,
        ('dataset115:row_0', 'dataset115:row_1'), ('C', 'CC'), ('C', 'CC'), ('g0', 'g1'),
        np.array([[1., 2., 3., 4., 5.], [np.nan]*5]), 'a'*64)


def joint():
    tasks = tuple(ANIMAL_SOURCE_TASKS)
    return assemble('B', tasks, ('toxacute:row_0', 'toxacute:row_1'), ('N', 'NN'), ('N', 'NN'),
                    np.array([[1.]*56, [2.]*56]), target())


def test_namespace_zero_mask_and_forbidden_label_blocks():
    view = joint()
    assert view.sample_ids == ('toxacute:row_0', 'toxacute:row_1', 'dataset115:row_0')
    assert np.isnan(view.labels[:2, -5:]).all()
    assert np.isnan(view.labels[2, :-5]).all()
    assert view.labels[2, -5:].tolist() == [1., 2., 3., 4., 5.]


@pytest.mark.parametrize('mutation', ['Human3', 'Nonhuman104', 'validation', 'namespace', 'overlap'])
def test_permission_rejections(mutation):
    tasks = tuple(ANIMAL_SOURCE_TASKS)
    ids = ('toxacute:row_0',)
    canonical = ('N',)
    t = target()
    if mutation == 'Human3':
        tasks = tasks[:-1] + ('human_oral_TDLo',)
    if mutation == 'Nonhuman104':
        tasks = tuple(f's{i}_oral_LD50' for i in range(104))
    if mutation == 'validation':
        t = target(split='validation')
    if mutation == 'namespace':
        ids = ('row_0',)
    if mutation == 'overlap':
        canonical = ('C',)
    with pytest.raises(ContractError):
        assemble('B', tasks, ids, ('N',), canonical, np.ones((1, len(tasks))), t)


def test_route_a_forbids_extra_human_tasks():
    tasks = tuple(f'animal{i}_oral_LD50' for i in range(103)) + ('child_skin_TDLo',)
    t = target('A')
    with pytest.raises(ContractError, match='Nonhuman104'):
        assemble('A', tasks, t.sample_ids, t.smiles, t.canonical, np.ones((2,104)), t)


def test_route_a_aligns_and_drops_only_empty_rows():
    tasks = tuple(f'animal{i}_oral_LD50' for i in range(104))
    t = target('A')
    labels = np.ones((2,104)); labels[1] = np.nan
    view = assemble('A', tasks, t.sample_ids, t.smiles, t.canonical, labels, t)
    assert view.labels.shape == (1,109)
    with pytest.raises(ContractError, match='aligned'):
        assemble('A', tasks, t.sample_ids[::-1], t.smiles, t.canonical, labels, t)


def test_exact_observations_not_just_counts(tmp_path):
    view = joint()
    rows = [dict(sample_id=sid, task=view.tasks[j], label=float(view.labels[i,j]),
                 role='target' if view.tasks[j] in PRIMARY else 'source')
            for i,sid in enumerate(view.sample_ids) for j in np.flatnonzero(np.isfinite(view.labels[i]))]
    p = tmp_path / 'allowed.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf8')
    verify_permissions(view, p)
    rows[0]['label'] += .5
    p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf8')
    with pytest.raises(ContractError, match='permissions'):
        verify_permissions(view, p)
    with pytest.raises(ContractError, match='SHA'):
        check_bytes(p, '0'*64)


def test_prepare_uses_train_only_and_no_cross_source_edges():
    view = joint()
    adj, features, scaler, contract = prepare(view)
    assert not adj[:-5,-5:].any()
    assert features.shape == (61,27)
    np.testing.assert_array_equal(scaler.means[:56], [1.5]*56)
    np.testing.assert_array_equal(scaler.means[-5:], [1,2,3,4,5])
    assert contract['observations'] == 117


def test_graph_shared_threshold_and_constant():
    y = np.column_stack([np.arange(15), np.arange(15), np.ones(15)]).astype(float)
    adj, audit = task_adjacency(y)
    assert audit['undirected_edges'] == 1
    np.testing.assert_allclose(adj, [[.5,.5,0],[.5,.5,0],[0,0,1]], atol=1e-7)
    assert task_adjacency(y[:-1])[1]['undirected_edges'] == 0
    y[:,1] *= -1
    assert task_adjacency(y)[1]['undirected_edges'] == 0


def test_source_infinity_cannot_disappear_with_empty_mask_rows():
    with pytest.raises(ContractError, match='nonfinite'):
        assemble('B', tuple(ANIMAL_SOURCE_TASKS), ('toxacute:row_0',), ('N',), ('N',),
                 np.full((1,56),np.inf), target())


def test_duplicate_canonical_task_requires_decision():
    view = joint()
    repeated = JointTrain(view.route,view.tasks,view.sample_ids,view.smiles,
                          ('N','N','C'),view.labels,view.input_identity)
    with pytest.raises(ContractError,match='repeated canonical'):
        prepare(repeated)


@pytest.mark.parametrize('tasks,width', [(61,27),(109,36)])
def test_generic_forward_backward(tasks, width):
    torch.manual_seed(4)
    model = ToxACoLNet(np.eye(tasks), np.ones((tasks,width)))
    x = torch.randn(2,1024,requires_grad=True)
    out = model(x)
    assert out.shape == (2,tasks)
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.tail_weight.grad).all()


def test_legacy_features_and_new_vocab_are_deterministic():
    old, schema = endpoint_feature_matrix()
    extended, schema2 = endpoint_feature_matrix(extend_vocabulary=True)
    np.testing.assert_array_equal(old, extended)
    assert schema == schema2 and old.shape == (59,26)
    _, b = endpoint_feature_matrix(tuple(ANIMAL_SOURCE_TASKS)+PRIMARY, extend_vocabulary=True)
    assert b['species'][:-1] == schema['species'] and b['species'][-1] == 'child'


@pytest.mark.parametrize('bad', ['shape','nan','asymmetric'])
def test_model_graph_rejects(bad):
    a = np.eye(3)
    if bad == 'shape': a = np.eye(4)
    if bad == 'nan': a[0,0] = np.nan
    if bad == 'asymmetric': a[0,1] = 1
    with pytest.raises(ValueError): ToxACoLNet(a, np.ones((3,26)))


def test_masked_nan_and_failed_prediction():
    p = torch.tensor([[1.,2.]], requires_grad=True)
    t = torch.tensor([[0.,float('nan')]])
    mask = torch.tensor([[True,False]])
    loss = masked_loss(p,t,mask); loss.backward()
    assert float(loss.detach()) == 1 and p.grad.tolist() == [[2.,0.]]
    with pytest.raises(ValueError): masked_loss(p*float('nan'),t,mask)
    with pytest.raises(ValueError): masked_loss(p,t,torch.zeros_like(mask))


def test_validation_complete_finite_and_tie():
    labels = np.ones((2,5))
    assert validation_metrics(labels, labels)['macro_rmse'] == 0
    labels[:,0] = np.nan
    with pytest.raises(ValueError): validation_metrics(labels,np.ones((2,5)))
    with pytest.raises(ValueError): validation_metrics(np.ones((2,5)), np.full((2,5),np.nan))
    assert not is_improvement(1., 1.)
    assert is_improvement(.9, 1.)


def test_epoch_windows_and_config():
    assert epoch_candidates(0) == list(range(10))
    assert epoch_candidates(119) == list(range(119,109,-1))
    assert epoch_candidates(5) == [5,4,6,3,7,2,8,1,9,0]
    with pytest.raises(ValueError): epoch_candidates(-1)
    with pytest.raises(ValueError): configuration(seed=99)


def training_args():
    from baselines.scaling import TaskScaler
    rng = np.random.RandomState(8)
    tasks = tuple(ANIMAL_SOURCE_TASKS)+PRIMARY
    labels = rng.normal(size=(33,len(tasks)))
    scaler = TaskScaler.fit(labels,tasks,allow_empty=False)
    features,_ = endpoint_feature_matrix(tasks,extend_vocabulary=True)
    return dict(train_x=rng.normal(size=(33,1024)).astype(np.float32), train_labels=labels,
        validation_x=rng.normal(size=(3,1024)).astype(np.float32), validation_labels=rng.normal(size=(3,5)),
        adjacency=np.eye(61),endpoint_features=features,scaler=scaler,
        contract={'tasks':list(tasks), 'role':'SYNTHETIC'},config=configuration())


def test_checkpoint_resume_matches_uninterrupted_with_singleton(tmp_path):
    kw = training_args()
    full = train_engine(**kw, output=tmp_path/'full',stop_after_epoch=2)
    train_engine(**kw, output=tmp_path/'first',stop_after_epoch=1)
    sha = hashlib.sha256((tmp_path/'first/epoch_000.pt').read_bytes()).hexdigest()
    resumed = train_engine(**kw,output=tmp_path/'resumed',stop_after_epoch=2,
                            resume=tmp_path/'first/epoch_000.pt', resume_sha256=sha)
    assert full['history'] == resumed['history'] and full['updates'] == 4
    p1 = torch.load(tmp_path/'full/epoch_001.pt',weights_only=True)
    p2 = torch.load(tmp_path/'resumed/epoch_001.pt',weights_only=True)
    for key in p1['model']: torch.testing.assert_close(p1['model'][key],p2['model'][key],rtol=0,atol=0)
    np.testing.assert_array_equal(np.load(tmp_path/'full/validation_001.npy'),np.load(tmp_path/'resumed/validation_001.npy'))
    kw['train_x'][0,0] += 1
    with pytest.raises(ValueError,match='mismatch'):
        train_engine(**kw,output=tmp_path/'bad',resume=tmp_path/'first/epoch_000.pt',
                     resume_sha256=sha,stop_after_epoch=2)
    assert not (tmp_path/'bad').exists()


def test_run_refuses_overwrite_and_config_drift(tmp_path):
    kw = training_args()
    train_engine(**kw,output=tmp_path/'out',stop_after_epoch=1)
    with pytest.raises(FileExistsError): train_engine(**kw,output=tmp_path/'out',stop_after_epoch=1)
    kw['config']['epochs'] = 2
    with pytest.raises(ValueError,match='configuration'): train_engine(**kw,output=tmp_path/'bad')


def test_resume_requires_hash_and_valid_selection(tmp_path):
    kw = training_args()
    train_engine(**kw,output=tmp_path/'first',stop_after_epoch=1)
    path = tmp_path/'first/epoch_000.pt'
    with pytest.raises(ValueError,match='SHA'):
        train_engine(**kw,output=tmp_path/'missing',resume=path,stop_after_epoch=2)
    payload = torch.load(path,weights_only=True)
    payload['best'] = -123
    tampered = tmp_path/'tampered.pt'
    torch.save(payload,tampered)
    sha = hashlib.sha256(tampered.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='selection'):
        train_engine(**kw,output=tmp_path/'bad',resume=tampered,resume_sha256=sha,stop_after_epoch=2)


def test_verifier_rejects_nonstandard_json(tmp_path):
    from scripts.verify_b115_smoke import read_json
    path=tmp_path/'bad.json'
    path.write_text('{"x":NaN}',encoding='utf8')
    with pytest.raises(ValueError,match='non-standard'):
        read_json(path)
