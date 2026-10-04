from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import d2_control as core
from d2_data import Data
from dataset115_adapter import GraphTaskView, TrainOnlyScaler
from reproducibility import state_dict_sha256
from tests.test_dataset115_adapter import args, view
from tests.test_dataset115_training import validation_view
from v9_joint_source import Source


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def data_factory():
    def make(condition, setting='A', seed=42):
        a = args()
        train, val = view(), validation_view()
        targets = list(train.tasks)
        if setting == 'ToxAcute':
            targets = ['man_oral_TDLo', 'women_oral_TDLo', 'human_oral_TDLo']
        datasets = {s: {t: GraphTaskView(v, t) for t in targets} for s, v in [('train', train), ('validation', val)]}
        rows = {s: [dict(task=t, sample_id=ds.get_sample_id(i), split=s, canonical=ds.view.canonical[j],
                         group=ds.view.groups[j], label=float(ds.view.labels[j, ds.view.tasks.index(t)]))
                    for t, ds in group.items() for i, j in enumerate(ds.indices)] for s, group in datasets.items()}
        scalers = {t: v for t, v in TrainOnlyScaler.fit(train).trainer_scalers().items() if t in targets}
        sv = replace(train, route='A', role='source', tasks=('rat_oral_LD50', 'mouse_oral_LD50'),
                     sample_ids=('dataset115:row_8', 'dataset115:row_9'),
                     smiles=('CCO', 'CCN'), canonical=('CCO', 'CCN'), groups=('CCO', 'CCN'),
                     labels=np.array([[1., 3.], [3., 5.]]))
        sd = {t: GraphTaskView(sv, t) for t in sv.tasks}
        sr = [dict(task=t, sample_id=ds.get_sample_id(i), split='train', canonical=sv.canonical[j],
                   group=sv.groups[j], label=float(sv.labels[j, sv.tasks.index(t)]))
              for t, ds in sd.items() for i, j in enumerate(ds.indices)]
        source = None
        if condition in ('FJ', 'PJ', 'RJ'):
            source = Source(sd, sr, torch.nn.ModuleDict({t: torch.nn.Linear(1, 1) for t in sd}),
                            TrainOnlyScaler.fit(sv).trainer_scalers(), dict(synthetic=True))
            # Audit identity must not depend on this irrelevant fixture-head RNG.
            source.identity['heads_sha256'] = 'fixture_only'
        pretrained = core.encoder(a, seed+100).state_dict()
        return Data(args=a, pretrained=pretrained, target_datasets=datasets, target_rows=rows,
                    target_scalers=scalers, source_tasks=list(sd), source_counts={t: len(ds) for t, ds in sd.items()},
                    source=source, identity=dict(fixture=True, setting=setting, condition=condition, seed=seed))
    return make


def small_spec():
    return dict(core.SPEC, source_passes=2, source_batch_size=1, target_batch_size=1,
                validation_every=2, early_validation=[1, 2])


@pytest.mark.parametrize('seed', range(42, 47))
def test_five_conditions_pair_heads_and_distinguish_true_initialization(data_factory, seed):
    models = {}
    for condition in core.CONDITIONS:
        data = data_factory(condition, seed=seed)
        j = dict(condition=condition, seed=seed)
        models[condition] = data.model(j, 'cpu')
    assert len({state_dict_sha256(m.decoders) for m in models.values()}) == 1
    assert len({state_dict_sha256(models[c].encoder) for c in ('FJ', 'PT', 'PJ')}) == 1
    assert state_dict_sha256(models['RT'].encoder) == state_dict_sha256(models['RJ'].encoder)
    assert state_dict_sha256(models['RT'].encoder) != state_dict_sha256(models['PJ'].encoder)
    assert all(p.requires_grad is False for p in models['FJ'].encoder.parameters())
    assert all(p.requires_grad for p in models['PJ'].encoder.parameters())


@pytest.mark.parametrize('condition', core.CONDITIONS)
def test_real_molecular_forward_backward_and_parameter_changes(data_factory, condition):
    data = data_factory(condition)
    job = next(j for j in core.jobs() if j['condition'] == condition)
    model = data.model(job, 'cpu')
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = core.optimizer(model, job['encoder_lr'])
    item = next(core.schedule(data.target_counts, data.source_counts, 42, small_spec()))
    row = core.update(model, opt, data, item, 42, 'cpu')
    g = row['gradient_l1']
    assert g['target'] > 0 and (g['source'] > 0) == model.joint
    assert (g['encoder'] > 0) == (condition != 'FJ')
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert any(n.startswith('encoder.') for n in changed) == (condition != 'FJ')
    assert any(n.startswith('decoders.'+item['source']['task']+'.') for n in changed) == model.joint
    assert model.encoder.training  # Pair dropout even with frozen weights.


def test_random_model_independent_of_every_source_tensor(data_factory):
    data = data_factory('RJ')
    j = dict(condition='RJ', seed=42)
    first = data.model(j, 'cpu')
    for v in data.pretrained.values():
        v.add_(100)
    second = data.model(j, 'cpu')
    assert state_dict_sha256(first) == state_dict_sha256(second)


def test_complete_source_passes_preserve_tails_and_target_stream():
    source = dict(a=65, b=2, c=9)
    target = dict(x=3, y=18)
    spec = small_spec()
    spec.update(source_batch_size=32, target_batch_size=8, source_passes=3)
    plan = list(core.schedule(target, source, 42, spec))
    assert plan == list(core.schedule(target, source, 42, spec))
    assert plan != list(core.schedule(target, source, 43, spec))
    for epoch in range(1, 4):
        for task, n in source.items():
            assert sorted(i for p in plan if p['source_pass'] == epoch and p['source']['task'] == task
                          for i in p['source']['indices']) == list(range(n))
    for epoch in {p['target_pass'] for p in plan[:-1]}:
        group = [p for p in plan if p['target_pass'] == epoch]
        if epoch == plan[-1]['target_pass']:
            continue
        for task, n in target.items():
            assert sorted(i for p in group if p['target']['task'] == task for i in p['target']['indices']) == list(range(n))


@pytest.mark.parametrize('condition', ['PT', 'RT'])
def test_target_only_cannot_request_source_batch_or_loss(data_factory, condition):
    data = data_factory(condition)
    with pytest.raises(ValueError, match='forbidden'):
        data.batch('source', 'train', data.source_tasks[0], [0], 'cpu')
    job = dict(condition=condition, seed=42)
    model = data.model(job, 'cpu')
    with pytest.raises(ValueError, match='forbidden'):
        model(None, data.source_tasks[0])


def test_target_rng_unaffected_by_source_branch(data_factory):
    rngs = []
    for condition in ('PT', 'PJ'):
        data = data_factory(condition)
        model = data.model(dict(condition=condition, seed=42), 'cpu')
        opt = core.optimizer(model, .001)
        item = next(core.schedule(data.target_counts, data.source_counts, 42, small_spec()))
        torch.manual_seed(12345)
        row = core.update(model, opt, data, item, 42, 'cpu')
        rngs.append((torch.get_rng_state(), row['losses']['target']))
    assert torch.equal(rngs[0][0], rngs[1][0]) and rngs[0][1] == rngs[1][1]


def test_bounded_matrix_and_early_selection():
    assert len(core.jobs()) == 27 and {j['seed'] for j in core.jobs()} == {42}
    assert len([j for j in core.jobs() if j['condition'] == 'FJ']) == 3
    assert core.SPEC['source_passes'] == 40
    selected = core.select([dict(step=i, metrics=dict(macro_rmse=v)) for i, v in [(1, 2.), (2, 1.), (3, 1.)]])
    assert selected['step'] == 2
    with pytest.raises(ValueError):
        core.check_job(dict(core.jobs()[0], seed=43))


def test_no_cross_lr_or_cross_scene_winner_collage():
    results = [dict(job=j, selected=dict(metrics=dict(macro_rmse=1.))) for j in core.jobs()]
    r = core.paired_report(results)
    assert len(r['rows']) == 6 and all(v == 0 for row in r['rows'] for v in row['paired_differences'].values())
    assert not r['automatic_replication_authorized'] and not r['unified_superiority_confirmed']
    with pytest.raises(ValueError):
        core.paired_report(results[:-1])
