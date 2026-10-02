from copy import deepcopy
import pytest
import torch

import v9_adaptive_transfer as m
import v9_s3_screen as s3
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory


def test_ten_substantive_candidates_exclude_controls_and_hyperparameters():
    assert len(m.CANDIDATES) == len(set(m.CANDIDATES)) == 10
    assert not set(m.CANDIDATES) & set(m.CONTROLS)
    assert len(m.GRADIENT_METHODS) == 5


def test_scaffold_split_label_blind_exhaustive_and_scaler_fit_only():
    rows = [dict(task=t, sample_id=str(i), canonical=str(i), group=str(i//2), label=float(i), split='train')
            for t in ('a', 'b') for i in range(20)]
    split = m.partition(rows); changed = deepcopy(rows)
    for r in changed:
        r['label'] = 10000 if r['group'] in split['groups'] else r['label']
    assert m.partition(changed)['meta'] == split['meta']
    assert m.inner_scalers(changed, m.partition(changed)) == m.inner_scalers(rows, split)
    for t in split['fit']:
        assert set(split['fit'][t]).isdisjoint(split['meta'][t])
        assert sorted(split['fit'][t]+split['meta'][t]) == list(range(20))
    assert m.fit_plan([dict(task='a', indices=split['meta']['a'])], split)[0]['indices']


def test_partition_fails_closed_on_impossible_or_bad_canonical():
    rows = [dict(task='x', group='a', canonical='CC', split='train', sample_id='0', label=1.)]
    with pytest.raises(ValueError): m.partition(rows)
    rows.append(dict(rows[0], group='b', sample_id='1'))
    with pytest.raises(ValueError, match='canonical'): m.partition(rows)


def test_projected_cosine_geometry_is_rotation_invariant_and_zero_safe():
    g = torch.tensor([[1., 0., 0.], [0., 2., 0.]])
    x = torch.tensor([[1., 0., 10.], [-1., -1., 0.]])
    score = m.subspace_scores(x, g)
    q, _ = torch.linalg.qr(torch.tensor([[1., 2., 3.], [2., -1., 1.], [0., 3., -2.]], dtype=torch.double))
    assert torch.allclose(score, m.subspace_scores(x.double()@q, g.double()@q), atol=1e-7)
    assert score[0] == 1 and score[1] < 0
    assert torch.equal(m.subspace_scores(x, torch.zeros_like(g)), torch.zeros(2, dtype=torch.double))
    assert m.hard_weights(torch.tensor([-.1, .9, .8, .7]), [True, False, True, True]) == [0., 0., .5, .5]


def test_ties_sign_election_global_density_and_lines_layer_schedule():
    initial = {'encoder.weight': torch.zeros(4)}
    first = {'encoder.weight': torch.tensor([5., -4., 1., 0.])}
    second = {'encoder.weight': torch.tensor([-3., -2., 1., 0.])}
    got = m.ties_merge(initial, first, second, {'encoder.weight': True, 'target_tasks': []}, density=.5)
    assert torch.equal(got['encoder.weight'], torch.tensor([5., -3., 0., 0.]))
    a = {f'encoder.backbone.layers.{i}.ffn.weight': torch.zeros(1) for i in range(3)}
    b = {k: torch.ones(1) for k in a}
    assert [float(x) for x in m.interpolate(a, b, 1., depth=True).values()] == [0., .5, 1.]


def test_protected_gate_rejects_harm_and_zero_exactly_recovers_target():
    a = [dict(task='t', sample_id=str(i), group=str(i), label=0., prediction=1.) for i in range(5)]
    bad = [dict(r, prediction=2.) for r in a]; good = [dict(r, prediction=0.) for r in a]
    assert m.gate_coefficients(a, bad) == {'t': 0.}
    assert 0 < m.gate_coefficients(a, good)['t'] < 1
    assert m.blend_rows(a, bad, {'t': 0.}) == a


@pytest.mark.parametrize('setting', s3.s2.c0.SETTINGS)
def test_zero_auxiliary_exact_target_path_and_private_head_no_leak(setting, joint):
    f, t, source, _ = joint(setting); base = deepcopy(t.model)
    tasks, counts = s3.s2.c0.tasks_and_counts(setting)
    item = s3.s2.epoch_plan(tasks, counts, 32, 0)[0]
    batch = s3.s2.batch_for(f, t, setting, 'train', item['task'], item['indices'], 'cpu')
    ai = s3.aux.Schedule(source.tasks, source.counts, 32).next()
    scaler = (t.task_scalers if setting == 'ToxAcute' else t.scalers)[item['task']]
    results = []
    for kind in ('TARGET_FULL', 'JOINT_FULL'):
        model, opt = m.build(base, source, kind, .0001, 'cpu')
        before = m.state(model)
        m.update(model, opt, batch, item['task'], scaler, source, ai, [0.]*len(ai['indices']), 'cpu', 1)
        after = m.state(model)
        assert all(torch.equal(v, after[k]) for k, v in before.items() if k.startswith('decoders.aux_'))
        results.append(after)
    assert all(torch.equal(results[0][k], v) for k, v in results[1].items())


@pytest.mark.parametrize('kind', m.GRADIENT_METHODS)
def test_controller_finite_bounded_restores_virtual_update_and_preserves_rng(kind, joint):
    f, t, source, _ = joint('A'); model, _ = m.build(t.model, source, kind, .001, 'cpu')
    task = next(iter(t.datasets['train'])); batch = s3.s2.batch_for(f, t, 'A', 'train', task, [0], 'cpu')
    ai = s3.aux.Schedule(source.tasks, source.counts, 32).next()
    ab = source.batch(ai['task'], ai['indices'], 'cpu'); before = m.state(model); rng = torch.get_rng_state()
    result = m.learn_weights(model, kind, batch, task, t.scalers[task], ab, ai['task'], source.scalers[ai['task']],
                            [(task, batch, t.scalers[task])], [True]*len(ai['indices']), {}, .001)
    assert len(result) == len(ai['indices']) and 0 <= sum(result) <= 1.000001
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('profile', ('TARGET_FROZEN', 'TARGET_LAST'))
def test_plasticity_parameter_and_dropout_boundaries(profile, joint):
    f, t, source, _ = joint('A'); model, _ = m.build(t.model, source, profile, .0001, 'cpu')
    m.training_mode(model)
    if profile == 'TARGET_FROZEN':
        assert not model.encoder.training and not any(p.requires_grad for p in model.encoder.parameters())
    else:
        assert model.encoder.backbone.layers[-1].training
        last = len(model.encoder.backbone.layers)-1
        assert all(not p.requires_grad for n, p in model.encoder.named_parameters()
                   if not n.startswith((f'backbone.layers.{last}.', 'backbone.final_norm.', 'readout.')))


@pytest.mark.parametrize('setting', s3.s2.c0.SETTINGS)
def test_individual_quantile_loss_preserves_frozen_training_objective(setting, joint):
    f, t, source, _ = joint(setting); t.model, _ = m.build(t.model, source, 'TARGET_FULL', .0001, 'cpu')
    task = t.model.target_tasks[0]; batch = s3.s2.batch_for(f, t, setting, 'train', task, [0], 'cpu')
    scaler = (t.task_scalers if setting == 'ToxAcute' else t.scalers)[task]
    t.model.eval()
    ours = m.losses(t.model, batch, task, scaler).mean()
    old = (t._training_step(batch, task, 0)[0] if setting == 'ToxAcute' else
           m.QuantileRegressionLoss().compute_loss(t.model(batch, task_name=task)[task],
                (batch.y.reshape(-1, 1)-scaler['mean'])/scaler['std']))
    assert torch.equal(ours, old)
