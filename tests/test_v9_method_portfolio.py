from copy import deepcopy
import pytest
import torch

import v9_method_portfolio as mod
import v9_s3_screen as s3
from tests.test_v9_s3_screen import joint
from tests.test_v9_s2_screen import runtime2
from tests.test_v9_screen import runtime, one_cpu_thread, setup_factory


@pytest.mark.parametrize('kind', mod.METHODS)
def test_initial_predictions_exact_and_supervision_routes(kind, joint):
    f, t, source, _ = joint('ToxAcute'); base = deepcopy(t.model)
    tasks, counts = s3.s2.c0.tasks_and_counts('ToxAcute')
    batch = s3.s2.batch_for(f, t, 'ToxAcute', 'train', tasks[0], [0, 1], 'cpu')
    state = torch.get_rng_state()
    model, optimizer = mod.build(base, source, kind, .001, 'cpu')
    assert torch.equal(state, torch.get_rng_state())
    model.eval(); base.eval()
    with torch.no_grad():
        assert torch.equal(model(batch, task_name=tasks[0])[tasks[0]], base(batch, task_name=tasks[0])[tasks[0]])
    model.train(); model.zero_grad(set_to_none=True)
    source_task = source.tasks[0]
    model(batch, task_name=source_task)[source_task].square().mean().backward()
    assert all(p.grad is None for p in model.private.parameters())
    assert all(p.grad is None for p in model.decoders[tasks[0]].parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.decoders[source_task].parameters() if p.grad is not None) > 0
    if kind not in mod.FULL:
        assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    else:
        assert sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None) > 0


@pytest.mark.parametrize('kind', mod.METHODS)
def test_two_updates_change_intended_parameters_and_isolate_auxiliary_rng(kind, joint):
    f, t, source, _ = joint('A')
    t.model, optimizer = mod.build(t.model, source, kind, .001, 'cpu'); t.model.train()
    before = {n: p.detach().clone() for n, p in t.model.named_parameters()}
    tasks, counts = s3.s2.c0.tasks_and_counts('A')
    task = tasks[0]; batch = s3.s2.batch_for(f, t, 'A', 'train', task, [0, 1], 'cpu')
    schedule = s3.aux.Schedule(source.tasks, source.counts, 32)
    for i in range(2):
        r = mod.step(t, 'A', batch, task, 0, source, schedule.next(), optimizer, 'cpu')
        assert r['target'] >= 0 and r['auxiliary'] >= 0 and r['gradient_norms']['source'] > 0
    assert any(not torch.equal(p, before[n]) for n, p in t.model.named_parameters() if not n.startswith('decoders.'))
    assert all(torch.equal(p, before[n]) for n, p in t.model.named_parameters() if n.startswith('decoders.'+tasks[-1]+'.'))
    if kind not in mod.FULL:
        assert all(torch.equal(p, before[n]) for n, p in t.model.named_parameters() if n.startswith('encoder.'))
    if kind == 'TPRS':
        assert all(torch.equal(p, before[n]) for n, p in t.model.named_parameters() if n.startswith('private.'+tasks[-1]+'.'))
        assert any(not torch.equal(p, before[n]) for n, p in t.model.named_parameters() if n.startswith('private.'+task+'.'))
    assert bool(mod.algorithm_state(optimizer)) == (kind == 'DB_MTL')


@pytest.mark.parametrize('vectors', [([1., 2.], [-3., 1.]), ([1., 0.], [2., 0.]), ([0., 0.], [2., 3.]), ([1., 0.], [-1., 0.])])
def test_config_agrees_with_pseudoinverse_and_degenerate_behavior(vectors):
    a, b = [torch.tensor(v, dtype=torch.float64) for v in vectors]
    actual = mod.config_gradient([a], [b])[0]
    if a.norm() == 0 or b.norm() == 0:
        assert torch.equal(actual, a+b)
    else:
        matrix = torch.stack([a/a.norm(), b/b.norm()])
        u = torch.linalg.pinv(matrix)@torch.ones(2, dtype=a.dtype)
        if u.norm() < 1e-8:
            expected = torch.zeros_like(a)
        else:
            u = u/u.norm(); expected = ((a+b)@u)*u
        assert torch.allclose(actual, expected, atol=1e-10)
    assert torch.isfinite(actual).all() and a@actual >= -1e-10 and b@actual >= -1e-10


def test_projection_target_guarantee_norm_cap_and_missing_gradients():
    a = torch.tensor([1., 0.]); b = torch.tensor([-10., 5.])
    merged, r = mod.priority.merge_block([a, None], [b, None])
    assert torch.allclose(merged[0], torch.tensor([1., 1.])) and merged[1] is None
    assert r['dot_before'] < 0 and r['dot_after'] == 0 and r['auxiliary_norm_after'] == 1
    result, r = mod.priority.merge_block([torch.zeros(2)], [b])
    assert torch.count_nonzero(result[0]) == 0
    with pytest.raises(ValueError, match='finite'):
        mod.priority.merge_block([a], [torch.tensor([float('nan'), 0.])])


def test_orthogonal_projection_is_symmetric_and_retains_nonconflicting():
    a = torch.tensor([1., 0.]); b = torch.tensor([-1., 1.])
    assert torch.allclose(mod.symmetric_projection([a], [b])[0], torch.tensor([.5, 1.5]))
    assert torch.equal(mod.symmetric_projection([a], [a])[0], a+a)


def test_alignment_molecule_equal_padding_and_permutation_invariance():
    x = torch.tensor([[[100.], [1.], [3.], [999.]], [[100.], [4.], [999.], [999.]]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    means = mod.atom_mean(x, mask)
    assert torch.equal(means, torch.tensor([[2.], [4.]]))
    assert mod.alignment({'x': means}, {'x': means}) == 0
    assert torch.allclose(mod.alignment({'x': means}, {'x': means+1}), mod.alignment({'x': means+1}, {'x': means}))


def test_mtl_task_specific_transform_and_masa_cross_layer_sharing(joint):
    _, t, a, _ = joint('A')
    model, _ = mod.build(t.model, a, 'MTL_LORA', .001, 'cpu')
    layer = next(iter(model.adapters.values()))
    assert layer.task_transform.shape[0] == len(model.task_name)
    with torch.no_grad():
        layer.B.normal_(); layer.task_transform[1].mul_(2)
    x = torch.randn(2, 4, layer.A.shape[-1]); mask = torch.ones(2, 3, dtype=torch.bool)
    assert not torch.equal(layer(x, x, 0, mask)[0], layer(x, x, 1, mask)[0])
    model, _ = mod.build(t.model, a, 'MASA', .001, 'cpu')
    assert len(model.bank) == 4*((len(model.encoder.backbone.layers)+1)//2)


@pytest.mark.parametrize('kind', ['ASE', 'MTLORA_BLOCK', 'LIME'])
def test_expert_routing_is_finite_and_atom_permutation_equivariant(kind):
    layer = mod.Adapter(8, 8, 3, kind)
    with torch.no_grad():
        layer.B.normal_(std=.01)
    x = torch.randn(2, 4, 8); base = torch.randn_like(x); mask = torch.ones(2, 3, dtype=torch.bool)
    perm = [0, 3, 1, 2]
    a, ra, _ = layer(x, base, 1, mask)
    b, rb, _ = layer(x[:, perm], base[:, perm], 1, mask)
    assert torch.allclose(a[:, perm], b, atol=1e-6) and torch.allclose(ra, rb, atol=1e-6)
    (a.square().mean()+ra).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
