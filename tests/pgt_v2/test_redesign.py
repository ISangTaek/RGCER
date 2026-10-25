"""Only the new scientific paths: prompt gradients, isolation and target selection."""
import numpy as np
import pytest
import torch
from rdkit import Chem

from pgt_v2.chemistry import graph_from_input, prepare_molecule
from pgt_v2.config import candidate
from pgt_v2.evaluation import load_model_checkpoint
from pgt_v2.minipilot import SOURCES, TARGETS, select_pilot_arrays
from pgt_v2.models import build_model, parameter_counts, task_metadata
from pgt_v2.redesign_models import relation_prior, task_smoother
from pgt_v2.training import ModelInputs, endpoint_loss, seed_everything, train_model, read_checkpoint

TASKS = ('mouse_oral_LD50', 'rat_oral_LD50', 'human_oral_TDLo')


def tiny_config(method):
    return {**candidate(method, 'C2'), 'hidden_dim': 16, 'ffn_dim': 24, 'heads': 4,
            'layers': 2, 'dropout': 0.0, 'max_epochs': 2, 'min_epochs': 2,
            'patience': 2, 'effective_batch': 3, 'micro_batch': 2, 'selection_tasks': [TASKS[-1]]}


def tiny_inputs(method):
    smiles = ('CO', 'CC(O)C', 'C')
    graphs = [graph_from_input(prepare_molecule(s), sample_id=f'r{i}', raw_row_index=i)
              for i, s in enumerate(smiles)]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    class Cache:
        def read(self, i):
            return graphs[i], mols[i]
    records = dict(record_id=np.array([f'r{i}' for i in range(3)]), cache_offset=np.arange(3),
        raw_row_index=np.arange(3), chemical_id=np.array([g.model_input_id for g in graphs]),
        group_id=np.array(['g0', 'g1', 'g2']))
    return ModelInputs(Cache(), records, method)


@pytest.mark.parametrize('method', ['N0', 'N1', 'N2', 'N3', 'N4'])
def test_new_models_learn_prompts_and_preserve_padding_and_record_order(method):
    seed_everything(42)
    model = build_model(method, TASKS, tiny_config(method), counts=[512, 512, 9]).eval()
    inputs = tiny_inputs(method)
    x = inputs.batch([0, 1, 2])
    y, aux = model(x, return_aux=True)
    assert y.shape == (3, 3) and torch.isfinite(y).all()
    loss = endpoint_loss(y, torch.tensor([[1., -1., 2.], [.2, 1., float('nan')], [-1., 2., -.1]]), [3, 3, 2], 3)
    loss = loss + model.regularization_loss()
    loss.backward()
    prompt_parameter = model.prompt_ids if method == 'N1' else model.prompt_bank.residual
    assert prompt_parameter.grad is not None and prompt_parameter.grad.norm() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(model(inputs.batch([2, 0, 1])), y[[2, 0, 1]], atol=2e-6, rtol=1e-5)
    replacement = model.prompts().detach().clone()
    replacement[0] += torch.linspace(-.5, .5, 16)
    assert (model(x, query_override=replacement) - y).abs().max() > 1e-6
    if method != 'N0':
        assert model.layer_gates.grad is not None and model.layer_gates.grad.norm() > 0
        attention = aux['prompt_atom_attention']
        mask = (~aux['atom_mask'])[:, None, None, :].expand_as(attention)
        assert torch.all(attention[mask] == 0)
    if method in ('N0', 'N4'):
        torch.testing.assert_close(y, aux['graph_prediction_part'] + aux['fingerprint_prediction_part'])


def test_task_smoothing_preserves_constants_and_separates_measurement_types():
    meta = task_metadata(TASKS)
    smoother = task_smoother([meta[t] for t in TASKS], [512, 9, 9])
    torch.testing.assert_close(smoother.sum(1), torch.ones(3))
    assert smoother[0, 2] == smoother[1, 2] == smoother[2, 0] == smoother[2, 1] == 0
    assert smoother[1, 0] > smoother[0, 1] > 0
    values = torch.tensor([[1.], [-1.], [3.]])
    assert (smoother @ values)[2, 0] == 3
    assert ((smoother @ values)[0] - (smoother @ values)[1]).abs() < (values[0] - values[1]).abs()


def test_mini_selection_keeps_targets_and_does_not_depend_on_label_values():
    names = list(SOURCES + TARGETS)
    rng = np.random.default_rng(17)
    labels = rng.normal(size=(600, 12))
    labels[10:, 6:] = np.nan
    train = dict(labels=labels, record_id=np.array([f'r{i}' for i in range(600)]),
        chemical_id=np.array([f'c{i}' for i in range(600)]), group_id=np.array([f'g{i}' for i in range(600)]),
        raw_row_index=np.arange(600), cache_offset=np.arange(600))
    val = {key: value[:4].copy() for key, value in train.items()}
    a, va = select_pilot_arrays(train, val, names)
    changed = {**train, 'labels': train['labels'] * -20 + 99}
    b, vb = select_pilot_arrays(changed, val, names)
    np.testing.assert_array_equal(a['record_id'], b['record_id'])
    np.testing.assert_array_equal(np.isfinite(a['labels']), np.isfinite(b['labels']))
    np.testing.assert_array_equal(np.isfinite(a['labels']).sum(0), [512] * 6 + [10] * 6)
    np.testing.assert_array_equal(va['labels'], vb['labels'])


def test_target_checkpoint_selection_differs_from_all_task_selection(tmp_path, monkeypatch):
    from pgt_v2 import training
    class Inputs:
        records = {'group_id': np.array(['a', 'b'])}
        def __len__(self): return 2
        def batch(self, ids): return torch.ones(len(ids), 1)
    # Epoch 1 has the best all-task mean; epoch 2 has the best target error.
    predictions = iter([np.array([[0., 3.], [0., 3.]]), np.array([[10., 1.], [10., 1.]])])
    monkeypatch.setattr(training, 'predict_array', lambda *a, **kw: next(predictions))
    model = torch.nn.Linear(1, 2)
    cfg = {**tiny_config('N2'), 'selection_tasks': ['target'], 'effective_batch': 2}
    stats = dict(mean=np.zeros(2), scale=np.ones(2), count=np.array([2, 2]))
    summary = train_model(model, Inputs(), np.zeros((2, 2)), Inputs(), np.zeros((2, 2)), stats,
        config=cfg, identity=dict(task_names=['source', 'target']), output=tmp_path/'selection', method='N2')
    assert summary['best_epoch'] == 2 and summary['selection_score'] == 1
    assert summary['validation']['macro_rmse'] == 5.5
    assert read_checkpoint(tmp_path/'selection/best.pt')['epoch'] == 2


def test_fused_model_two_epoch_training_and_checkpoint_reload(tmp_path):
    seed_everything(43)
    cfg = tiny_config('N4')
    inputs = tiny_inputs('N4')
    labels = np.array([[1., -.5, .2], [.3, 2., -.2], [1.2, -.1, .4]])
    stats = dict(mean=np.zeros(3), scale=np.ones(3), count=np.array([3, 3, 3]), adjacency=np.eye(3))
    model = build_model('N4', TASKS, cfg, counts=stats['count'])
    summary = train_model(model, inputs, labels, inputs, labels, stats, config=cfg,
        identity=dict(task_names=list(TASKS), method_id='N4', config=cfg), output=tmp_path/'fit', method='N4')
    restored, checkpoint = load_model_checkpoint(tmp_path/'fit/best.pt')
    model.load_state_dict(checkpoint['model']); model.eval()
    torch.testing.assert_close(restored(inputs.batch([0, 1, 2])), model(inputs.batch([0, 1, 2])))
    assert summary['optimizer_steps'] == 2 and summary['terminal'] == 'COMPLETE'


@pytest.mark.parametrize('seed', [42, 43])
def test_shared_prompt_matches_full_n1_initialization_and_rng(seed):
    tasks = SOURCES + TARGETS
    seed_everything(seed)
    original = build_model('N1', tasks, candidate('N1', 'C2'))
    original_rng = torch.get_rng_state().clone()
    seed_everything(seed)
    shared = build_model('N1S', tasks, candidate('N1S', 'C2'))
    assert torch.equal(original_rng, torch.get_rng_state())
    for name, value in original.state_dict().items():
        expected = value[:1] if name == 'prompt_ids' else value
        assert torch.equal(shared.state_dict()[name], expected), name
    assert shared.prompt_ids.shape == (1, 96)
    assert shared.prompts().shape == (12, 96)
    assert parameter_counts(original)['trainable'] == 476672
    assert parameter_counts(shared)['trainable'] == 475616


def test_shared_prompt_gradient_accumulates_across_all_task_slots():
    seed_everything(42)
    independent = build_model('N1', TASKS, tiny_config('N1')).eval()
    seed_everything(42)
    shared = build_model('N1S', TASKS, tiny_config('N1S')).eval()
    with torch.no_grad():
        independent.prompt_ids.copy_(shared.prompts())
    inputs = tiny_inputs('N1S').batch([0, 1, 2])
    expected = independent(inputs)
    prediction = shared(inputs)
    torch.testing.assert_close(prediction, expected, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(prediction[:, 0], prediction[:, 1])
    labels = torch.tensor([[1., -.5, .2], [.3, 2., -.2], [1.2, -.1, .4]])
    endpoint_loss(expected, labels, [3, 3, 3], 3).backward()
    endpoint_loss(prediction, labels, [3, 3, 3], 3).backward()
    assert shared.prompt_ids.grad.norm() > 0
    torch.testing.assert_close(shared.prompt_ids.grad, independent.prompt_ids.grad.sum(0, keepdim=True),
                               rtol=1e-5, atol=1e-7)


def test_shared_prompt_training_and_checkpoint_reload(tmp_path):
    seed_everything(43)
    cfg = tiny_config('N1S')
    inputs = tiny_inputs('N1S')
    labels = np.array([[1., -.5, .2], [.3, 2., -.2], [1.2, -.1, .4]])
    stats = dict(mean=np.zeros(3), scale=np.ones(3), count=np.array([3, 3, 3]), adjacency=np.eye(3))
    model = build_model('N1S', TASKS, cfg, counts=stats['count'])
    initial_prompt = model.prompt_ids.detach().clone()
    summary = train_model(model, inputs, labels, inputs, labels, stats, config=cfg,
        identity=dict(task_names=list(TASKS), method_id='N1S', config=cfg), output=tmp_path/'fit', method='N1S')
    assert not torch.equal(initial_prompt, model.prompt_ids)
    restored, checkpoint = load_model_checkpoint(tmp_path/'fit/best.pt')
    model.load_state_dict(checkpoint['model']); model.eval()
    assert restored.prompt_ids.shape == (1, 16)
    assert torch.equal(restored.prompts()[0], restored.prompts()[2])
    torch.testing.assert_close(restored(inputs.batch([0, 1, 2])), model(inputs.batch([0, 1, 2])))
    assert summary['optimizer_steps'] == 2 and summary['terminal'] == 'COMPLETE'


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_hierarchical_prompts_match_shared_initialization_and_predictions(seed):
    tasks = SOURCES + TARGETS
    adjacency = np.eye(len(tasks))
    seed_everything(seed)
    shared = build_model('N1S', tasks, candidate('N1S', 'C2')).eval()
    expected_rng = torch.get_rng_state().clone()
    x = tiny_inputs('N1S').batch([0, 1, 2])
    with torch.no_grad():
        expected = shared(x)
    for method in ('N1H', 'N1R', 'N1RP'):
        seed_everything(seed)
        numpy_rng = np.random.get_state()
        model = build_model(method, tasks, candidate(method, 'C2'), adjacency=adjacency).eval()
        assert torch.equal(expected_rng, torch.get_rng_state())
        actual_numpy_rng = np.random.get_state()
        assert numpy_rng[0] == actual_numpy_rng[0] and numpy_rng[2:] == actual_numpy_rng[2:]
        np.testing.assert_array_equal(numpy_rng[1], actual_numpy_rng[1])
        for name, value in shared.state_dict().items():
            assert torch.equal(model.state_dict()[name], value), name
        assert parameter_counts(model)['trainable'] == 476768
        assert torch.count_nonzero(model.prompt_delta) == 0
        assert model.regularization_loss().item() == 0
        with torch.no_grad():
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_relation_prior_weights_permutation_and_regularization_gradient():
    tasks = SOURCES + TARGETS
    metadata = task_metadata(tasks)
    adjacency = np.eye(len(tasks))
    adjacency[1, 3] = adjacency[3, 1] = 0.4
    adjacency[2, 3] = adjacency[3, 2] = 0.3
    weights = relation_prior([metadata[t] for t in tasks], adjacency)
    assert torch.count_nonzero(torch.triu(weights, 1)) == 11
    assert torch.triu(weights, 1).sum() == 3.5
    assert weights[1, 3] == weights[2, 3] == 1
    assert weights[6, 7] == 0.25 and weights[0, 6] == 0
    assert weights[5, 10] == weights[5, 11] == 0.125
    permutation = [11, 4, 1, 5, 8, 7, 0, 6, 2, 3, 10, 9]
    for method in ('N1H', 'N1R', 'N1RP'):
        model = build_model(method, tasks, candidate(method, 'C2'), adjacency=adjacency)
        expected = (torch.zeros_like(weights) if method == 'N1H' else
                    weights[permutation][:, permutation] if method == 'N1RP' else weights)
        torch.testing.assert_close(model.relation_weights, expected)
        if method == 'N1RP':
            assert model.relation_permutation.tolist() == permutation
        z = torch.arange(12 * 96, dtype=torch.float32).reshape(12, 96) / (12 * 96)
        with torch.no_grad():
            model.prompt_delta.copy_(0.02 * z)
        laplacian = torch.diag(expected.sum(1)) - expected
        mass = torch.triu(expected, 1).sum()
        energy = ((z * (laplacian @ z)).sum() / (96 * mass)) if mass > 0 else 0
        loss = model.regularization_loss()
        torch.testing.assert_close(loss, 0.01 * (z.square().mean() + energy))
        loss.backward()
        gradient = 2 * 0.01 * z / z.numel()
        if mass > 0:
            gradient += 2 * 0.01 * (laplacian @ z) / (96 * mass)
        torch.testing.assert_close(model.prompt_delta.grad, gradient / 0.02, atol=1e-8, rtol=1e-5)
        assert model.prompt_ids.grad is None  # The prior constrains differences, not the shared base.


@pytest.mark.parametrize('method', ['N1H', 'N1R', 'N1RP'])
def test_hierarchical_training_and_checkpoint_reload(method, tmp_path):
    seed_everything(43)
    cfg = tiny_config(method)
    inputs = tiny_inputs(method)
    labels = np.array([[1., -.5, .2], [.3, 2., -.2], [1.2, -.1, .4]])
    stats = dict(mean=np.zeros(3), scale=np.ones(3), count=np.array([3, 3, 3]), adjacency=np.eye(3))
    model = build_model(method, TASKS, cfg, adjacency=stats['adjacency'], counts=stats['count'])
    summary = train_model(model, inputs, labels, inputs, labels, stats, config=cfg,
        identity=dict(task_names=list(TASKS), method_id=method, config=cfg), output=tmp_path/'fit', method=method)
    assert model.prompt_delta.norm() > 0 and model.regularization_loss() > 0
    restored, checkpoint = load_model_checkpoint(tmp_path/'fit/best.pt')
    model.load_state_dict(checkpoint['model']); model.eval()
    torch.testing.assert_close(restored(inputs.batch([0, 1, 2])), model(inputs.batch([0, 1, 2])))
    torch.testing.assert_close(restored.relation_weights, model.relation_weights)
    torch.testing.assert_close(restored.regularization_loss(), model.regularization_loss())
    assert summary['optimizer_steps'] == 2 and summary['terminal'] == 'COMPLETE'
