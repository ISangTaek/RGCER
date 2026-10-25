"""Only the new scientific paths: prompt gradients, isolation and target selection."""
import numpy as np
import pytest
import torch
from rdkit import Chem

from pgt_v2.chemistry import graph_from_input, prepare_molecule
from pgt_v2.config import candidate
from pgt_v2.evaluation import load_model_checkpoint
from pgt_v2.minipilot import SOURCES, TARGETS, select_pilot_arrays
from pgt_v2.models import build_model, task_metadata
from pgt_v2.redesign_models import task_smoother
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
