"""Minimal scientific contracts; no data, checkpoint or GPU fixtures."""
import numpy as np
import pytest
import torch
from torch import nn

from baselines.metrics import regression_metrics
from conformal import ConformalCalibrator
from d8_retention import collect_probe_items, build_hybrid_model
from trainer import Trainer
from architecture.toxacute_tasks import HUMAN_TARGET_TASKS


class Store:
    sample_ids = ['shared']
    missing = False
    def get_task_indices(self, task, **kwargs): return [0]
    def get_label(self, index, task): return None if self.missing and task == 'b' else 1.
    def get_graph_data(self, index, **kwargs): return (index, kwargs['task_name'])


def test_d1_shared_molecule_retains_both_endpoint_observations():
    pairs = collect_probe_items(Store(), ['a', 'b'], {'a': ['shared'], 'b': ['shared']})
    assert [task for task, _ in pairs] == ['a', 'b']


def test_d1_missing_endpoint_not_hidden_by_other_endpoint():
    store = Store(); store.missing = True
    with pytest.raises(RuntimeError, match='incomplete'):
        collect_probe_items(store, ['a', 'b'], {'a': ['shared'], 'b': ['shared']})


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_d2_reject_nonfinite_prediction_at_observed_label(bad):
    with pytest.raises(ValueError, match='nonfinite prediction'):
        regression_metrics(np.array([[0.], [100.]]), np.array([[0.], [bad]]), ['a'])


def test_d2_missing_label_allowed_and_normal_result_unchanged():
    result = regression_metrics(np.array([[0.], [2.], [np.nan]]), np.array([[1.], [1.], [np.nan]]), ['a'])
    assert result['macro_rmse'] == 1.
    assert result['pooled_n'] == 2


def make_trainer():
    trainer = Trainer.__new__(Trainer)
    trainer.task_name = list(HUMAN_TARGET_TASKS)
    trainer.selection_scope = 'human3'
    trainer.task_dict = {t: {'metrics': ['RMSE'], 'weight': [-1]} for t in trainer.task_name}
    return trainer


def test_d3_missing_required_endpoint_invalidates_selection():
    trainer = make_trainer()
    buffers = {t: {'pred': [torch.ones(1, 1)], 'label': [torch.zeros(1, 1)]} for t in trainer.task_name}
    buffers[trainer.task_name[-1]] = {'pred': [], 'label': []}
    result = trainer._score_buffers(buffers)
    assert result['selection_score'] == -float('inf')


def test_d3_complete_selection_unchanged():
    trainer = make_trainer()
    buffers = {t: {'pred': [torch.ones(1, 1)], 'label': [torch.zeros(1, 1)]} for t in trainer.task_name}
    assert trainer._score_buffers(buffers)['selection_score'] == -1.


def test_d3_nonfinite_required_endpoint_cannot_select():
    trainer = make_trainer()
    buffers = {t: {'pred': [torch.tensor([[float('inf')]])], 'label': [torch.zeros(1, 1)]} for t in trainer.task_name}
    try:
        result = trainer._score_buffers(buffers)
    except ValueError:
        return  # metric backend may fail closed before aggregation
    assert result['selection_score'] == -float('inf')


def test_d3_animal_only_all_tasks_fallback_still_selects():
    trainer = make_trainer(); trainer.task_name = ['rat_oral_LD50']
    trainer.task_dict = {'rat_oral_LD50': {'metrics': ['RMSE'], 'weight': [-1]}}
    result = trainer._score_buffers({'rat_oral_LD50': {'pred': [torch.ones(1, 1)], 'label': [torch.zeros(1, 1)]}})
    assert result['selection_scope'] == 'all_tasks'
    assert result['selection_score'] == -1.


def test_d4_column_vector_normalizes_without_cross_product():
    cal = ConformalCalibrator(alpha=.5, min_calibration_size=0)
    state = cal.fit_task('t', torch.zeros(3, 1), torch.ones(3, 1)*2, torch.ones(3))
    assert state.count == 3
    assert state.qhat == -1.


@pytest.mark.parametrize('shape', [(3, 2), (1, 3), (2,)])
def test_d4_reject_nonpaired_shapes(shape):
    with pytest.raises(ValueError):
        ConformalCalibrator.conformity_scores(torch.zeros(shape), torch.ones(3), torch.ones(3))


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_d4_reject_nonfinite_calibration(bad):
    with pytest.raises(ValueError, match='finite'):
        ConformalCalibrator(alpha=.5).fit_task('t', [0., 0., bad], [2., 2., 2.], [1., 1., 1.])


def teacher():
    model = nn.Module(); model.encoder = nn.Module()
    model.encoder.backbone = nn.Linear(2, 2)
    model.head = nn.Linear(2, 1)
    return model


def test_d5_missing_backbone_rejected():
    model = teacher()
    with pytest.raises(RuntimeError, match='backbone'):
        build_hybrid_model(model, {'encoder.backbone.weight': torch.zeros(2, 2)})


def test_d5_complete_preserves_head():
    model = teacher()
    state = {k: torch.zeros_like(v) for k, v in model.state_dict().items() if k.startswith('encoder.backbone.')}
    hybrid = build_hybrid_model(model, state)
    for k, v in hybrid.state_dict().items():
        assert torch.equal(v, state[k] if k in state else model.state_dict()[k])


def test_d5_wrong_shape_rejected():
    model = teacher(); state = dict(model.state_dict())
    state['encoder.backbone.weight'] = torch.zeros(1, 1)
    with pytest.raises(RuntimeError): build_hybrid_model(model, state)
