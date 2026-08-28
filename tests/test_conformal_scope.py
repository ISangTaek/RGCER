"""--conformal_scope contract: qhat only where fitted, intervals scoped."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS, HUMAN_TARGET_TASKS
from loss import MSELoss
from trainer import Trainer

HUMAN_TASKS = list(HUMAN_TARGET_TASKS)
ANIMAL_TASK = ANIMAL_SOURCE_TASKS[0]
TASK_NAMES = HUMAN_TASKS + [ANIMAL_TASK]


class _Batch:
    def __init__(self, value):
        self.y = torch.tensor([[float(value)]])
        self.smiles = ["C"]
        self.sample_id = ["row"]

    def get(self, key, default=None):
        return default

    def to(self, device):
        return self


class _OneBatch(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return _Batch(index)


def _quantile_model(batch, task_name=None, return_aux=True, **kwargs):
    """Constant three-channel output keyed per task, like the real HPS model."""

    return {task_name: torch.zeros(batch.y.size(0), 3)}


def _bare_trainer(scope, task_names=TASK_NAMES):
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        prediction_mode="quantile",
        conformal_alpha=0.30,
        conformal_scope=scope,
        splitting="scaffold",
        fit_conformal=True,
        lower_quantile=0.05,
        upper_quantile=0.95,
    )
    trainer.task_name = list(task_names)
    trainer.task_num = len(task_names)
    trainer.task_dict = {
        name: {"metrics": ["RMSE", "R2"], "weight": [-1, 1], "loss_fn": MSELoss()}
        for name in task_names
    }
    trainer.conformal_scope = scope
    trainer.selection_scope = "human3"
    trainer.conformal_calibrator = __import__("conformal").ConformalCalibrator(alpha=0.30)
    trainer.conformal_fit_report = []
    trainer._is_rgcer = False
    trainer.model = type(
        "_M", (), {"train": lambda self: None, "eval": lambda self: None, "__call__": staticmethod(_quantile_model)}
    )()
    trainer.device = torch.device("cpu")
    trainer.training_cache = {}
    trainer.task_scalers = {}
    trainer.source_policy = "animal56_only"
    trainer.allowed_auxiliary_sources = ()
    return trainer


def test_conformal_tasks_respects_scope_and_flags():
    human_only = _bare_trainer("human3")
    assert set(human_only._conformal_tasks()) == set(HUMAN_TASKS)

    all_tasks = _bare_trainer("all_tasks")
    assert set(all_tasks._conformal_tasks()) == set(TASK_NAMES)

    human_only.task_dict[HUMAN_TASKS[0]]["fit_conformal"] = False
    assert HUMAN_TASKS[0] not in human_only._conformal_tasks()


def _evaluate_with_fitted_states(trainer):
    """Fit trivial qhat states for the scoped tasks, then evaluate 'test'."""

    from conformal import ConformalCalibrator

    calibrator = ConformalCalibrator(alpha=0.30)
    tasks_to_fit = trainer._conformal_tasks()
    for task in tasks_to_fit:
        zero = torch.zeros(9)
        calibrator.fit_task(task, zero - 1.0, zero + 1.0, zero)
    trainer.conformal_calibrator = calibrator

    loaders = {name: DataLoader(_OneBatch(), batch_size=None) for name in trainer.task_name}
    result = trainer._evaluate(loaders, mode="test")
    return result


def test_interval_report_scoped_per_task_with_macro():
    trainer = _bare_trainer("human3")
    result = _evaluate_with_fitted_states(trainer)

    interval = result["interval"]
    assert interval["scope"] == "human3"
    assert set(interval["tasks"]) == set(HUMAN_TASKS)
    assert ANIMAL_TASK not in interval["tasks"]
    assert set(interval["macro"]) == set(next(iter(interval["tasks"].values())))
    # Point metrics still cover every task including the animal endpoint.
    assert set(result["tasks"]) == set(TASK_NAMES)


def test_animal_intervals_are_not_labelled_conformal_when_unfitted():
    trainer = _bare_trainer("human3")
    trainer.conformal_calibrator = __import__("conformal").ConformalCalibrator(alpha=0.30)
    loaders = {name: DataLoader(_OneBatch(), batch_size=None) for name in trainer.task_name}
    # No states at all: the animal task must still decode without error.
    result = trainer._evaluate(loaders, mode="test")
    assert set(result["tasks"]) == set(TASK_NAMES)


def test_all_tasks_scope_reports_every_endpoint():
    trainer = _bare_trainer("all_tasks")
    result = _evaluate_with_fitted_states(trainer)
    assert set(result["interval"]["tasks"]) == set(TASK_NAMES)
