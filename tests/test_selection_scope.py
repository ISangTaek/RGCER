"""Best-checkpoint selection scope (Commit B).

The formal ToxAcute model serves the three human target endpoints, so the
default ``human3`` scope must pick the best checkpoint by human3 macro RMSE
even when another epoch looks better on the all-task macro score.  The
counterexamples below make the two scopes disagree: one epoch is human-best
but all-task-worst, the other the reverse.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from architecture.toxacute_tasks import ANIMAL_SOURCE_TASKS, HUMAN_TARGET_TASKS
from trainer import Trainer

HUMAN_TASKS = list(HUMAN_TARGET_TASKS)
ANIMAL_TASK = "mouse_oral_LD50"
TASK_NAMES = HUMAN_TASKS + [ANIMAL_TASK]

assert ANIMAL_TASK in ANIMAL_SOURCE_TASKS

# Scenario X: best on human3 (2.0) but worst on all-task macro (4.75).
SCENARIO_HUMAN_BEST = {task: 2.0 for task in HUMAN_TASKS}
SCENARIO_HUMAN_BEST[ANIMAL_TASK] = 13.0
# Scenario Y: worst on human3 (6.0) but best on all-task macro (4.625).
SCENARIO_ALL_BEST = {task: 6.0 for task in HUMAN_TASKS}
SCENARIO_ALL_BEST[ANIMAL_TASK] = 0.5


class _Batch:
    def __init__(self, value):
        self.y = torch.tensor([[float(value)]])

    def get(self, key, default=None):
        return default

    def to(self, device):
        return self


class _Loader:
    def __len__(self):
        return 1

    def __iter__(self):
        return iter([_Batch(0.0)])


def _buffers(rmse_by_task):
    """One-sample buffers whose per-task RMSE is exactly the given value."""
    return {
        task: {
            "pred": [torch.tensor([[value]])],
            "label": [torch.tensor([[0.0]])],
        }
        for task, value in rmse_by_task.items()
    }


def _make_trainer(selection_scope):
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        selection_scope=selection_scope,
        lower_quantile=0.05,
        upper_quantile=0.95,
        ckpt_name="model",
    )
    trainer.selection_scope = selection_scope
    trainer.task_name = list(TASK_NAMES)
    trainer.task_num = len(TASK_NAMES)
    trainer.task_dict = {
        task: {"metrics": ["RMSE"], "weight": [-1]} for task in TASK_NAMES
    }
    trainer.task_scalers = {
        task: {"mean": 0.0, "std": 1.0} for task in TASK_NAMES
    }
    trainer.loaded_epoch = None
    trainer._best_state = None
    trainer._best_training_state = None
    trainer.best_val_score = -float("inf")
    trainer.best_checkpoint_path = None
    trainer.best_epoch = None
    trainer.best_routing_enabled = None
    trainer.loaded_routing_enabled = None
    trainer.final_test_result = None
    trainer.optimizer_updates = 0
    trainer.train_loss_buffer = None
    trainer._is_rgcer = False
    trainer.model = nn.Linear(1, 1)
    trainer.loss_balancer = nn.Identity()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer._save_checkpoint = lambda epoch, filename: f"ckpt_epoch{epoch}_{filename}"
    trainer._train_epoch = lambda loaders, epoch: {
        "tasks": {task: {"RMSE": 1.0} for task in TASK_NAMES},
        "score": 1.0,
        "loss": {task: 1.0 for task in TASK_NAMES},
        "updates": 1,
        "schedule_usage": {},
        "routing": {},
    }
    return trainer


def test_selection_scope_human3_vs_all59_counterexample():
    trainer = _make_trainer("human3")
    result_x = trainer._score_buffers(_buffers(SCENARIO_HUMAN_BEST))
    result_y = trainer._score_buffers(_buffers(SCENARIO_ALL_BEST))

    # Both epochs are visible in every reported aggregate...
    for result in (result_x, result_y):
        assert result["selection_scope"] == "human3"
        assert result["selection_tasks"] == HUMAN_TASKS
    assert result_x["human3_macro_rmse"] == pytest.approx(2.0)
    assert result_x["all_task_macro_rmse"] == pytest.approx(4.75)
    assert result_y["human3_macro_rmse"] == pytest.approx(6.0)
    assert result_y["all_task_macro_rmse"] == pytest.approx(4.625)

    # ...but the human3 scope must prefer the human-best epoch...
    assert result_x["selection_score"] > result_y["selection_score"]
    assert result_x["selection_score"] == pytest.approx(-2.0)
    assert result_y["selection_score"] == pytest.approx(-6.0)

    # ...while the all_tasks scope prefers the all-task-best epoch.
    trainer.selection_scope = "all_tasks"
    result_x_all = trainer._score_buffers(_buffers(SCENARIO_HUMAN_BEST))
    result_y_all = trainer._score_buffers(_buffers(SCENARIO_ALL_BEST))
    assert result_x_all["selection_scope"] == "all_tasks"
    assert result_x_all["selection_tasks"] == TASK_NAMES
    assert result_y_all["selection_score"] > result_x_all["selection_score"]
    assert result_y_all["selection_score"] == pytest.approx(-4.625)
    assert result_x_all["selection_score"] == pytest.approx(-4.75)


def test_selection_scope_bad_checkpoint_rejected():
    # Epoch 0 is the all59-best / human3-worst checkpoint; epoch 1 is the
    # reverse.  Under the default human3 scope the all59-best epoch must be
    # rejected for best-checkpoint selection.
    scenario_by_epoch = {0: SCENARIO_ALL_BEST, 1: SCENARIO_HUMAN_BEST}

    for scope, expected_epoch, expected_score in (
        ("human3", 1, -2.0),
        ("all_tasks", 0, -4.625),
    ):
        trainer = _make_trainer(scope)

        def _evaluate(loaders, mode="validation", epoch=0, routing_enabled_override=None):
            del loaders, routing_enabled_override
            assert mode == "validation"
            return trainer._score_buffers(_buffers(scenario_by_epoch[epoch]))

        trainer._evaluate = _evaluate
        trainer.train(
            train_dataloaders_dict={task: _Loader() for task in TASK_NAMES},
            val_dataloaders_dict={task: _Loader() for task in TASK_NAMES},
            epochs=2,
        )

        assert trainer.best_epoch == expected_epoch
        assert trainer.best_val_score == pytest.approx(expected_score)
        if scope == "human3":
            # The all59-best epoch (0) must not have won the checkpoint.
            assert trainer.best_checkpoint_path.startswith("ckpt_epoch1_")
