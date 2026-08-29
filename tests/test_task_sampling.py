"""Task sampling actual-consumption contract (review §14-17, §45)."""

from types import SimpleNamespace

import torch
from torch import nn

from trainer import Trainer


class _Batch:
    def __init__(self, value):
        self.y = torch.tensor([[float(value)]])

    def get(self, key, default=None):
        return default

    def to(self, device):
        return self


class _Loader:
    """Sized loader whose consumed batches are recorded per instance."""

    def __init__(self, values):
        self.values = list(values)
        self.batches_yielded = 0

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for value in self.values:
            self.batches_yielded += 1
            yield _Batch(value)


class _Balancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vectors = []

    def init_param(self):
        pass

    def backward(self, losses, active_mask=None):
        self.vectors.append(losses.detach().clone())
        losses.sum().backward()


def _make_trainer(task_names, sampling):
    trainer = Trainer.__new__(Trainer)
    trainer.task_name = list(task_names)
    trainer.task_num = len(task_names)
    trainer.task_dict = {task: {"metrics": ["RMSE"], "weight": [-1]} for task in task_names}
    trainer.seed = 11
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(tasks_per_update=1, grad_clip=0, task_sampling=sampling)
    trainer._is_rgcer = False
    trainer.model = nn.Linear(1, 1, bias=False)
    trainer.loss_balancer = _Balancer()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.task_scalers = {task: {"mean": 0.0, "std": 1.0} for task in task_names}
    trainer.training_cache = {}
    trainer.optimizer_updates = 0
    trainer.schedule_usage = {}
    trainer.decode_task_output = lambda task, raw, apply_conformal=False: {
        "median": raw,
        "lower": raw,
        "upper": raw,
    }
    trainer._record_training_output = lambda bundle: None
    trainer._score_buffers = lambda buffers: {"tasks": {}, "score": 0.0}
    trainer._training_step = lambda batch, task, epoch: (
        trainer.model.weight.sum() * 0.0 + batch.y.sum() * 0.0 + 1.0,
        {
            "task": task,
            "labels": batch.y,
            "final_raw": trainer.model(torch.ones(1, 1)),
            "base_raw": trainer.model(torch.ones(1, 1)),
            "route_raw": trainer.model(torch.ones(1, 1)),
            "diagnostics": {},
            "loss": torch.tensor(1.0),
            "final_loss": torch.tensor(0.75),
            "base_loss": torch.tensor(0.5),
        },
    )
    return trainer


def test_proportional_consumes_every_batch_once():
    human = _Loader([1, 2])
    animal = _Loader([3, 4, 5, 6])
    trainer = _make_trainer(["human_oral_TDLo", "mouse_oral_LD50"], "proportional")

    result = trainer._train_epoch(
        {"human_oral_TDLo": human, "mouse_oral_LD50": animal}, epoch=0
    )

    assert result["schedule_usage"] == {"human_oral_TDLo": 2, "mouse_oral_LD50": 4}
    diagnostics = result["schedule_diagnostics"]
    assert diagnostics["planned_task_batches"] == {"human_oral_TDLo": 2, "mouse_oral_LD50": 4}
    assert diagnostics["actual_task_batches"] == {"human_oral_TDLo": 2, "mouse_oral_LD50": 4}
    assert diagnostics["actual_task_samples"] == {"human_oral_TDLo": 2, "mouse_oral_LD50": 4}


def test_human_target_floor_extra_pass_is_really_consumed():
    # Review §45: human loader len=2, floor adds one full extra pass →
    # planned 4 AND actual 4.  The old code planned 4 but consumed 2.
    human = _Loader([1, 2])
    trainer = _make_trainer(["human_oral_TDLo"], "human_target_floor")

    result = trainer._train_epoch({"human_oral_TDLo": human}, epoch=0)

    assert human.batches_yielded == 4, "extra pass must draw real batches"
    assert result["schedule_usage"] == {"human_oral_TDLo": 4}
    diagnostics = result["schedule_diagnostics"]
    assert diagnostics["planned_task_batches"] == {"human_oral_TDLo": 4}
    assert diagnostics["actual_task_batches"] == {"human_oral_TDLo": 4}
    assert diagnostics["human3_actual_fraction"] == 1.0


def test_no_unbounded_restart_beyond_planned_passes():
    loader = _Loader([1, 2])
    trainer = _make_trainer(["human_oral_TDLo"], "proportional")

    result = trainer._train_epoch({"human_oral_TDLo": loader}, epoch=0)

    assert loader.batches_yielded == 2
    assert result["schedule_usage"] == {"human_oral_TDLo": 2}
