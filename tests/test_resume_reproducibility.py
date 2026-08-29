"""Epoch-boundary deterministic resume contract (review §31-35).

The seed policy makes each epoch's random trajectory a pure function of
(base_seed, epoch[, task]), so a fresh Trainer resuming at an epoch
boundary replays exactly what a continuous run would have done.
"""

import copy
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
    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __iter__(self):
        for index in range(self.size):
            yield _Batch(index)


class _Balancer(nn.Module):
    def init_param(self):
        pass

    def backward(self, losses, active_mask=None):
        losses.sum().backward()


def _make_trainer(seed):
    torch.manual_seed(seed)
    trainer = Trainer.__new__(Trainer)
    trainer.task_name = ["task"]
    trainer.task_num = 1
    trainer.task_dict = {"task": {"metrics": ["RMSE"], "weight": [-1]}}
    trainer.seed = seed
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(tasks_per_update=1, grad_clip=1.0, task_sampling="proportional")
    trainer._is_rgcer = False
    trainer.model = nn.Linear(2, 1, bias=False)
    trainer.loss_balancer = _Balancer()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.05)
    trainer.task_scalers = {"task": {"mean": 0.0, "std": 1.0}}
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
        (trainer.model(torch.ones(1, 2).fill_(batch.y.item())) ** 2).sum(),
        {
            "task": task,
            "labels": batch.y,
            "final_raw": trainer.model(torch.ones(1, 2)),
            "base_raw": trainer.model(torch.ones(1, 2)),
            "route_raw": trainer.model(torch.ones(1, 2)),
            "diagnostics": {},
            "loss": torch.tensor(1.0),
            "final_loss": torch.tensor(0.75),
            "base_loss": torch.tensor(0.5),
        },
    )
    return trainer


def _run_epoch(trainer, epoch, loaders=None):
    trainer._train_epoch(loaders or {"task": _Loader(6)}, epoch)


def test_epoch_boundary_resume_matches_continuous_run():
    continuous = _make_trainer(42)
    _run_epoch(continuous, 0)
    state_after_epoch0 = copy.deepcopy(continuous.model.state_dict())
    updates_after_epoch0 = continuous.optimizer_updates
    _run_epoch(continuous, 1)

    # "Resume": a fresh Trainer restores the epoch-0 checkpoint (weights and
    # the optimizer_updates counter), then runs only epoch 1 there.
    resumed = _make_trainer(42)
    resumed.model.load_state_dict(state_after_epoch0)
    resumed.optimizer_updates = updates_after_epoch0
    _run_epoch(resumed, 1)

    for parameter_continuous, parameter_resumed in zip(
        continuous.model.parameters(), resumed.model.parameters()
    ):
        torch.testing.assert_close(parameter_continuous, parameter_resumed, rtol=0, atol=0)
    assert continuous.optimizer_updates == resumed.optimizer_updates


def test_epoch_order_is_reproducible_across_trainer_instances():
    first = _make_trainer(42)
    _run_epoch(first, 0)
    second = _make_trainer(42)
    _run_epoch(second, 0)

    for parameter_first, parameter_second in zip(
        first.model.parameters(), second.model.parameters()
    ):
        torch.testing.assert_close(parameter_first, parameter_second, rtol=0, atol=0)


def test_checkpoint_payload_records_reproducibility_policy():
    trainer = _make_trainer(42)
    trainer.initial_model_sha256 = "abc123"
    trainer.data_metadata = None
    trainer.conformal_fit_report = []
    trainer.best_val_score = -float("inf")
    trainer.best_epoch = None
    trainer.best_routing_enabled = None
    trainer.best_checkpoint_path = None
    trainer._best_state = None
    trainer._best_training_state = None
    trainer.conformal_calibrator = SimpleNamespace(
        state_dict=lambda: {},
        states={},
    )
    trainer._architecture_config = lambda: {}
    trainer._rgcer_config = lambda: {}
    trainer._data_config = lambda: {}
    trainer._manifest_hash = lambda: "hash"
    trainer._prediction_mode = lambda task=None: "quantile"
    trainer._conformal_validity = lambda: {}
    trainer._routing_enabled = lambda epoch: False

    payload = trainer._checkpoint_payload(0)

    block = payload["reproducibility"]
    assert block["base_seed"] == 42
    assert block["seed_policy_version"] == 2
    assert "base_seed" in block["epoch_seed_scheme"]
    assert "task" in block["loader_seed_scheme"]
    assert block["initial_model_sha256"] == "abc123"
