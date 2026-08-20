from types import SimpleNamespace

import pytest
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
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter([_Batch(value) for value in self.values])


class _Balancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vectors = []

    def init_param(self):
        pass

    def backward(self, losses, active_mask=None):
        self.vectors.append(losses.detach().clone())
        losses.sum().backward()


def test_duplicate_task_batches_are_aggregated_and_one_update_is_counted():
    trainer = Trainer.__new__(Trainer)
    trainer.task_name = ["task"]
    trainer.task_num = 1
    trainer.task_dict = {"task": {"metrics": ["RMSE"], "weight": [-1]}}
    trainer.seed = 11
    trainer.device = torch.device("cpu")
    trainer.args = type("Args", (), {"tasks_per_update": 2, "grad_clip": 0})()
    trainer._is_rgcer = False
    trainer.model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    trainer.loss_balancer = _Balancer()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.task_scalers = {"task": {"mean": 0.0, "std": 1.0}}
    trainer.training_cache = {}
    trainer.optimizer_updates = 0
    trainer.decode_task_output = lambda task, raw, apply_conformal=False: {
        "median": raw,
        "lower": raw,
        "upper": raw,
    }
    trainer._record_training_output = lambda bundle: None
    trainer._score_buffers = lambda buffers: {"tasks": {}, "score": 0.0}
    values = iter([1.0, 3.0])

    def training_step(batch, task, epoch):
        value = next(values)
        loss = trainer.model.weight.sum() * value
        raw = trainer.model(torch.ones(1, 1))
        return loss, {
            "task": task,
            "labels": batch.y,
            "final_raw": raw,
            "base_raw": raw,
            "route_raw": raw,
            "diagnostics": {},
            "loss": loss.detach(),
        }

    trainer._training_step = training_step
    result = trainer._train_epoch({"task": _Loader([1, 2])}, epoch=0)
    assert result["schedule_usage"] == {"task": 2}
    assert result["updates"] == 1
    assert torch.allclose(trainer.loss_balancer.vectors[0], torch.tensor([2.0]))


def test_warmup_checkpoint_state_controls_routing_for_non_epoch_evaluation():
    class _RoutingModel:
        def __init__(self):
            self.calls = []

        def __call__(self, batch, **kwargs):
            self.calls.append(kwargs)
            return {"task": torch.zeros(1, 1)}, {}

    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(routing_enabled=True, hps_warmup_epochs=2)
    trainer._is_rgcer = True
    trainer.loaded_routing_enabled = None
    trainer.model = _RoutingModel()

    trainer._forward_task(None, "task", epoch=0)
    trainer._forward_task(None, "task", epoch=2)
    trainer.loaded_routing_enabled = False
    trainer._forward_task(None, "task", epoch=None)

    assert [call["routing_enabled"] for call in trainer.model.calls] == [False, True, False]


def test_lambda_quantile_applies_during_hps_warmup():
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        lambda_quantile=0.1,
        lambda_base=0.25,
        routing_enabled=True,
        hps_warmup_epochs=2,
        rgcer_use_base_aux_loss=True,
    )
    trainer.task_dict = {"task": {"loss_fn": lambda prediction, labels: prediction}}
    trainer._is_rgcer = True
    trainer.task_scalers = {"task": {"mean": 0.0, "std": 1.0}}
    trainer._forward_task = lambda batch, task, epoch, return_aux=True: (
        {"task": torch.tensor([[2.0]])},
        {"base_raw": torch.tensor([[2.0]])},
    )
    trainer._prediction_loss = lambda task, raw, labels: (raw ** 2).mean()
    batch = _Batch(0.0)

    loss, _ = trainer._training_step(batch, "task", epoch=0)

    assert torch.isclose(loss, torch.tensor(0.4))


def test_train_retains_final_test_result():
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(lower_quantile=0.05, upper_quantile=0.95)
    trainer.task_name = ["task"]
    trainer.task_num = 1
    trainer.task_dict = {"task": {"metrics": ["RMSE"], "weight": [-1]}}
    trainer.task_scalers = {"task": {"mean": 0.0, "std": 1.0}}
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
    trainer._save_checkpoint = lambda epoch, filename: None

    train_result = {"tasks": {"task": {"RMSE": 1.0}}, "score": 1.0, "loss": {"task": 1.0}}
    validation_result = {"tasks": {"task": {"RMSE": 1.0}}, "score": 1.0, "routing": {}}
    test_result = {"tasks": {"task": {"RMSE": 2.0}}, "score": 2.0, "routing": {"task": {}}}
    trainer._train_epoch = lambda loaders, epoch: train_result

    def _evaluate(loaders, mode="validation", epoch=0, routing_enabled_override=None):
        del loaders, epoch, routing_enabled_override
        return validation_result if mode == "validation" else test_result

    trainer._evaluate = _evaluate

    trainer.train(
        train_dataloaders_dict={"task": _Loader([1])},
        val_dataloaders_dict={"task": _Loader([1])},
        test_dataloaders_dict={"task": _Loader([1])},
        epochs=1,
    )

    assert trainer.final_test_result == test_result


def test_record_training_output_mixed_device_bundle():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda:0")
    trainer = Trainer.__new__(Trainer)
    trainer.task_dict = {"task": {"metrics": ["RMSE"], "weight": [-1]}}
    trainer.task_scalers = {"task": {"mean": 0.0, "std": 1.0}}
    trainer.training_cache = {}
    trainer.decode_task_output = lambda task, raw, apply_conformal=False: {
        "median": raw,
        "lower": raw,
        "upper": raw,
    }

    raw = torch.tensor([[2.0]], device=device)
    bundle = {
        "task": "task",
        "labels": torch.tensor([[1.0]], device=device),
        "final_raw": raw,
        "base_raw": raw,
        "route_raw": raw,
        "diagnostics": {},
    }

    trainer._record_training_output(bundle)

    cache = trainer.training_cache["task"]
    assert all(tensor.device.type == "cpu" for tensor in cache["target"])
    assert all(tensor.device.type == "cpu" for tensor in cache["route_regret"])
    assert all(tensor.device.type == "cpu" for tensor in cache["final_regret"])
    assert torch.allclose(cache["route_regret"][0], torch.zeros(1, 1))
