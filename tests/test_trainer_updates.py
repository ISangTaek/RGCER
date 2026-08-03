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
