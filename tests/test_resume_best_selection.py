"""Historical best-selection continuity across resume (review-2 §21-22).

The selection contract has two halves: the training trajectory (covered by
``test_resume_reproducibility.py``) and the model-selection trajectory — a
resumed run must keep the pre-interruption best and only replace it when a
new epoch is strictly better.
"""

import copy
from types import SimpleNamespace

import torch
from torch import nn

from trainer import Trainer


class _BestModel(nn.Module):
    """Tiny model whose weights encode "which epoch produced it"."""

    def __init__(self, fill=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[float(fill)]]))


def _selection_trainer(seed_value, best_score=None):
    trainer = Trainer.__new__(Trainer)
    trainer.task_name = ["task"]
    trainer.task_num = 1
    trainer.task_dict = {"task": {"metrics": ["RMSE"], "weight": [-1]}}
    trainer.seed = 42
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(
        routing_enabled=False, hps_warmup_epochs=0, ckpt_name="model"
    )
    trainer._is_rgcer = False
    trainer.model = _BestModel(seed_value)
    trainer.loss_balancer = nn.Identity()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    trainer.train_loss_buffer = torch.ones(1, 2).numpy().copy()
    trainer.optimizer_updates = seed_value
    trainer.save_path = None
    trainer.best_val_score = -float("inf") if best_score is None else best_score
    trainer.best_epoch = None if best_score is None else 0
    trainer.best_routing_enabled = None
    trainer.best_checkpoint_path = None
    trainer._best_state = None
    trainer._best_training_state = None
    return trainer


def _mark_best(trainer, epoch, score):
    trainer._best_state = copy.deepcopy(trainer.model.state_dict())
    trainer._best_training_state = {
        "model_state": copy.deepcopy(trainer.model.state_dict()),
        "optimizer_state": copy.deepcopy(trainer.optimizer.state_dict()),
        "weighting_state": copy.deepcopy(trainer.loss_balancer.state_dict()),
        "train_loss_buffer": trainer.train_loss_buffer.copy(),
        "optimizer_updates": int(trainer.optimizer_updates),
        "epoch": epoch,
    }
    trainer.best_val_score = score
    trainer.best_epoch = epoch


def _validation(score):
    return {"selection_score": score}


def _restore_best(trainer):
    """End-of-training restore, mirroring Trainer.train()'s tail."""

    if trainer._best_training_state is not None:
        trainer.model.load_state_dict(trainer._best_training_state["model_state"], strict=True)
        trainer.optimizer_updates = trainer._best_training_state["optimizer_updates"]


def test_early_historical_best_survives_worse_later_epoch():
    # Review-2 §21 Test A: epoch 0 scored 10 (the best), interruption, then
    # a resumed epoch scores 5 — it must not steal the checkpoint.
    source = _selection_trainer(seed_value=1)
    _mark_best(source, epoch=0, score=10.0)

    resumed = _selection_trainer(seed_value=2)
    resumed.model.load_state_dict(source.model.state_dict())
    resumed.best_val_score = source.best_val_score
    resumed.best_epoch = source.best_epoch
    resumed._best_state = copy.deepcopy(source._best_state)
    resumed._best_training_state = copy.deepcopy(source._best_training_state)

    replaced = resumed._maybe_update_best(_validation(5.0), epoch=1)

    assert replaced is False
    assert resumed.best_epoch == 0
    assert resumed.best_val_score == 10.0
    _restore_best(resumed)
    assert resumed.model.weight.item() == 1.0, "final model must be the epoch-0 best"
    assert resumed.optimizer_updates == 1


def test_genuinely_better_new_epoch_replaces_historical_best():
    # Review-2 §21 Test B / §43: epoch 0 scored 10, resumed epoch scores 11.
    source = _selection_trainer(seed_value=1)
    _mark_best(source, epoch=0, score=10.0)

    resumed = _selection_trainer(seed_value=2)
    resumed.best_val_score = source.best_val_score
    resumed.best_epoch = source.best_epoch
    resumed.best_routing_enabled = False

    replaced = resumed._maybe_update_best(_validation(11.0), epoch=1)

    assert replaced is True
    assert resumed.best_epoch == 1
    assert resumed.best_val_score == 11.0
    assert resumed._best_training_state["epoch"] == 1


def test_continuous_vs_interrupted_resume_end_states_match():
    # Review-2 §21 Test C: identical epoch score sequences 10, 5, 7.
    # Run A trains straight through; Run B restores from a last.pt-style
    # snapshot after epoch 0. Final best selection must agree exactly.
    schedule = {0: 10.0, 1: 5.0, 2: 7.0}
    weights = {0: 1.0, 1: 2.0, 2: 3.0}

    continuous = _selection_trainer(seed_value=99)
    for epoch in sorted(schedule):
        continuous.model = _BestModel(weights[epoch])
        continuous.optimizer = torch.optim.SGD(continuous.model.parameters(), lr=0.0)
        continuous.optimizer_updates = epoch + 1
        continuous._maybe_update_best(_validation(schedule[epoch]), epoch=epoch)
    _restore_best(continuous)

    interrupted = _selection_trainer(seed_value=99)
    interrupted.model = _BestModel(weights[0])
    interrupted.optimizer_updates = 1  # after epoch 0
    interrupted._maybe_update_best(_validation(schedule[0]), epoch=0)
    snapshot = copy.deepcopy(
        {
            "best_val_score": interrupted.best_val_score,
            "best_epoch": interrupted.best_epoch,
            "_best_state": interrupted._best_state,
            "_best_training_state": interrupted._best_training_state,
        }
    )

    resumed = _selection_trainer(seed_value=99)
    resumed.model = _BestModel(weights[0])
    resumed.best_val_score = snapshot["best_val_score"]
    resumed.best_epoch = snapshot["best_epoch"]
    resumed._best_state = copy.deepcopy(snapshot["_best_state"])
    resumed._best_training_state = copy.deepcopy(snapshot["_best_training_state"])
    for epoch in (1, 2):
        resumed.model = _BestModel(weights[epoch])
        resumed.optimizer_updates = epoch + 1
        resumed._maybe_update_best(_validation(schedule[epoch]), epoch=epoch)
    _restore_best(resumed)

    assert continuous.best_epoch == resumed.best_epoch == 0
    assert continuous.best_val_score == resumed.best_val_score == 10.0
    assert continuous.model.weight.item() == resumed.model.weight.item() == 1.0
    assert continuous.optimizer_updates == resumed.optimizer_updates == 1
