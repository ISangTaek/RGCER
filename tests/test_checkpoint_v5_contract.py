"""Checkpoint v5 architecture provenance and schedule diagnostics (§42, §39)."""

from types import SimpleNamespace

import pytest
import torch

from tests.test_checkpoint_ablation_config import (
    EW,
    MSELoss,
    Trainer,
    _args,
    _trainer,
)


def test_v5_architecture_config_records_training_behaviour(tmp_path):
    trainer = _trainer(tmp_path / "run")
    config = trainer._architecture_config()
    for field in (
        "hidden_dim",
        "a_layers",
        "a_heads",
        "mid_dim",
        "head_hidden_dim",
        "head_dropout",
        "adapter_ratio",
        "response_hidden_dim",
        "task_sampling",
        "conformal_scope",
    ):
        assert field in config, field


def test_changed_head_dropout_is_rejected_on_load(tmp_path):
    """Silent training-behaviour knobs must fail resume loudly (§42)."""

    source = _trainer(tmp_path / "source", use_null=True)
    checkpoint = source._save_checkpoint(0, "model_best.pt")

    # Same flag value resumes cleanly.
    same = _trainer(tmp_path / "same", use_null=True, load_path=checkpoint)

    # Flipped flag: the strict architecture comparison must reject it.
    args = _args(tmp_path / "shifted", use_null=True)
    args.head_dropout = float(getattr(source.args, "head_dropout", 0.0)) + 0.25
    with pytest.raises(ValueError, match="head_dropout"):
        Trainer(
            task_dict={"task": {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1]}},
            weighting=EW,
            architecture=__import__(
                "tests.test_checkpoint_ablation_config", fromlist=["_DummyArchitecture"]
            )._DummyArchitecture,
            encoder_class=torch.nn.Identity,
            decoders=torch.nn.ModuleDict(),
            optim_param={"optim": "adamw", "lr": 1e-3, "weight_decay": 0.0},
            args=args,
            save_path=tmp_path / "shifted",
            load_path=checkpoint,
        )
    del same


class _Loader(list):
    pass


def _dataloaders(task_sizes):
    return {task: _Loader([object()] * size) for task, size in task_sizes.items()}


def test_schedule_diagnostics_report_proportional_exposure():
    from architecture.toxacute_tasks import HUMAN_TARGET_TASKS
    from trainer import Trainer

    human_tasks = list(HUMAN_TARGET_TASKS)[:1]
    animal = "mouse_oral_LD50"
    tasks = [*human_tasks, animal]
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(seed=1, task_sampling="proportional")
    trainer.seed = 1
    trainer.task_name = tasks
    loaders = _dataloaders({human_tasks[0]: 2, animal: 18})

    schedule = trainer._task_schedule(loaders, epoch=0)
    diagnostics = trainer._schedule_diagnostics(schedule, loaders)

    assert diagnostics["total_updates"] == 20
    assert diagnostics["task_batches"] == {animal: 18, human_tasks[0]: 2}
    assert diagnostics["human3_fraction"] == pytest.approx(0.1)


def test_human_target_floor_appends_extra_human_passes():
    from architecture.toxacute_tasks import HUMAN_TARGET_TASKS
    from trainer import Trainer

    human_tasks = list(HUMAN_TARGET_TASKS)[:1]
    animal = "mouse_oral_LD50"
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(seed=1, task_sampling="human_target_floor")
    trainer.seed = 1
    trainer.task_name = [*human_tasks, animal]
    loaders = _dataloaders({human_tasks[0]: 3, animal: 15})

    schedule = trainer._task_schedule(loaders, epoch=0)
    diagnostics = trainer._schedule_diagnostics(schedule, loaders)

    assert schedule.count(human_tasks[0]) == 6  # proportional + one extra pass
    assert diagnostics["human3_fraction"] == pytest.approx(6 / 21)


import pytest  # noqa: E402  (kept at bottom to mirror fixture-style imports above)
