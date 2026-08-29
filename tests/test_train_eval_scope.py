"""Diagnostic-phase contracts: validation-only scope and read-only logging.

Plan §5/§6-§9: ``--train_eval_scope validation_only`` must hand the trainer
only train+validation loaders (calibration/test never reach it, CQR never
fits, no test evaluation), and the diagnostic writer must emit per-epoch
artifacts plus best-epoch per-sample dumps without touching model math.
"""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import main as main_module
from run_diagnostics import RunDiagnosticsWriter
from trainer import Trainer


class _Batch:
    def __init__(self, value):
        self.y = torch.tensor([[float(value)]])
        self.sample_id = [f"sample_{value}"]

    def get(self, key, default=None):
        return default

    def to(self, device):
        return self


class _Loader:
    def __init__(self, tag):
        self.tag = tag
        self.iterations = 0

    def __len__(self):
        return 1

    def __iter__(self):
        self.iterations += 1
        return iter([_Batch(0.0)])


def test_scoped_loader_dicts_isolate_calibration_and_test():
    loaders = {
        "train": {"t": "train_loader"},
        "val": {"t": "val_loader"},
        "calibration": {"t": "calibration_loader"},
        "test": {"t": "test_loader"},
    }
    train_loaders, val_loaders, calibration, test = main_module._scoped_loader_dicts(
        loaders, "validation_only"
    )
    assert train_loaders is loaders["train"]
    assert val_loaders is loaders["val"]
    assert calibration is None
    assert test is None

    train_loaders, val_loaders, calibration, test = main_module._scoped_loader_dicts(loaders, "full")
    assert calibration is loaders["calibration"]
    assert test is loaders["test"]


def _make_trainer(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        selection_scope="human3",
        lower_quantile=0.05,
        upper_quantile=0.95,
        ckpt_name="model",
        diagnostics=True,
        routing_enabled=True,
        hps_warmup_epochs=0,
    )
    trainer.selection_scope = "human3"
    trainer.task_name = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
    trainer.task_num = len(trainer.task_name)
    trainer.task_dict = {task: {"metrics": ["RMSE"], "weight": [-1]} for task in trainer.task_name}
    trainer.task_scalers = {task: {"mean": 0.0, "std": 1.0} for task in trainer.task_name}
    trainer.save_path = tmp_path
    trainer.seed = 42
    trainer.device = torch.device("cpu")
    trainer.conformal_scope = "human3"
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
    trainer.sample_row_index = {}
    trainer.conformal_calibrator = SimpleNamespace(states={})
    trainer._save_checkpoint = lambda epoch, filename, **kwargs: Path(
        f"ckpt_epoch{epoch}_{filename}"
    )
    trainer.load_checkpoint = lambda path: None

    def _unexpected_fit(calibration_dataloaders_dict, routing_enabled_override=None):
        raise AssertionError("conformal must not be fitted under validation_only")

    trainer._fit_conformal = _unexpected_fit

    def _train_epoch(loaders, epoch):
        return {
            "tasks": {task: {"RMSE": 1.0} for task in trainer.task_name},
            "score": 1.0,
            "loss": {task: 1.0 for task in trainer.task_name},
            "final_loss": {task: 0.75 for task in trainer.task_name},
            "base_loss": {task: 0.5 for task in trainer.task_name},
            "updates": 1,
            "schedule_usage": {},
            "routing": {},
        }

    trainer._train_epoch = _train_epoch
    return trainer


def _validation_result(rmse):
    return {
        "tasks": {task: {"RMSE": rmse} for task in HUMAN_TASKS},
        "score": 1.0,
        "selection_score": -rmse,
        "selection_scope": "human3",
        "selection_tasks": list(HUMAN_TASKS),
        "human3_macro_rmse": rmse,
        "all_task_macro_rmse": rmse,
        "routing": {},
    }


HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]


def test_train_validation_only_never_touches_calibration_or_test(tmp_path):
    trainer = _make_trainer(tmp_path)

    touched = []

    def _evaluate(loaders, mode="validation", epoch=0, routing_enabled_override=None):
        touched.append((mode, tuple(sorted(loaders))))
        return _validation_result(2.0 - 0.1 * epoch)

    trainer._evaluate = _evaluate

    train_loaders = {task: _Loader("train") for task in trainer.task_name}
    val_loaders = {task: _Loader("val") for task in trainer.task_name}

    history = trainer.train(
        train_loaders,
        val_loaders,
        calibration_dataloaders_dict=None,
        test_dataloaders_dict=None,
        epochs=2,
    )

    assert len(history) == 2
    assert trainer.final_test_result is None
    assert trainer.conformal_calibrator.states == {}
    assert {mode for mode, _ in touched} == {"validation"}
    # metrics.json assembly in main() only adds "test" when final_test_result
    # is set — asserted here via the trainer contract.


def test_train_full_scope_fits_conformal_and_evaluates_test(tmp_path):
    trainer = _make_trainer(tmp_path)

    fit_calls = []

    def _fit_conformal(calibration_dataloaders_dict, routing_enabled_override=None):
        fit_calls.append(tuple(sorted(calibration_dataloaders_dict)))

    trainer._fit_conformal = _fit_conformal
    evaluated = []

    def _evaluate(loaders, mode="validation", epoch=0, routing_enabled_override=None):
        if mode == "test":
            evaluated.append(mode)
            return _validation_result(1.0)
        return _validation_result(2.0)

    trainer._evaluate = _evaluate

    train_loaders = {task: _Loader("train") for task in trainer.task_name}
    val_loaders = {task: _Loader("val") for task in trainer.task_name}
    calibration_loaders = {task: _Loader("calibration") for task in trainer.task_name}
    test_loaders = {task: _Loader("test") for task in trainer.task_name}

    trainer.train(
        train_loaders,
        val_loaders,
        calibration_dataloaders_dict=calibration_loaders,
        test_dataloaders_dict=test_loaders,
        epochs=1,
    )

    assert fit_calls and fit_calls[0] == tuple(sorted(trainer.task_name))
    assert evaluated == ["test"]
    assert trainer.final_test_result is not None


def _route_record(values, with_sources=True):
    tensor = lambda rows: [torch.tensor([[value]]) for value in rows]
    record = {
        "sample_id": [f"s_{int(value)}" for value in values],
        "base": tensor(values),
        "route": tensor([value + 0.5 for value in values]),
        "final": tensor([value + 0.25 for value in values]),
        "target": tensor([value + 1.0 for value in values]),
        "route_regret": tensor([-0.25 for _ in values]),
        "final_regret": tensor([-0.5 for _ in values]),
        "entropy": tensor([0.7 for _ in values]),
    }
    if with_sources:
        # One [batch, task] tensor per validation batch — the writer must
        # concatenate before indexing samples globally (regression guard for
        # the flattened-column bug that produced "column_496" style names).
        record["null"] = [torch.tensor([[0.2]]), torch.tensor([[0.3]])]
        record["source_weights"] = [
            torch.tensor([[0.6, 0.4, 0.0]]),
            torch.tensor([[0.3, 0.7, 0.0]]),
        ]
        record["joint_source_weights"] = [
            torch.tensor([[0.48, 0.32, 0.0]]),
            torch.tensor([[0.2, 0.6, 0.0]]),
        ]
    return record


def test_diagnostics_writer_writes_epoch_and_best_artifacts(tmp_path):
    writer = RunDiagnosticsWriter(
        tmp_path / "diagnostics",
        task_names=HUMAN_TASKS,
        sample_row_index={"s_1": 11, "s_2": 22},
    )
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoders = nn.ModuleDict({"task": nn.Linear(2, 2)})
            self.encoder = nn.ModuleDict({"backbone": nn.Linear(2, 2)})

    model = _Model()
    model.decoders["task"](torch.ones(3, 2)).sum().backward()
    writer.record_gradients(model)
    train_result = {
        "human3_macro_rmse": 1.5,
        "all_task_macro_rmse": 1.7,
        "loss": {task: 1.0 for task in HUMAN_TASKS},
        "final_loss": {task: 0.75 for task in HUMAN_TASKS},
        "base_loss": {task: 0.5 for task in HUMAN_TASKS},
    }
    validation_result = _validation_result(1.25)
    validation_result["routing"] = {
        task: {"mean_null_weight": 0.3, "std_null": 0.1, "mean_transfer_mass": 0.7}
        for task in HUMAN_TASKS
    }
    route_records = {task: _route_record([1.0, 2.0]) for task in HUMAN_TASKS}
    writer.note_best_epoch(0, route_records)
    writer.log_epoch(
        0,
        train_result,
        validation_result,
        route_records=route_records,
        routing_enabled=False,
        is_best=True,
    )
    writer.write_best_artifacts(0)

    with (tmp_path / "diagnostics" / "epoch_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["epoch"] == "0"
    assert rows[0]["routing_enabled"] == "False"
    assert rows[0]["is_best"] == "True"
    assert float(rows[0]["base_human3_rmse"]) == pytest.approx(1.0)
    assert float(rows[0]["route_human3_rmse"]) == pytest.approx(0.5)
    assert float(rows[0]["final_human3_rmse"]) == pytest.approx(0.75)
    assert float(rows[0]["human3_mean_null_weight"]) == pytest.approx(0.3)

    with (tmp_path / "diagnostics" / "human3_path_metrics.csv").open() as handle:
        path_rows = list(csv.DictReader(handle))
    assert {(row["task"], row["path"]) for row in path_rows} == {
        (task, path) for task in HUMAN_TASKS for path in ("base", "route", "final")
    }

    with (tmp_path / "diagnostics" / "best_validation_human3_predictions.csv").open() as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == 6  # 3 tasks x 2 samples
    first = prediction_rows[0]
    assert first["sample_id"] == "s_1"
    assert first["row_index"] == "11"
    assert float(first["route_regret"]) == pytest.approx(-0.25)

    with (tmp_path / "diagnostics" / "best_validation_human3_routing.csv").open() as handle:
        routing_rows = list(csv.DictReader(handle))
    assert len(routing_rows) == 6
    by_sample = {(row["task"], row["sample_id"]): row for row in routing_rows}
    first = by_sample[(HUMAN_TASKS[0], "s_1")]
    assert first["top1_source"] == HUMAN_TASKS[0]
    assert float(first["top1_conditional_weight"]) == pytest.approx(0.6)
    assert float(first["top1_joint_weight"]) == pytest.approx(0.48)
    assert float(first["transfer_mass"]) == pytest.approx(0.8)
    # The second sample lives in the second validation batch: its top-1 source
    # must resolve through the concatenated weights, not batch-relative ones.
    second = by_sample[(HUMAN_TASKS[0], "s_2")]
    assert second["top1_source"] == HUMAN_TASKS[1]
    assert float(second["top1_conditional_weight"]) == pytest.approx(0.7)
    assert float(second["top1_joint_weight"]) == pytest.approx(0.6)
    assert float(second["transfer_mass"]) == pytest.approx(0.7)

    frequency = json.loads(
        (tmp_path / "diagnostics" / "routing_source_frequency.json").read_text()
    )
    for task in HUMAN_TASKS:
        entry = frequency["tasks"][task]
        assert entry["samples"] == 2
        assert entry["unique_top1_sources"] == 2
        assert entry["top1_source_counts"][HUMAN_TASKS[0]] == 1
        assert entry["top1_source_counts"][HUMAN_TASKS[1]] == 1

    with (tmp_path / "diagnostics" / "gradient_norms.csv").open() as handle:
        gradient_rows = list(csv.DictReader(handle))
    by_group = {row["group"]: row for row in gradient_rows}
    assert float(by_group["prediction_heads"]["last"]) > 0.0
    assert float(by_group["backbone"]["last"]) == 0.0  # no grad flowed there
    assert math.isfinite(float(by_group["router_query_projection"]["mean"])) is False
