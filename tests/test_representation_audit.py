"""Representation-path audit metrics and human3-only aggregation (D3 §39-§42).

The plan requires a regression test where human3 RMSE ~= 1.2 and animal RMSE
~= 0.5 must aggregate base_human3_rmse from the human tasks alone — never the
all-task macro — and the audit metrics must decompose the FiLM/adapter chain
per sample.
"""

import math

import pytest
import torch
from torch import nn

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS
from representation_audit import (
    REPRESENTATION_METRIC_FIELDS,
    compute_representation_metrics,
)
from run_diagnostics import RunDiagnosticsWriter


def _model_with_router():
    """A minimal RGCER-shaped stub exposing the conditioner submodules the
    audit helper reaches into (prompt bank, response encoder, value projection).
    """

    class _PromptBank(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompts = nn.Parameter(torch.randn(3, 4))

        def forward(self):
            return self.prompts

    class _ResponseEncoder(nn.Module):
        def forward(self, profile):
            return profile.expand(-1, -1, 4) * 0.5

    class _Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_source_response = True
            self.response_encoder = _ResponseEncoder()
            self.value_projection = nn.Linear(4, 4, bias=False)

    class _Conditioner(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_bank = _PromptBank()
            self.router = _Router()
            self.response_stacking = None

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.task_conditioner = _Conditioner()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()

    return _Model()


def _fake_diagnostics(batch=2, hidden=4):
    h = torch.randn(batch, hidden)
    gamma = torch.randn(batch, hidden) * 0.1
    beta = torch.randn(batch, hidden) * 0.1
    adapter_output = torch.randn(batch, hidden) * 0.2
    weights = torch.softmax(torch.randn(batch, 3), dim=-1)
    return {
        "base_representation": h,
        "task_context": torch.randn(batch, hidden),
        "gamma": gamma,
        "beta": beta,
        "adapter_output": adapter_output,
        "route_representation": h + adapter_output,
        "source_weights": weights,
        "response_profile": torch.randn(batch, 3, 1),
        "routing_entropy": torch.rand(batch),
    }


def test_representation_metrics_shapes_and_ratios():
    model = _model_with_router()
    diagnostics = _fake_diagnostics(batch=3)
    metrics = compute_representation_metrics(model, "man_oral_TDLo", diagnostics)

    for field in REPRESENTATION_METRIC_FIELDS:
        if field == "stacked_response_norm":
            # Only produced by the response_stacking conditioner.
            assert field not in metrics
            continue
        assert field in metrics, field
        assert metrics[field].numel() == 3

    assert torch.allclose(
        metrics["film_delta_norm"],
        (diagnostics["gamma"] * diagnostics["base_representation"] + diagnostics["beta"])
        .float()
        .norm(dim=-1),
        atol=1e-5,
    )
    assert torch.allclose(
        metrics["adapter_delta_norm"], diagnostics["adapter_output"].float().norm(dim=-1), atol=1e-5
    )
    assert torch.allclose(
        metrics["route_rep_delta_norm"], metrics["adapter_delta_norm"], atol=1e-5
    )
    assert torch.allclose(
        metrics["routing_variance"],
        diagnostics["source_weights"].var(dim=-1, unbiased=False),
        atol=1e-6,
    )
    assert metrics["source_context_ratio"].min() >= 0.0


def test_representation_metrics_handles_target_only_style_diagnostics():
    model = _model_with_router()
    model.encoder.task_conditioner.router = None  # target_only/response_stacking
    diagnostics = _fake_diagnostics(batch=2)
    diagnostics["source_weights"] = torch.zeros(2, 3)
    diagnostics["response_profile"] = torch.zeros(2, 3, 1)
    metrics = compute_representation_metrics(model, "human_oral_TDLo", diagnostics)
    assert metrics["h_norm"].numel() == 2
    assert "source_context_norm" not in metrics
    assert math.isfinite(float(metrics["film_delta_norm"].mean()))


def test_epoch_summary_human3_macro_excludes_animal_tasks(tmp_path):
    """Plan §40: human3 ~= 1.2, animal ~= 0.5 — base_human3_rmse must be 1.2."""

    writer = RunDiagnosticsWriter(
        tmp_path / "diagnostics",
        task_names=list(HUMAN_TARGET_TASKS) + ["mouse_oral_LD50"],
        sample_row_index={},
    )
    epoch_path = {
        **{task: {"base": {"rmse": 1.2, "r2": 0.0, "mae": 1.0, "n": 10},
                  "route": {"rmse": 1.2, "r2": 0.0, "mae": 1.0, "n": 10},
                  "final": {"rmse": 1.2, "r2": 0.0, "mae": 1.0, "n": 10}}
           for task in HUMAN_TARGET_TASKS},
        "mouse_oral_LD50": {"base": {"rmse": 0.5, "r2": 0.0, "mae": 0.4, "n": 10},
                            "route": {"rmse": 0.5, "r2": 0.0, "mae": 0.4, "n": 10},
                            "final": {"rmse": 0.5, "r2": 0.0, "mae": 0.4, "n": 10}},
    }
    assert writer._human3_macro(epoch_path, "base", "rmse") == pytest.approx(1.2)
    assert writer._human3_macro(epoch_path, "route", "rmse") == pytest.approx(1.2)
    assert writer._human3_macro(epoch_path, "final", "rmse") == pytest.approx(1.2)


def test_writer_writes_representation_epoch_and_best_summary(tmp_path):
    writer = RunDiagnosticsWriter(
        tmp_path / "diagnostics",
        task_names=list(HUMAN_TARGET_TASKS),
        sample_row_index={"s_1": 5},
    )
    rep_records = {
        task: {
            "sample_id": ["s_1"],
            "metrics": [compute_representation_metrics(_model_with_router(), task, _fake_diagnostics(batch=1))],
        }
        for task in HUMAN_TARGET_TASKS
    }
    train_result = {
        "human3_macro_rmse": 1.1,
        "all_task_macro_rmse": 1.1,
        "loss": {task: 1.0 for task in HUMAN_TARGET_TASKS},
        "final_loss": {task: 1.0 for task in HUMAN_TARGET_TASKS},
        "base_loss": {task: 1.0 for task in HUMAN_TARGET_TASKS},
    }
    validation_result = {
        "tasks": {},
        "score": 0.0,
        "selection_score": -1.1,
        "selection_scope": "human3",
        "human3_macro_rmse": 1.1,
        "all_task_macro_rmse": 1.1,
        "routing": {},
    }
    route_records = {
        task: {
            "sample_id": ["s_1"],
            "base": [torch.tensor([[1.0]])],
            "route": [torch.tensor([[1.2]])],
            "final": [torch.tensor([[1.1]])],
            "target": [torch.tensor([[0.9]])],
            "route_regret": [torch.tensor([[0.1]])],
            "final_regret": [torch.tensor([[0.0]])],
            "null": [torch.tensor([[0.0]])],
            "entropy": [torch.tensor([[0.5]])],
            "source_weights": [torch.tensor([[0.5, 0.5, 0.0]])],
            "joint_source_weights": [torch.tensor([[0.4, 0.4, 0.0]])],
        }
        for task in HUMAN_TARGET_TASKS
    }
    writer.note_best_epoch(0, route_records, representation_records=rep_records)
    writer.log_epoch(
        0,
        train_result,
        validation_result,
        route_records=route_records,
        routing_enabled=True,
        is_best=True,
        representation_records=rep_records,
    )
    writer.write_best_artifacts(0)

    epoch_rows = (tmp_path / "diagnostics" / "representation_epoch.csv").read_text().strip().splitlines()
    assert len(epoch_rows) == 1 + len(HUMAN_TARGET_TASKS)
    header = epoch_rows[0].split(",")
    assert "h_norm_mean" in header and "film_delta_ratio_mean" in header

    summary_rows = (tmp_path / "diagnostics" / "representation_path_summary.csv").read_text().strip().splitlines()
    assert len(summary_rows) == 1 + len(HUMAN_TARGET_TASKS)
    assert "route_minus_base_abs_mean" in summary_rows[0]
