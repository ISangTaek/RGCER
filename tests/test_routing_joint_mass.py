"""Joint/conditional/NULL routing artifact contract (P1-3)."""

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.routing_analysis import (
    _as_joint_weights,
    _check_joint_invariant,
    collect_routing_records,
    fake_endpoint_stress_summary,
    routing_summary,
)
from architecture.response_guided_router import RGCERDiagnostics


def _records(rows=3, with_joint=True, fake_task="shuffled_mouse_oral_LD50"):
    task_names = ["human_oral_TDLo", "mouse_oral_LD50", fake_task]
    data = {
        "null_weight": [0.9, 0.5, 0.1][:rows],
        "route::human_oral_TDLo": [0.0, 0.2, 0.4][:rows],
        "conditional_route::human_oral_TDLo": [0.0, 0.2, 0.4][:rows],
        "joint_route::human_oral_TDLo": [0.005, 0.05, 0.3][:rows] if with_joint else None,
        "route::mouse_oral_LD50": [0.06, 0.15, 0.35][:rows],
        "conditional_route::mouse_oral_LD50": [0.06, 0.15, 0.35][:rows],
        "joint_route::mouse_oral_LD50": [0.03, 0.2, 0.25][:rows] if with_joint else None,
        f"route::{fake_task}": [0.04, 0.3, 0.25][:rows],
        f"conditional_route::{fake_task}": [0.04, 0.3, 0.25][:rows],
        f"joint_route::{fake_task}": [0.0004, 0.006, 0.05][:rows] if with_joint else None,
    }
    return pd.DataFrame({k: v for k, v in data.items() if v is not None})


def test_records_expose_conditional_and_joint_columns_with_alias():
    records = _records()
    for task in ("human_oral_TDLo", "mouse_oral_LD50"):
        assert f"route::{task}" in records.columns
        assert f"conditional_route::{task}" in records.columns
        assert f"joint_route::{task}" in records.columns
    # route:: is the conditional alias, kept for existing notebooks.
    assert np.allclose(
        records["route::human_oral_TDLo"], records["conditional_route::human_oral_TDLo"]
    )


def test_joint_invariant_sum_plus_null_is_one():
    joint = torch.tensor([[0.55, 0.20], [0.30, 0.69]])
    null = torch.tensor([[0.25], [0.01]])
    _check_joint_invariant(joint, null)  # 0.75+0.25 and 0.99+0.01

    broken_null = torch.tensor([[0.40], [0.01]])
    with pytest.raises(RuntimeError, match="routing invariant violated"):
        _check_joint_invariant(joint, broken_null)


def test_as_joint_weights_validates_shape():
    diagnostics = {"joint_source_weights": torch.zeros(2, 3)}
    assert _as_joint_weights(diagnostics, 2, 3, torch.device("cpu")).shape == (2, 3)
    with pytest.raises(ValueError, match="does not match"):
        _as_joint_weights({"joint_source_weights": torch.zeros(4, 3)}, 2, 3, torch.device("cpu"))
    assert _as_joint_weights({}, 2, 3, torch.device("cpu")) is None


def test_stress_summary_uses_joint_fake_mass_not_conditional():
    """§8: NULL-dominant rows must read as low dependence despite a high
    conditional weight on the fake endpoint."""

    summary = fake_endpoint_stress_summary(
        _records(),
        auxiliary_tasks=["shuffled_mouse_oral_LD50"],
    )
    # Row 1: null=0.9, conditional_fake=0.30 but joint_fake only 0.0040.
    assert summary["mean_joint_fake_mass"] == pytest.approx((0.0004 + 0.006 + 0.05) / 3)
    assert summary["mean_conditional_fake_mass"] > 10 * summary["mean_joint_fake_mass"]
    assert summary["mean_null_weight"] == pytest.approx(0.5)
    assert summary["transfer_mass"] == pytest.approx(0.5)
    assert summary["per_auxiliary_joint_mass"]["shuffled_mouse_oral_LD50"] == pytest.approx(0.0564)


def test_stress_summary_empty_without_auxiliary_joint_columns():
    assert fake_endpoint_stress_summary(_records(with_joint=False), ["nope"]) == {}
    assert fake_endpoint_stress_summary(pd.DataFrame(), ["x"]) == {}


class _RoutingModel(torch.nn.Module):
    """Minimal RGCER-shaped model producing valid routing diagnostics."""

    def __init__(self):
        super().__init__()
        self.task_names = ["t_a", "t_b"]

    def forward(self, batch, task_name=None, return_all_tasks=False, return_aux=False, **kwargs):
        batch_size = batch.y.size(0)
        count = len(self.task_names)
        raw = {"t_a": torch.full((batch_size, 1), 0.5), "t_b": torch.full((batch_size, 1), -0.5)}
        target_index = self.task_names.index(task_name)
        # One transfer unit is split evenly across the allowed (non-target)
        # sources; conditional weights renormalize the same support to 1.0.
        live = [i for i in range(count) if i != target_index]
        joint_values = {i: 0.9 / len(live) for i in live}
        conditional_values = {i: 1.0 / len(live) for i in live}
        joint = torch.zeros(batch_size, count)
        conditional = torch.zeros(batch_size, count)
        for index, weight in joint_values.items():
            joint[:, index] = weight
        for index, weight in conditional_values.items():
            conditional[:, index] = weight
        fixed_null = torch.full((batch_size, 1), 0.10)
        diagnostics = {
            task: RGCERDiagnostics(
                target_task=task,
                target_index=self.task_names.index(task),
                response_profile=torch.zeros(batch_size, count, 1),
                source_weights=conditional,
                joint_source_weights=joint,
                null_weight=fixed_null,
                routing_entropy=torch.zeros(batch_size),
                task_context=torch.zeros(batch_size, 4),
                gamma=torch.zeros(batch_size, 4),
                beta=torch.zeros(batch_size, 4),
            )
            for task in self.task_names
        }
        predictions = {task: raw[task] for task in self.task_names}
        return predictions, diagnostics[task_name]

    def source_policy_mask(self):
        return None


class _Batch(dict):
    def __init__(self, size):
        super().__init__(is_empty=False)
        self.y = torch.zeros(size, 1)

    def get(self, key, default=None):
        return dict.get(self, key, default)

    def to(self, device):
        return self


@pytest.mark.parametrize("target", ["t_a", "t_b"])
def test_collect_routing_records_enforces_invariant_and_writes_joint(target):
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return _Batch(1)

    records = collect_routing_records(
        _RoutingModel(),
        DataLoader(_DS(), batch_size=None, shuffle=False),
        target_task=target,
        task_names=["t_a", "t_b"],
        prediction_mode="point",
    )
    assert "joint_route::t_a" in records.columns
    totals = records[[c for c in records.columns if c.startswith("joint_route::")]].sum(axis=1)
    assert np.allclose(totals + records["null_weight"], 1.0, atol=1e-4)
    summary = routing_summary(records, task_names=["t_a", "t_b"])
    assert summary["mean_null_weight"] == pytest.approx(summary["mean_null"])
    assert summary["mean_joint_source_mass"] <= 1.0