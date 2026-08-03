import torch

from analysis.interval_analysis import coverage_width, risk_coverage_curve
from analysis.negative_transfer import intervention_masks, shuffled_endpoint_name


def test_source_interventions_delete_one_real_source_per_sample():
    weights = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.0, 0.0]])
    masks = intervention_masks(weights, strategies=("top", "lowest"))
    top_mask, top_deleted = masks["top"]
    low_mask, low_deleted = masks["lowest"]
    assert top_deleted.tolist() == [0, -1]
    assert low_deleted.tolist() == [1, -1]
    assert top_mask[0].tolist() == [False, True, True]
    assert low_mask[0].tolist() == [True, False, True]


def test_interval_and_risk_coverage_helpers():
    summary = coverage_width(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 2.0]), torch.tensor([1.0, 3.0]), alpha=0.1)
    assert summary["coverage"] == 0.5
    assert summary["mean_width"] == 1.5
    curve = risk_coverage_curve(
        prediction=torch.tensor([1.0, 2.0, 10.0]),
        target=torch.tensor([1.0, 4.0, 8.0]),
        uncertainty=torch.tensor([0.1, 0.2, 0.9]),
        points=3,
    )
    assert curve["coverage"].is_monotonic_increasing
    assert shuffled_endpoint_name("task") == "shuffled_task"
