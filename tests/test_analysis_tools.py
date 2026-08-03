import torch

from analysis.interval_analysis import coverage_width, risk_coverage_curve
from analysis.negative_transfer import (
    intervention_masks,
    shuffled_endpoint_batches,
    shuffled_endpoint_name,
)


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


def test_shuffled_endpoint_uses_one_split_global_permutation():
    class Batch:
        def __init__(self, ids, values):
            self.sample_id = ids
            self.y = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)

        def __deepcopy__(self, memo):
            return Batch(list(self.sample_id), self.y.clone().reshape(-1).tolist())

    batches = [Batch(["a", "b"], [1.0, 2.0]), Batch(["c", "d"], [3.0, 4.0])]
    shuffled = list(shuffled_endpoint_batches(batches, seed=7))
    assert [value for batch in shuffled for value in batch.y.reshape(-1).tolist()] != [1.0, 2.0, 3.0, 4.0]
    repeated = list(shuffled_endpoint_batches(batches, seed=7))
    assert torch.equal(shuffled[0].y, repeated[0].y)
    assert torch.equal(shuffled[1].y, repeated[1].y)
