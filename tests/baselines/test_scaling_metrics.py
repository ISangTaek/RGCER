import numpy as np
import pytest

from baselines.metrics import regression_metrics
from baselines.scaling import TaskScaler


def test_scaler_is_train_only_ddof_zero_and_masks_nan():
    labels = np.asarray([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    scaler = TaskScaler.fit(labels, ("a", "b"), allow_empty=True)
    assert scaler.means.tolist() == [2.0, 0.0]
    assert scaler.stds.tolist() == [1.0, 1.0]
    assert scaler.counts.tolist() == [2, 0]
    scaled, mask = scaler.transform(labels)
    assert np.isfinite(scaled).all()
    assert mask[:, 1].sum() == 0


def test_empty_task_rejected_when_not_explicitly_allowed():
    with pytest.raises(ValueError, match="No finite training labels"):
        TaskScaler.fit(np.asarray([[np.nan]], dtype=np.float32), ("a",), allow_empty=False)


def test_metrics_use_equal_task_macro_not_pooled_rmse():
    truth = np.asarray([[0.0, 0.0], [2.0, np.nan], [np.nan, 0.0]])
    pred = np.asarray([[0.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
    metrics = regression_metrics(truth, pred, ("a", "b"))
    assert metrics["macro_rmse"] == pytest.approx((np.sqrt(2.0) + 2.0) / 2.0)
    assert metrics["pooled_rmse"] == pytest.approx(np.sqrt(3.0))


def test_r2_is_null_with_machine_readable_reason_for_constant_target():
    metrics = regression_metrics(np.ones((2, 1)), np.zeros((2, 1)), ("a",))
    assert metrics["per_task"]["a"]["r2"] is None
    assert metrics["per_task"]["a"]["r2_reason"] == "constant_targets"
