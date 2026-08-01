import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score

from metric import ClsMetric, RegMetric


def test_regression_metric_is_computed_over_the_full_epoch():
    metric = RegMetric()
    metric.update_fun(np.array([0.0]), np.array([0.0]))
    metric.update_fun(np.array([0.0, 20.0]), np.array([10.0, 20.0]))
    rmse, r2 = metric.score_fun()
    expected_rmse = np.sqrt(mean_squared_error([0.0, 10.0, 20.0], [0.0, 0.0, 20.0]))
    assert np.isclose(rmse, expected_rmse)
    assert np.isfinite(r2)


def test_classification_metric_handles_single_class_batches():
    metric = ClsMetric()
    metric.update_fun(np.array([0.1, 0.2]), np.array([0, 0]))
    metric.update_fun(np.array([0.8, 0.9]), np.array([1, 1]))
    auroc, _ = metric.score_fun()
    assert np.isclose(auroc, roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]))
