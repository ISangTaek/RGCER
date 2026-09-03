"""Unit tests for the D8 result-correction post-processing scripts.

Only the pure computational helpers are covered here (no model loads, no
datastore, no GPU): exact test statistics, prediction schema mapping,
degenerate-input handling, nesting ladders, manifest hashing and the
Figure 4 aggregation primitives.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from scripts.d8_drift_recompute import probe_manifest_sha256
from scripts.d8_exact_paired_stats import (
    compute_paired_stats,
    exact_sign_test_p_one_sided,
    exact_wilcoxon_p_one_sided,
)
from scripts.d8_label_scaling_mechanism import mean_std_median
from scripts.d8_label_scaling_subset_audit import nesting_ladder
from scripts.d8_recompute_prediction_metrics import (
    endpoint_metrics,
    macro_metrics,
    normalize_prediction_rows,
)


# ---------------------------------------------------------------------------
# Task B: exact paired statistics
# ---------------------------------------------------------------------------


def test_exact_wilcoxon_all_positive_is_1_over_32():
    gains = [0.1367, 0.1547, 0.1103, 0.1891, 0.1788]
    assert exact_wilcoxon_p_one_sided(gains) == pytest.approx(1 / 32)


def test_exact_wilcoxon_mixed_case_hand_computed():
    # ranks of |gains| = 1..5, W+ = 1 + 3 + 5 = 9;
    # 13 of the 32 sign patterns reach W+ >= 9.
    gains = [1.0, -2.0, 3.0, -4.0, 5.0]
    assert exact_wilcoxon_p_one_sided(gains) == pytest.approx(13 / 32)


def test_exact_wilcoxon_matches_scipy_when_available():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    for _ in range(20):
        gains = rng.normal(size=6).round(3)
        gains[0] = abs(gains[0])  # avoid zero-diff edge for a stable comparison
        expected = float(
            scipy_stats.wilcoxon(gains, alternative="greater", method="exact").pvalue
        )
        assert exact_wilcoxon_p_one_sided(gains) == pytest.approx(expected, abs=1e-12)


def test_exact_sign_test():
    assert exact_sign_test_p_one_sided([0.1, 0.2, 0.3, 0.4, 0.5]) == pytest.approx(1 / 32)
    assert exact_sign_test_p_one_sided([1.0, -2.0, 3.0, -4.0, 5.0]) == pytest.approx(0.5)


def test_compute_paired_stats_fields():
    gains = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats = compute_paired_stats(gains, [42, 43, 44, 45, 46], "test engine")
    assert stats["comparison"] == "S1_vs_B1"
    assert stats["metric"] == "StableRMSE"
    assert stats["wins"] == 5 and stats["ties"] == 0 and stats["losses"] == 0
    assert stats["wilcoxon_exact_one_sided_p"] == pytest.approx(1 / 32)
    assert stats["sign_test_exact_one_sided_p"] == pytest.approx(1 / 32)
    assert stats["test"] == "Wilcoxon signed-rank"
    assert stats["alternative"] == "greater"
    assert stats["method"] == "exact"


# ---------------------------------------------------------------------------
# Task A: prediction schema mapping + metric computation
# ---------------------------------------------------------------------------


def test_normalize_prediction_rows_maps_aliases():
    rows = [
        {"sample_id": "row_1", "task": "man_oral_TDLo", "target": "1.0", "prediction": "1.5"},
        {"sample_id": "row_2", "task": "man_oral_TDLo", "target": "2.0", "prediction": "1.0"},
    ]
    normalized, mapping = normalize_prediction_rows(rows)
    assert mapping["y_true"] == "target"
    assert mapping["y_pred"] == "prediction"
    assert normalized[0] == {
        "endpoint": "man_oral_TDLo", "sample_id": "row_1", "y_true": 1.0, "y_pred": 1.5,
    }


def test_normalize_prediction_rows_canonical_columns():
    rows = [
        {"sample_id": "row_1", "endpoint": "women_oral_TDLo",
         "label": "3.0", "final_prediction": "2.5"},
    ]
    normalized, mapping = normalize_prediction_rows(rows)
    assert mapping["y_true"] == "label"
    assert mapping["y_pred"] == "final_prediction"
    assert normalized[0]["y_true"] == 3.0


def test_normalize_prediction_rows_rejects_unknown_schema():
    with pytest.raises(ValueError):
        normalize_prediction_rows([{"foo": "1", "bar": "2"}])


def test_endpoint_metrics_known_values():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.5, 2.5, 2.5, 4.5]
    metrics = endpoint_metrics(y_true, y_pred)
    assert metrics["status"] == "OK"
    # residuals are +0.5, -0.5, +0.5, -0.5
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["mae"] == pytest.approx(0.5)
    # R2 = 1 - SS_res / SS_tot = 1 - 1.0 / 5.0
    assert metrics["r2"] == pytest.approx(0.8)
    assert -1.0 <= metrics["pearson_r"] <= 1.0
    assert metrics["pearson_r"] > 0.8
    assert 0.0 <= metrics["pearson_p"] <= 1.0


def test_endpoint_metrics_constant_input_is_nan_not_zero():
    metrics = endpoint_metrics([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    assert metrics["status"] == "CONSTANT_INPUT"
    assert math.isnan(metrics["pearson_r"])
    # RMSE/MAE remain well defined
    assert metrics["rmse"] > 0


def test_endpoint_metrics_insufficient_n():
    metrics = endpoint_metrics([1.0, 2.0], [1.5, 1.5])
    assert metrics["status"] == "INSUFFICIENT_N"
    assert math.isnan(metrics["pearson_r"])


def test_macro_metrics_ignores_nan_and_counts_valid():
    rows = [
        {"rmse": 1.0, "mae": 0.5, "pearson_r": 0.8, "spearman_rho": 0.7, "r2": 0.6},
        {"rmse": 2.0, "mae": 1.0, "pearson_r": float("nan"), "spearman_rho": 0.5, "r2": 0.2},
    ]
    macro = macro_metrics(rows)
    assert macro["macro_rmse"] == pytest.approx(1.5)
    assert macro["macro_pearson"] == pytest.approx(0.8)
    assert macro["pearson_valid_tasks"] == 1
    assert macro["spearman_valid_tasks"] == 2


# ---------------------------------------------------------------------------
# Task F: nesting ladder
# ---------------------------------------------------------------------------


def test_nesting_ladder_accepts_proper_subset_chain():
    ids = {10: {"a", "b"}, 25: {"a", "b", "c"}, 50: {"a", "b", "c", "d"},
           75: {"a", "b", "c", "d", "e"}, 100: {"a", "b", "c", "d", "e", "f"}}
    result = nesting_ladder(ids)
    assert result["nested"] is True
    assert result["rungs"]["nested_75_100"] is True


def test_nesting_ladder_rejects_violation():
    ids = {10: {"x"}, 25: {"a", "b"}, 50: {"a", "b", "c"}, 75: {"a", "b", "c", "d"},
           100: {"a", "b", "c", "d", "e"}}
    result = nesting_ladder(ids)
    assert result["nested"] is False
    assert result["rungs"]["nested_10_25"] is False


# ---------------------------------------------------------------------------
# Task C: probe manifest hashing + Figure 4 primitives
# ---------------------------------------------------------------------------


def test_probe_manifest_sha256_is_order_insensitive_and_stable():
    a = probe_manifest_sha256(["row_3", "row_1", "row_2"])
    b = probe_manifest_sha256(["row_2", "row_3", "row_1"])
    assert a == b
    assert len(a) == 64


def test_mean_std_median_known_values():
    mean, std, median = mean_std_median([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    assert std == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    assert median == pytest.approx(2.5)
