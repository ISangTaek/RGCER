"""Manifest-level distribution statistics and guard rules used by preflight."""

from data_preflight import (
    DataPreflightError,
    _acyclic_concentration_error,
    _manifest_distribution_stats,
)
import pytest


def _record(sample_id, split, scaffold="C1CCCCC1", canonical=None):
    row_number = int(sample_id.split("_")[1])
    return {
        "sample_id": sample_id,
        "row_index": row_number,
        "split": split,
        "canonical_smiles": canonical or f"C{row_number}CC",
        "scaffold": scaffold,
        "split_group": ("scaffold::" + scaffold) if scaffold != "__ACYCLIC__" else None,
    }


def test_distribution_reports_actual_counts_and_deviations():
    records = []
    # 70 train / 10 val / 10 cal / 10 test over distinct cyclic scaffolds.
    plan = {"train": 70, "validation": 10, "calibration": 10, "test": 10}
    index = 0
    for split, count in plan.items():
        for _ in range(count):
            records.append(_record(f"row_{index}", split, scaffold=f"S{index}"))
            index += 1
    manifest = {
        "manifest_version": 2,
        "splitting": "scaffold",
        "seed": 42,
        "ratios": {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
        "records": records,
    }

    stats = _manifest_distribution_stats(manifest)
    assert stats["actual_split_counts"] == plan
    assert abs(stats["actual_split_ratios"]["train"] - 0.7) < 1e-9
    assert all(abs(v) < 1e-6 for v in stats["ratio_deviation_pct"].values())
    assert stats["largest_split_group"]["size"] == 1
    assert stats["num_unique_canonical_smiles"] == 100
    assert stats["num_duplicate_canonical_groups"] == 0


def test_distribution_counts_acyclic_by_split():
    records = [_record("row_0", "train", scaffold="__ACYCLIC__", canonical="CCC")]
    records.append(_record("row_1", "test", scaffold="__ACYCLIC__", canonical="CCO"))
    records.append(_record("row_2", "test", scaffold="__ACYCLIC__", canonical="CCC"))
    manifest = {"records": records}

    stats = _manifest_distribution_stats(manifest)
    assert stats["acyclic_count"] == 3
    assert stats["acyclic_by_split"] == {"train": 1, "validation": 0, "calibration": 0, "test": 2}
    # Two of three acyclic rows share one canonical structure.
    assert stats["num_duplicate_canonical_groups"] == 1
    assert stats["duplicate_canonical_row_count"] == 2


def test_duplicate_rows_of_one_canonical_form_one_largest_group():
    canonical = "CCCCCCCC"
    records = [
        _record("row_201", "train", canonical=canonical),
        _record("row_202", "train", canonical=canonical),
        _record("row_203", "train", canonical=canonical),
        _record("row_204", "test", scaffold="S1"),
    ]
    stats = _manifest_distribution_stats({"records": records})
    largest = stats["largest_split_group"]
    assert largest["size"] == 3
    assert largest["split"] == "train"
    assert stats["num_duplicate_canonical_groups"] == 1


def _distribution_with_acyclic(acyclic_by_split):
    total_others = {"train": 500, "validation": 50, "calibration": 50, "test": 50}
    counts = dict(total_others)
    for name in acyclic_by_split:
        counts[name] += 1
    return {
        "acyclic_count": sum(acyclic_by_split.values()),
        "acyclic_by_split": {
            **{name: 0 for name in ("train", "validation", "calibration", "test")},
            **acyclic_by_split,
        },
        "_counts": counts,
    }


def test_acyclic_concentration_hard_fails_when_two_eval_splits_miss_acyclic():
    distribution = _distribution_with_acyclic({"train": 150})
    with pytest.raises(DataPreflightError, match="acyclic"):
        raise DataPreflightError(_acyclic_concentration_error(distribution))


def test_acyclic_concentration_allows_small_or_well_spread_sets():
    # Small acyclic population is only informational.
    small = _distribution_with_acyclic({"train": 80})
    assert _acyclic_concentration_error(small) is None

    spread = _distribution_with_acyclic({"train": 200, "validation": 30, "calibration": 25, "test": 40})
    assert _acyclic_concentration_error(spread) is None

    # Two missing eval splits hard-fail; a single one stays warning-level.
    single_missing = _distribution_with_acyclic({"train": 280, "validation": 15, "calibration": 5})
    assert _acyclic_concentration_error(single_missing) is None

    double_missing = _distribution_with_acyclic({"train": 320})
    error = _acyclic_concentration_error(double_missing)
    assert error is not None and "acyclic" in error


def test_concentration_rule_reads_nested_payload_format_used_in_reports():
    payload = {"acyclic_count": 120, "acyclic_by_split": {"train": 120}}
    assert _acyclic_concentration_error(payload) is not None
