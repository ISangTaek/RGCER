"""D6 aggregation contracts (review P0-2/3/4, P1-7/8/9, §75-§76).

Uses synthetic run directories to prove: per-seed run resolution is exact,
incomplete trajectories are rejected, Stage-C StableGain has the correct sign
(RMSE lower is better), the Stage-C endpoint gate is enforced, Stage A stops
without enough eligible original candidates, and Stage C rejects a
non-original winner.
"""

import json

import pytest

from scripts.d6_aggregate import (
    candidate_run_dir,
    stage_a,
    stage_b,
    stage_c,
)

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]


def _write_run(
    root,
    candidate,
    tag,
    seed,
    macro_by_epoch,
    endpoint_by_epoch=None,
):
    diagnostics = root / candidate / tag / f"seed_{seed}" / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    with (diagnostics / "epoch_summary.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,val_human3_macro_rmse\n")
        for epoch, value in sorted(macro_by_epoch.items()):
            handle.write(f"{epoch},{value}\n")
    with (diagnostics / "human3_path_metrics.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,task,path,rmse\n")
        for epoch, per_task in (endpoint_by_epoch or {}).items():
            for task, rmse in per_task.items():
                handle.write(f"{epoch},{task},final,{rmse}\n")


def _flat_macro(last_epoch, value):
    return {epoch: value for epoch in range(last_epoch + 1)}


def test_candidate_run_dir_is_seed_specific(tmp_path):
    for seed in (42, 44, 46):
        (tmp_path / "o1" / "d6_o1_e40" / f"seed_{seed}").mkdir(parents=True)
    assert candidate_run_dir(tmp_path, "o1", "d6_o1_e40", 42) == tmp_path / "o1" / "d6_o1_e40" / "seed_42"
    assert candidate_run_dir(tmp_path, "o1", "d6_o1_e40", 44).name == "seed_44"
    assert candidate_run_dir(tmp_path, "o1", "d6_o1_e40", 46).name == "seed_46"
    assert candidate_run_dir(tmp_path, "o1", "d6_o1_e40", 45) is None
    assert candidate_run_dir(tmp_path, "o1", "d6_missing_e40", 42) is None


def test_summarize_run_requires_complete_stage_a(tmp_path):
    _write_run(tmp_path, "b1", "d6_b1_e20", 42, {epoch: 1.0 for epoch in range(0, 15)})
    from scripts.d6_aggregate import summarize_run

    row = summarize_run(
        tmp_path / "b1" / "d6_b1_e20" / "seed_42", "b1", 42, expected_last_epoch=19
    )
    assert row["status"] == "INCOMPLETE_TRAJECTORY"


def test_summarize_run_requires_complete_stage_b(tmp_path):
    _write_run(tmp_path, "b1", "d6_b1_e40", 42, {epoch: 1.0 for epoch in range(0, 8)})
    from scripts.d6_aggregate import summarize_run

    row = summarize_run(
        tmp_path / "b1" / "d6_b1_e40" / "seed_42", "b1", 42, expected_last_epoch=39
    )
    assert row["status"] == "INCOMPLETE_TRAJECTORY"


def test_stage_b_reads_three_different_seeds(tmp_path):
    # Distinct stable RMSE per seed proves each seed directory was read.
    for seed, value in ((42, 1.05), (44, 1.06), (46, 1.07)):
        _write_run(
            tmp_path, "b0", "d6_b0_e40", seed, _flat_macro(39, 1.10),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(35, 40)
            },
        )
        _write_run(
            tmp_path, "o1", "d6_o1_e40", seed, _flat_macro(39, value),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 0.9, "women_oral_TDLo": 0.9, "human_oral_TDLo": 0.9}
                for epoch in range(35, 40)
            },
        )
        # O1's gate now requires the anchor-only control (review P0-4/§25-§26).
        _write_run(
            tmp_path, "b0a", "d6_b0a_e40", seed, _flat_macro(39, value + 0.02),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(35, 40)
            },
        )
        _write_run(
            tmp_path, "o2", "d6_o2_e40", seed, _flat_macro(39, value + 0.05),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.2, "women_oral_TDLo": 1.2, "human_oral_TDLo": 1.2}
                for epoch in range(35, 40)
            },
        )
    trace = {}
    top1 = stage_b(tmp_path, ["o1", "o2"], [42, 44, 46], tmp_path, trace)
    assert top1 == "o1"
    paired = [
        row
        for row in csv_rows(tmp_path / "D6B_PAIRED_COMPARISON.csv")
        if row["candidate"] == "o1"
    ]
    assert {int(row["seed"]) for row in paired} == {42, 44, 46}
    assert {float(row["b0_stable_rmse"]) for row in paired} == {1.10}
    assert {round(float(row["stable_gain"]), 6) for row in paired} == {
        round(1.10 - 1.05, 6), round(1.10 - 1.06, 6), round(1.10 - 1.07, 6),
    }


def csv_rows(path):
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_stage_c_gain_is_positive_when_candidate_is_better(tmp_path):
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        _write_run(
            tmp_path, "b0", "d6_b0_e100", seed, _flat_macro(99, 1.10),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(95, 100)
            },
        )
        _write_run(
            tmp_path, "o1", "d6_o1_e100", seed, _flat_macro(99, 1.00),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 0.9, "women_oral_TDLo": 0.9, "human_oral_TDLo": 1.2}
                for epoch in range(95, 100)
            },
        )
    trace = {}
    stage_c(tmp_path, "o1", seeds, tmp_path, trace)
    gate = trace["stage_c"]["gate"]
    assert gate["paired_mean_stable_gain"] == pytest.approx(0.10)
    assert gate["positive_seeds"] == 5
    assert gate["gate_pass"] is True


def test_stage_c_rejects_worse_candidate(tmp_path):
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        _write_run(
            tmp_path, "b0", "d6_b0_e100", seed, _flat_macro(99, 1.10),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(95, 100)
            },
        )
        _write_run(
            tmp_path, "o1", "d6_o1_e100", seed, _flat_macro(99, 1.20),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.3, "women_oral_TDLo": 1.3, "human_oral_TDLo": 1.3}
                for epoch in range(95, 100)
            },
        )
    trace = {}
    stage_c(tmp_path, "o1", seeds, tmp_path, trace)
    gate = trace["stage_c"]["gate"]
    assert gate["paired_mean_stable_gain"] == pytest.approx(-0.10)
    assert gate["gate_pass"] is False


def test_stage_c_requires_endpoint_gate(tmp_path):
    seeds = [42, 43, 44, 45, 46]
    # Macro improves (+0.10) but only man_oral_TDLo is non-worse.
    for seed in seeds:
        _write_run(
            tmp_path, "b0", "d6_b0_e100", seed, _flat_macro(99, 1.10),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(95, 100)
            },
        )
        _write_run(
            tmp_path, "o1", "d6_o1_e100", seed, _flat_macro(99, 1.00),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 0.8, "women_oral_TDLo": 1.3, "human_oral_TDLo": 1.3}
                for epoch in range(95, 100)
            },
        )
    trace = {}
    stage_c(tmp_path, "o1", seeds, tmp_path, trace)
    gate = trace["stage_c"]["gate"]
    assert gate["paired_mean_stable_gain"] > 0
    assert gate["endpoints_non_worse"] == 1
    assert gate["gate_pass"] is False


def test_stage_a_stops_without_original_candidate(tmp_path):
    # B1 passes (diagnostic), O1 eliminated by the §31 margin, O2/O3 missing.
    _write_run(
        tmp_path, "b0", "d6_b0_e20", 42, _flat_macro(19, 1.10),
        endpoint_by_epoch={
            epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
            for epoch in range(15, 20)
        },
    )
    _write_run(
        tmp_path, "b1", "d6_b1_e20", 42, _flat_macro(19, 1.09),
        endpoint_by_epoch={
            epoch: {"man_oral_TDLo": 0.9, "women_oral_TDLo": 0.9, "human_oral_TDLo": 0.9}
            for epoch in range(15, 20)
        },
    )
    _write_run(
        tmp_path, "o1", "d6_o1_e20", 42, _flat_macro(19, 1.15),
        endpoint_by_epoch={
            epoch: {"man_oral_TDLo": 1.3, "women_oral_TDLo": 1.3, "human_oral_TDLo": 1.3}
            for epoch in range(15, 20)
        },
    )
    trace = {}
    top2 = stage_a(tmp_path, 42, tmp_path, trace)
    assert top2 == ["b1"]
    assert trace["stage_a"]["top2_insufficient"] is True
    assert "STOP" in trace["stage_a"]["stop_reason"]
    summary = {row["candidate"]: row for row in csv_rows(tmp_path / "D6A_MICROSCREEN_SUMMARY.csv")}
    assert summary["o1"]["status"] == "ELIMINATED"


def test_stage_b_unknown_candidate_fails_fast(tmp_path):
    with pytest.raises(SystemExit):
        stage_b(tmp_path, ["unknown"], [42, 44, 46], tmp_path, {})
    with pytest.raises(SystemExit):
        stage_b(tmp_path, ["o1", "o1"], [42, 44, 46], tmp_path, {})


def test_stage_c_rejects_non_original_winner(tmp_path):
    with pytest.raises(SystemExit):
        stage_c(tmp_path, "b1", [42, 43, 44, 45, 46], tmp_path, {})
    with pytest.raises(SystemExit):
        stage_c(tmp_path, "unknown", [42, 43, 44, 45, 46], tmp_path, {})


def test_stage_c_missing_one_seed_cannot_pass(tmp_path):
    # Review P0-5: four better seeds plus one missing seed must FAIL — Stage C
    # is the 5-seed final validation.
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        _write_run(
            tmp_path, "b0", "d6_b0_e100", seed, _flat_macro(99, 1.10),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(95, 100)
            },
        )
    for seed in seeds[:-1]:  # seed 46 missing
        _write_run(
            tmp_path, "o1", "d6_o1_e100", seed, _flat_macro(99, 1.00),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 0.9, "women_oral_TDLo": 0.9, "human_oral_TDLo": 0.9}
                for epoch in range(95, 100)
            },
        )
    trace = {}
    stage_c(tmp_path, "o1", seeds, tmp_path, trace)
    gate = trace["stage_c"]["gate"]
    assert gate["complete_seed_pairs"] is False
    assert gate["expected_seed_count"] == 5
    assert gate["gate_pass"] is False


def test_stage_b_requires_exactly_two_candidates(tmp_path):
    with pytest.raises(SystemExit):
        stage_b(tmp_path, ["o1"], [42, 44, 46], tmp_path, {})
    with pytest.raises(SystemExit):
        stage_b(tmp_path, ["o1", "o2", "b1"], [42, 44, 46], tmp_path, {})


def test_stage_b_stops_when_b1_clearly_dominates_original(tmp_path):
    # Review P0-6: B1 = 0.90 clearly beats O1 = 1.07 (B0 = 1.10).
    for seed in (42, 44, 46):
        _write_run(
            tmp_path, "b0", "d6_b0_e40", seed, _flat_macro(39, 1.10),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(35, 40)
            },
        )
        _write_run(
            tmp_path, "b1", "d6_b1_e40", seed, _flat_macro(39, 0.90),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 0.8, "women_oral_TDLo": 0.8, "human_oral_TDLo": 0.8}
                for epoch in range(35, 40)
            },
        )
        _write_run(
            tmp_path, "o1", "d6_o1_e40", seed, _flat_macro(39, 1.07),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 0.9, "women_oral_TDLo": 0.9, "human_oral_TDLo": 0.9}
                for epoch in range(35, 40)
            },
        )
        _write_run(
            tmp_path, "b0a", "d6_b0a_e40", seed, _flat_macro(39, 1.09),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(35, 40)
            },
        )
    trace = {}
    top1 = stage_b(tmp_path, ["o1", "b1"], [42, 44, 46], tmp_path, trace)
    assert top1 is None
    assert trace["stage_b"]["b1_dominates"] is True
    assert "dominates" in trace["stage_b"]["top1_reason"]


def test_stage_b_allows_original_when_tied_with_b1(tmp_path):
    # B1 = 1.01 vs O1 = 1.015: within the 0.01 tolerance → tied, original wins
    # on simplicity/originality (§35).
    for seed in (42, 44, 46):
        _write_run(
            tmp_path, "b0", "d6_b0_e40", seed, _flat_macro(39, 1.10),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(35, 40)
            },
        )
        _write_run(
            tmp_path, "b1", "d6_b1_e40", seed, _flat_macro(39, 1.01),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 0.95, "women_oral_TDLo": 0.95, "human_oral_TDLo": 0.95}
                for epoch in range(35, 40)
            },
        )
        _write_run(
            tmp_path, "o1", "d6_o1_e40", seed, _flat_macro(39, 1.015),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 0.9, "women_oral_TDLo": 0.9, "human_oral_TDLo": 0.9}
                for epoch in range(35, 40)
            },
        )
        _write_run(
            tmp_path, "b0a", "d6_b0a_e40", seed, _flat_macro(39, 1.03),
            endpoint_by_epoch={
                epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
                for epoch in range(35, 40)
            },
        )
    trace = {}
    top1 = stage_b(tmp_path, ["o1", "b1"], [42, 44, 46], tmp_path, trace)
    assert top1 == "o1"
    assert trace["stage_b"]["b1_dominates"] is False
    assert trace["stage_b"]["tie_break"] is None  # O1 strictly better than B1 here


def test_o1_requires_anchor_control(tmp_path):
    # Review §72: without a passing B0A control, O1's gain cannot be attributed
    # to the semantic delta and it must not be promoted.
    _write_run(
        tmp_path, "b0", "d6_b0_e20", 42, _flat_macro(19, 1.10),
        endpoint_by_epoch={
            epoch: {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.0, "human_oral_TDLo": 1.0}
            for epoch in range(15, 20)
        },
    )
    _write_run(
        tmp_path, "b1", "d6_b1_e20", 42, _flat_macro(19, 1.09),
        endpoint_by_epoch={
            epoch: {"man_oral_TDLo": 0.9, "women_oral_TDLo": 0.9, "human_oral_TDLo": 0.9}
            for epoch in range(15, 20)
        },
    )
    # O1 beats B0 clearly but the B0A anchor control is missing entirely.
    _write_run(
        tmp_path, "o1", "d6_o1_e20", 42, _flat_macro(19, 1.05),
        endpoint_by_epoch={
            epoch: {"man_oral_TDLo": 0.9, "women_oral_TDLo": 0.9, "human_oral_TDLo": 0.9}
            for epoch in range(15, 20)
        },
    )
    trace = {}
    top2 = stage_a(tmp_path, 42, tmp_path, trace)
    summary = {row["candidate"]: row for row in csv_rows(tmp_path / "D6A_MICROSCREEN_SUMMARY.csv")}
    assert summary["o1"]["meets_original_minimum"] in ("False", "")
    assert "o1" not in top2
