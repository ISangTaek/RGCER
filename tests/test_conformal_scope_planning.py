"""conformal_scope=all_tasks planning contract (review §19-21, §46).

``all_tasks`` must resolve to every task in the run — historically the
ad-hoc CLI mapping passed ``None``/``[]`` into the planner, so no
calibration floor was enforced for animal endpoints.  With the shared
resolver the planner must hard-fail any endpoint whose calibration
presence is below the alpha-derived minimum.
"""

import pandas as pd
import pytest

from conformal import resolve_conformal_tasks
from toxacute_datastore import plan_toxacute_split

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]


def _write_raw(path, rows=120):
    frame = pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(rows)],
            "Pubchem CID": list(range(rows)),
            "IUPAC Name": [""] * rows,
            "SMILES": [""] * rows,
            # Distinct acyclic molecules: each canonical SMILES is its own
            # scaffold group, so the constrained planner can distribute them.
            "smiles": [f"{'C' * (i + 1)}O" for i in range(rows)],
            "InChIKey": [""] * rows,
            "man_oral_TDLo": [2.0] * rows,
            "women_oral_TDLo": [2.0] * rows,
            "human_oral_TDLo": [2.0] * rows,
            # Animal endpoint with a single labelled row — below any floor.
            "rat_oral_LD50": [2.0 if i == 7 else None for i in range(rows)],
        }
    )
    frame.to_csv(path, index=False)


def _plan(raw_csv, scope):
    tasks = [
        "man_oral_TDLo",
        "women_oral_TDLo",
        "human_oral_TDLo",
        "rat_oral_LD50",
    ]
    return plan_toxacute_split(
        raw_csv,
        task_names=tasks,
        splitting="scaffold",
        valid_size=0.1,
        calibration_size=0.1,
        test_size=0.1,
        split_seed=42,
        conformal_alpha=0.34,  # minimum_calibration_size == 2
        conformal_scope=scope,
        priority_task_names=list(HUMAN_TASKS) if scope == "human3" else [],
        conformal_task_names=resolve_conformal_tasks(scope, tasks),
        num_candidates=8,
        min_priority_eval_count=2,
    )


def test_all_tasks_scope_hard_fails_endpoint_below_calibration_floor(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    _write_raw(raw_csv)

    # The planner fails hard before any LMDB write, either by raising on an
    # infeasible candidate search or by returning a FAIL report.
    with pytest.raises(ValueError, match="FAIL_CONFORMAL_RANK:task=rat_oral_LD50"):
        _plan(raw_csv, "all_tasks")


def test_human3_scope_ignores_below_floor_animal_endpoint(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    _write_raw(raw_csv)

    _, report = _plan(raw_csv, "human3")

    assert report["status"] == "PASS"
    assert report["hard_constraint_failures"] == []


def test_resolver_reports_every_task_under_all_tasks():
    tasks = ["man_oral_TDLo", "human_oral_TDLo", "rat_oral_LD50"]
    assert len(resolve_conformal_tasks("all_tasks", tasks)) == 3
