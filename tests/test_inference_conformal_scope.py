"""Conformal scope resolution and inference gating (review §19-25, §47)."""

import pytest

from conformal import resolve_conformal_tasks

ALL_TASKS = [
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
    "mouse_oral_LD50",
    "rat_oral_LD50",
]


def test_human3_scope_restricts_to_human_tasks_present_in_run():
    assert resolve_conformal_tasks("human3", ALL_TASKS) == [
        "man_oral_TDLo",
        "women_oral_TDLo",
        "human_oral_TDLo",
    ]


def test_all_tasks_scope_covers_every_task_never_empty():
    resolved = resolve_conformal_tasks("all_tasks", ALL_TASKS)
    assert resolved == ALL_TASKS
    assert len(resolved) > 0


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="conformal_scope"):
        resolve_conformal_tasks("sometimes", ALL_TASKS)


def test_human3_scope_filters_to_tasks_in_smaller_run():
    assert resolve_conformal_tasks("human3", ["human_oral_TDLo"]) == ["human_oral_TDLo"]
    assert resolve_conformal_tasks("human3", ["mouse_oral_LD50"]) == []


class _Calibrator:
    def __init__(self, states):
        self.states = states


class _Trainer:
    def __init__(self, states):
        self.conformal_calibrator = _Calibrator(states)


class _Params:
    fit_conformal = True


def test_inference_gating_applies_conformal_only_where_qhat_exists():
    from main import _apply_conformal_for_task

    trainer = _Trainer(states={"human_oral_TDLo"})
    params = _Params()

    assert _apply_conformal_for_task(params, trainer, "human_oral_TDLo") is True
    assert _apply_conformal_for_task(params, trainer, "mouse_oral_LD50") is False


def test_inference_gating_respects_fit_conformal_false():
    from main import _apply_conformal_for_task

    trainer = _Trainer(states={"human_oral_TDLo"})
    params = _Params()
    params.fit_conformal = False

    assert _apply_conformal_for_task(params, trainer, "human_oral_TDLo") is False
