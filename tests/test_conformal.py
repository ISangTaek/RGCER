import pytest
import torch

from conformal import ConformalCalibrator


def test_conformal_rank_apply_and_state_roundtrip():
    calibrator = ConformalCalibrator(alpha=0.1, min_calibration_size=3)
    with pytest.warns(RuntimeWarning):
        calibrator.fit_task("task", torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0]))
    lower, upper = calibrator.apply("task", torch.tensor([0.0]), torch.tensor([0.0]))
    assert lower.item() <= 0.0 <= upper.item()
    state = calibrator.state_dict()
    restored = ConformalCalibrator(alpha=0.1, min_calibration_size=0)
    restored.load_state_dict(state)
    assert restored.states["task"].qhat == calibrator.states["task"].qhat


def test_conformal_without_calibration_samples_raises():
    with pytest.raises(ValueError):
        ConformalCalibrator().fit_task("empty", torch.tensor([]), torch.tensor([]), torch.tensor([]))
