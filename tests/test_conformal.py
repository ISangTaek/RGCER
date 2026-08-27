import pytest
import torch

from conformal import ConformalCalibrator, InsufficientCalibrationError, exchangeability_metadata


def test_conformal_rank_apply_and_state_roundtrip():
    calibrator = ConformalCalibrator(alpha=0.1, min_calibration_size=3)
    lower_i, upper_i, target_i = _scores(12)
    calibrator.fit_task("task", lower_i, upper_i, target_i)
    lower, upper = calibrator.apply("task", torch.tensor([0.0]), torch.tensor([4.0]))
    assert lower.item() <= upper.item()
    state = calibrator.state_dict()
    restored = ConformalCalibrator(alpha=0.1, min_calibration_size=0)
    restored.load_state_dict(state)
    assert restored.states["task"].qhat == calibrator.states["task"].qhat


def test_conformal_without_calibration_samples_raises():
    with pytest.raises(ValueError):
        ConformalCalibrator().fit_task("empty", torch.tensor([]), torch.tensor([]), torch.tensor([]))


def test_conformal_keeps_signed_negative_scores():
    scores = ConformalCalibrator.conformity_scores(
        torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([1.0])
    )
    assert scores.item() == -1.0


def _scores(n):
    return torch.arange(float(n)), torch.arange(float(n)) + 2.0, torch.arange(float(n)) + 1.0


def test_conformal_rejects_impossible_finite_rank():
    """alpha=.05 with n=14 needs corrected rank 15 > n: no silent clipping."""

    lower, upper, target = _scores(14)
    with pytest.raises(InsufficientCalibrationError, match="rank 15 > n"):
        ConformalCalibrator(alpha=0.05).fit_task("thin", lower, upper, target)


def test_conformal_alpha_010_n14_is_finite():
    """alpha=.10 needs rank ceil(15*.9)=14 <= 14: still valid, exactly at limit."""

    lower, upper, target = _scores(14)
    state = ConformalCalibrator(alpha=0.10).fit_task("edge", lower, upper, target)
    assert int(state.count) == 14
    # qhat equals the largest signed score at this limiting configuration.
    scores = ConformalCalibrator.conformity_scores(lower, upper, target)
    assert abs(state.qhat - float(scores.max())) < 1e-6


def test_insufficient_calibration_error_names_minimum_sample_count():
    try:
        ConformalCalibrator(alpha=0.05).fit_task("t", *_scores(18))
    except InsufficientCalibrationError as exc:
        message = str(exc)
        assert "ceil((1-alpha)/alpha)=19" in message
        assert exc.corrected_rank == 19 and exc.count == 18
    else:
        pytest.fail("expected InsufficientCalibrationError")


def test_exchangeability_metadata_tracks_split_semantics():
    scaffold = exchangeability_metadata("scaffold")
    assert scaffold["finite_sample_exchangeability_guarantee_applicable"] is False
    assert "empirical" in scaffold["coverage_interpretation"]

    random = exchangeability_metadata("random")
    assert random["finite_sample_exchangeability_guarantee_applicable"] is True
    assert "exchangeability" in random["coverage_interpretation"]

    unknown = exchangeability_metadata(None)
    assert unknown["finite_sample_exchangeability_guarantee_applicable"] is False
