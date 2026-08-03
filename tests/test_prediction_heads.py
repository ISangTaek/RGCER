import pytest
import torch

from architecture.prediction_heads import TaskPredictionHead, decode_prediction
from loss import QuantileRegressionLoss


def test_point_and_quantile_head_shapes_and_ordering():
    h = torch.randn(5, 12)
    point = TaskPredictionHead(12, mode="point")
    quantile = TaskPredictionHead(12, mode="quantile")
    assert point(h).shape == (5, 1)
    raw = quantile(h)
    decoded = decode_prediction(raw, "quantile")
    assert raw.shape == (5, 3)
    assert torch.all(decoded.lower <= decoded.median)
    assert torch.all(decoded.median <= decoded.upper)


def test_quantile_loss_and_invalid_mode():
    with pytest.raises(ValueError):
        TaskPredictionHead(8, mode="invalid")
    loss = QuantileRegressionLoss(0.05, 0.95)
    raw = torch.zeros(4, 3, requires_grad=True)
    value = loss.compute_loss(raw, torch.ones(4, 1))
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(raw.grad).all()
