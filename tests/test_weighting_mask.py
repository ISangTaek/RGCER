import torch

from weighting.EW import EW


def test_equal_weighting_ignores_inactive_tasks():
    weighting = EW()
    weighting.task_num = 3
    weighting.device = torch.device("cpu")
    losses = torch.tensor([1.0, 100.0, 3.0], requires_grad=True)
    active_mask = torch.tensor([True, False, True])
    weights = weighting.backward(losses, active_mask=active_mask)
    assert torch.allclose(losses.grad, torch.tensor([0.5, 0.0, 0.5]))
    assert weights.tolist() == [0.5, 0.0, 0.5]
