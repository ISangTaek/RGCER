import torch
import numpy as np

from weighting.abstract_weighting import AbsWeighting


class EW(AbsWeighting):
    r"""Equal Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """
    def __init__(self):
        super(EW, self).__init__()
        
    def backward(self, losses, active_mask=None, **kwargs):
        if active_mask is None:
            active_mask = torch.ones_like(losses, dtype=torch.bool)
        active_mask = active_mask.to(device=losses.device, dtype=torch.bool)
        active_count = active_mask.sum().clamp_min(1)
        weights = active_mask.to(losses.dtype) / active_count.to(losses.dtype)
        loss = torch.mul(losses, weights).sum()
        loss.backward()
        return weights.detach().cpu().numpy()
