import torch
import torch.nn as nn

from weighting.abstract_weighting import AbsWeighting


class UW(AbsWeighting):
    r"""Uncertainty Weights (UW).
    
    This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018) <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`

    """
    def __init__(self):
        super(UW, self).__init__()
        
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([-0.5]*self.task_num, device=self.device))

    def backward(self, losses, active_mask=None, **kwargs):
        if active_mask is None:
            active_mask = torch.ones_like(losses, dtype=torch.bool)
        active_mask = active_mask.to(device=losses.device, dtype=torch.bool)
        mask = active_mask.to(losses.dtype)
        active_count = mask.sum().clamp_min(1.0)
        task_terms = losses / (2 * self.loss_scale.exp()) + self.loss_scale / 2
        loss = (task_terms * mask).sum() / active_count
        loss.backward()
        return (mask / (2 * torch.exp(self.loss_scale))).detach().cpu().numpy()
