import torch
import torch.nn.functional as F

from weighting.abstract_weighting import AbsWeighting


class DWA(AbsWeighting):
    r"""Dynamic Weight Average (DWA).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 

    Args:
        T (float, default=2.0): The softmax temperature.

    """
    def __init__(self):
        super(DWA, self).__init__()
        
    def backward(self, losses, active_mask=None, **kwargs):
        if active_mask is None:
            active_mask = torch.ones_like(losses, dtype=torch.bool)
        active_mask = active_mask.to(device=losses.device, dtype=torch.bool)
        mask = active_mask.to(losses.dtype)
        active_count = mask.sum().clamp_min(1.0)
        T = 2.0
        if getattr(self, 'epoch', 0) > 1 and hasattr(self, 'train_loss_buffer'):
            w_i = torch.Tensor(
                self.train_loss_buffer[:, self.epoch-1] / self.train_loss_buffer[:,self.epoch-2]
            ).to(self.device)
            w_i = w_i.masked_fill(~active_mask, float('-inf'))
            batch_weight = active_count * F.softmax(w_i / T, dim=-1)
        else:
            batch_weight = mask / active_count * active_count

        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()
