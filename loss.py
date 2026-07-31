import numpy as np
import torch.nn as nn


class BCELoss():
    r"""The BCE loss founction.
    """
    def __init__(self):
        self.record = []
        self.bs = []
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss
    
    def update_loss(self, pred, gt):
        r"""
        """
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss
    
    def average_loss(self):
        r"""
        """
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()
    
    def reinit(self):
        r"""
        """
        self.record = []
        self.bs = []


class MSELoss():
    r"""The MSE loss founction.
    """
    def __init__(self):
        self.record = []
        self.bs = []
        self.loss_fn = nn.MSELoss()
    
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss
    
    def update_loss(self, pred, gt):
        r"""
        """
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss
    
    def average_loss(self):
        r"""
        """
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()
    
    def reinit(self):
        r"""
        """
        self.record = []
        self.bs = []