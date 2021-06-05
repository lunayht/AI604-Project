import torch
import numpy as np

from typing import Optional

from torch import nn
from torch.nn import functional as F

"""Functional criterions for semantic segmentation tasks.

Args:
    - pred: torch.Tensor([N, C, H, W])
    - true: torch.Tensor([N, H, W])
Return:
    - loss: torch.Tensor([1])
"""

"""
    NOTE: For DiceLoss, use TverskyLoss with (alpha, beta) = (0.5, 0.5)
"""
class CELoss(nn.Module):
    """Cross Entropy Loss with logits (before softmax).
       Modularized to handle image input.
    """
    def __init__(
        self,
        ignore_index: Optional[int]=None
    ):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, true, weight: Optional[torch.Tensor]=None):
        assert pred.shape[0] == true.shape[0]
        C = pred.shape[1]
        pred = pred.permute(0, 2, 3, 1).reshape(-1, C)
        true = true.view(-1)
        return F.cross_entropy(
            pred, true,
            weight=weight,
            ignore_index=self.ignore_index
        )

class TverskyLoss(nn.Module):
    """Dice Loss with logits (before softmax)."""
    def __init__(
        self,
        alpha: float,
        beta: float,
        eps: float=1e-8,
        ignore_index: Optional[int]=None
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.ignore_index = ignore_index
    
    def forward(self, pred, true):
        assert pred.shape[0] == true.shape[0]
        C = pred.shape[1]
        # Flatten / one-hot
        pred = pred.permute(0, 2, 3, 1).reshape(-1, C)
        true = F.one_hot(true.view(-1), C)
        pred = pred.log_softmax(dim=1).exp()
        # Remove ignore_index
        if self.ignore_index is not None:
            mask = torch.ones(true.shape, dtype=torch.bool)
            mask[:, self.ignore_index] = False
            true = true[mask]
            pred = pred[mask]
        sect = torch.sum(true * pred)
        fp = torch.sum(pred * (-true + 1.0))
        fn = torch.sum((-pred + 1.0) * true)
        loss = 1. - sect / (sect + self.alpha * fp + self.beta * fn + self.eps)
        return loss

class FocalLoss(nn.Module):
    """Focal Loss with logits (before softmax)."""
    def __init__(
        self,
        gamma: float=0.1,
        alpha: Optional[float]=None,
        ignore_index: Optional[int]=None
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, pred, true):
        assert pred.shape[0] == true.shape[0]
        C = pred.shape[1]
        # Flatten / one-hot
        pred = pred.permute(0, 2, 3, 1).reshape(-1, C)
        true = F.one_hot(true.view(-1), C)
        pred = pred.log_softmax(dim=1)
        # Remove ignore_index
        if self.ignore_index is not None:
            mask = torch.ones(true.shape, dtype=torch.bool)
            mask[:, self.ignore_index] = False
            true = true[mask]
            pred = pred[mask]
        
        loss =  -(true * pred) * ((1 - pred) ** self.gamma) * true * self.alpha
        return loss.mean()

if __name__ == '__main__':
    pred = torch.randn([16, 22, 256, 256])
    true = torch.randint(10, [16, 256, 256])

    loss_1 = CELoss(ignore_index=0)
    loss_2 = FocalLoss(gamma=2, alpha=0.25, ignore_index=0)
    loss_3 = TverskyLoss(0.5, 0.5, ignore_index=0)
    print('Cross Entropy Loss: ', loss_1(pred, true))
    print('Focal Loss: ', loss_2(pred, true))
    print('Tversky Loss: ', loss_3(pred, true))