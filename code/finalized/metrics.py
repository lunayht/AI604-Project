import torch

from typing import Optional

"""
    Computes segmentation metrics with confusion matrix.
"""

def compute_accuracy(
    matrix: "torch.Tensor",
    ignore_index: Optional[int]=None
):
    if ignore_index is not None:
        matrix = matrix[torch.arange(matrix.shape[0]) != ignore_index]
        matrix = matrix[torch.arange(matrix.shape[1]) != ignore_index]
    acc = matrix.diag().sum() / (matrix.sum() + 1e-15)
    return acc

def compute_dice(
    matrix: "torch.Tensor",
    ignore_index: Optional[int]=None,
    reduction: str='mean'
):
    assert reduction in ('mean', 'sum', 'none')
    dice = 2.0 * matrix.diag() / (matrix.sum(dim=1) + matrix.sum(dim=0) + 1e-15)

    if ignore_index is not None:
        if ignore_index >= len(dice):
            raise ValueError(
                "ignore_index is larger than the length of the dice vector ..."
            )   
        indices = list(range(len(dice)))
        indices.remove(ignore_index)
        dice = dice[indices]
    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()   
    else:
        return dice

def compute_miou(
    matrix: "torch.Tensor",
    ignore_index: Optional[int]=None
):
    iou = matrix.diag() / (matrix.sum(dim=1) + matrix.sum(dim=0) - matrix.diag() + 1e-15)
    if ignore_index is not None:
        if ignore_index >= len(iou):
            raise ValueError(
                "ignore_index is larger than the length of the iou vector ..."
            )
        indices = list(range(len(iou)))
        indices.remove(ignore_index)
        iou = iou[indices]
    return iou.mean()