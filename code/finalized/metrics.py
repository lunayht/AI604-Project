from typing import Optional

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

