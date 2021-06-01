import torch
import torchmetrics
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from typing import Optional

def one_hot(
    labels: "torch.Tensor",
    num_classes: int,
    device: Optional["torch.device"]=None,
    dtype: Optional["torch.dtype"]=None,
    eps: Optional[float]=1e-6
):
    """
        Utility function for use along with DiceLoss. Taken from the source
        code of PyTorch Geometric.

            - https://github.com/rusty1s/pytorch_geometric
    """
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

class DiceLoss(nn.Module):
    """
        Differentiable DiceLoss for segmentation training. Taken from the
        source code of PyTorch Geometric.

            - https://github.com/rusty1s/pytorch_geometric
    """

    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super().__init__()
        self.eps: float = 1e-6

    def forward(
        self,
        input: "torch.Tensor",
        target: "torch.Tensor"
    ):
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,   
        args: "Namespace",
        model: "torch.nn.Module"
    ):
        super().__init__()

        self.args = args
        self.num_classes = args.num_classes_classification

        self.lr = args.learning_rate

        self.model = model
        if args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = DiceLoss()
        
        # Metric trackers
        train_metrics = \
            {
                'pixel_acc': torchmetrics.Accuracy(
                    compute_on_step=False,
                    num_classes=self.num_classes,
                ),
                'iou': torchmetrics.IoU(
                    compute_on_step=False,
                    num_classes=self.num_classes
                ),
                'dice': torchmetrics.F1(
                    compute_on_step=False,
                    num_classes=self.num_classes
                )
            }
        valid_metrics = \
            {
                'pixel_acc': torchmetrics.Accuracy(
                    compute_on_step=False,
                    num_classes=self.num_classes,
                ),
                'iou': torchmetrics.IoU(
                    compute_on_step=False,
                    num_classes=self.num_classes
                ),
                'dice': torchmetrics.F1(
                    compute_on_step=False,
                    num_classes=self.num_classes
                )
            }
        self.train_metrics = nn.ModuleDict(train_metrics)
        self.valid_metrics = nn.ModuleDict(valid_metrics)

    def setup(self, stage):
        if stage == 'fit':
            self.train_steps = len(self.train_dataloader()) * self.args.max_epochs // self.args.accumulate_grad_batches
            
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        train_loss = self.criterion(logits, labels)
        logits = F.softmax(logits, dim=-1)
        return {'loss': train_loss, 'pred': logits, 'label': labels}
    
    def training_step_end(self, outputs):
        self.log('train_loss', outputs['loss'], on_step=True, prog_bar=True, logger=True)
        for k in self.train_metrics.keys():
            self.train_metrics[k].update(outputs['pred'], outputs['label'])
        return outputs['loss']

    def training_epoch_end(self, *args, **kwargs):
        for k in self.train_metrics.keys():
            score = self.train_metrics[k].compute()
            self.log('train_{}'.format(k), score, on_epoch=True)
            self.train_metrics[k].reset()             

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        valid_loss = self.criterion(logits, labels)
        logits = F.softmax(logits, dim=-1)
        return {'loss': valid_loss, 'pred': logits, 'label': labels}

    def validation_step_end(self, outputs):
        self.log('valid_loss', outputs['loss'], on_step=True, prog_bar=True, logger=True)
        for k in self.valid_metrics.keys():
            score = self.valid_metrics[k].update(outputs['pred'], outputs['label'])
        return outputs['loss']

    def validation_epoch_end(self, *args, **kwargs):
        for k in self.valid_metrics.keys():
            score = self.valid_metrics[k].compute()
            self.log('valid_{}'.format(k), score, on_epoch=True)
            self.valid_metrics[k].reset()    

    def configure_optimizers(self):
        # Optimizer w/ weight decay
        no_decay  = ['bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [p for n,p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {    
                'params': [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(
            grouped_params, lr=self.lr,
            betas=(self.args.beta_1, self.args.beta_2),
            weight_decay=self.args.weight_decay
        )
        # Scheduler (linear schedule w/ torch.optim.lr_scheduler.LambdaLR)
        train_steps = self.train_steps
        warmup_steps = train_steps * self.args.warmup_proportion
        def linear_schedule(current_step: int):
            """Function for LambdaLR. Must take current_step (int) as input."""
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0., float(train_steps - current_step) / float(max(1, train_steps - warmup_steps)))
        scheduler = LambdaLR(optimizer, linear_schedule, -1)
        return { 'optimizer': optimizer, 'lr_scheduler':  scheduler }
