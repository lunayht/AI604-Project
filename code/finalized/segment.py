import os
from re import M
import yaml
import wandb
import torch
import pytorch_lightning as pl
import albumentations as alb

from einops import rearrange

from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from torchmetrics import MetricCollection, ConfusionMatrix

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from simple_parsing import ArgumentParser

from models.transformer import ViTForSegmentation

from arguments import SegmentationArgs
from metrics import compute_accuracy, compute_dice, compute_miou
from loader import CrowdsourcingDataset, SegmentationBatchCollector

class SegmentationModel(pl.LightningModule):
    def __init__(self, args: "Namespace"):
        super().__init__()

        self.args = args
        self.num_classes = args.num_classes

        self.lr = args.learning_rate

        self.model = ViTForSegmentation(args)
        self.weight = None
        if args.loss_type == "weighted_cross_entropy":
            self.weight = torch.Tensor([0, 0.9333333333333333, 0.8666666666666667, 0.6666666666666667, 0.8, 0.7333333333333334])

        # Metric trackers
        metrics = {
            'confusion_matrix': ConfusionMatrix(
                compute_on_step=False,
                dist_sync_on_step=True,
                num_classes=self.num_classes,
                normalize=None
            )
        }
        self.train_metrics = MetricCollection(metrics, prefix='train_')
        self.valid_metrics = MetricCollection(metrics, prefix='valid_')
    
    def setup(self, stage):
        if stage == 'fit':
            self.train_steps = len(self.train_dataloader()) * self.args.max_epochs // self.args.accumulate_grad_batches

    def forward(self, x):
        return self.model(x)

    def criterion(
        self,
        logits, labels,
        ignore_index: int=0
    ):
        """
            Returns loss according to configuration.

            Args:
                - logits: torch.Tensor([b * h * w, c])
                - labels: torch.Tensor([b * h * w])
                - ignore_index: int = 0
            Returns:
                - losses: torch.Tensor([1])

            NOTE: Dice is a special case of Tversky (alpha = 0.5, beta = 0.5)
        """

        loss_type = self.args.loss_type
        if loss_type in ("weighted_cross_entropy", "cross_entropy"):
            return F.cross_entropy(
                logits, labels,
                weight=self.weight.to(logits.device) if self.weight is not None else None,
                ignore_index=ignore_index
            )
        labels = F.one_hot(labels, logits.shape[-1])
        logits = logits.log_softmax(dim=-1).exp()
        if ignore_index is not None:
            mask = torch.ones(labels.shape, dtype=torch.bool)
            mask[:, ignore_index] = False
            logits = logits[mask]
            labels = labels[mask]
        if loss_type == "dice":
            it = torch.sum(logits * labels)
            fp = torch.sum(logits * (1. - labels))
            fn = torch.sum(labels * (1. - logits))
            return 1. - it / (it + 0.5 * fp + 0.5 * fn + 1e-15)
        elif loss_type == "focal":
            ls = -(logits * labels) * ((1. - logits) ** self.args.gamma) * labels * self.args.alpha
            return ls.mean()
        elif loss_type == "tversky":
            it = torch.sum(logits * labels)
            fp = torch.sum(logits * (1. - labels))
            fn = torch.sum(logits * (1. - logits))
            return 1. - it / (it + self.args.alpha * fp + self.args.beta * fn + 1e-15)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        logits = rearrange(logits, 'b c h w -> (b h w) c')
        labels = rearrange(labels, 'b h w -> (b h w)')
        losses = self.criterion(logits, labels)
        self.log('train_loss', losses, on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.train_metrics(logits.softmax(-1), labels)
        return losses

    def training_epoch_end(self, outputs):
        scores = self.train_metrics.compute()
        matrix = scores.pop('train_confusion_matrix')
        scores['train_pixel_acc'] = compute_accuracy(matrix)
        scores['train_miou'] = compute_miou(matrix)
        scores['train_dice'] = compute_dice(matrix)
        self.train_metrics.reset()
        self.log_dict(scores)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        logits = rearrange(logits, 'b c h w -> (b h w) c')
        labels = rearrange(labels, 'b h w -> (b h w)')
        self.valid_metrics(logits.softmax(-1), labels)        

    def validation_epoch_end(self, outputs):
        scores = self.valid_metrics.compute()
        matrix = scores.pop('valid_confusion_matrix')
        scores['valid_pixel_acc'] = compute_accuracy(matrix)
        scores['valid_miou'] = compute_miou(matrix)
        scores['valid_dice'] = compute_dice(matrix)
        self.valid_metrics.reset()
        self.log_dict(scores)

    def configure_optimizers(self):
        # Optimizer w/ weight decay
        no_decay  = ['bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [p for n,p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {    
                'params': [p for n,p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = getattr(torch.optim, self.args.optimizer)
        if self.args.optimizer == 'AdamW':
            optimizer = optimizer(
                grouped_params, lr=self.args.learning_rate,
                betas=(self.args.beta_1, self.args.beta_2),
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = optimizer(grouped_params, lr=args.learning_rate)
        # Scheduler (linear schedule w/ torch.optim.lr_scheduler.LambdaLR)
        train_steps = self.train_steps
        warmup_steps = int(train_steps * self.args.warmup_proportion)
        def poly_schedule(current_step: int):
            if current_step > train_steps:
                return 1e-6
            return (1 - current_step / train_steps) ** 0.9

        def linear_schedule(current_step: int):
            """Function for LambdaLR. Must take current_step (int) as input."""
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0., float(train_steps - current_step) / float(max(1, train_steps - warmup_steps)))
            
        scheduler = LambdaLR(
            optimizer,
            linear_schedule if args.scheduler == 'linear' else poly_schedule,
            -1
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, args: "Namespace"):
        super().__init__()

        self.args = args

        self.train_ds = CrowdsourcingDataset(data_path=os.path.join(args.data_dir, 'train'))
        self.valid_ds = CrowdsourcingDataset(data_path=os.path.join(args.data_dir, 'test'))
        
        # Transforms
        augmentation = None
        if args.augmentation:
            augmentation = alb.Compose([
                alb.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03
                ),
                alb.Rotate(),
                alb.Flip()
            ])
        self.train_cl = SegmentationBatchCollector(augmentation=augmentation)
        self.valid_cl = SegmentationBatchCollector(augmentation=None)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            collate_fn=self.train_cl,
            num_workers=self.args.workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.args.batch_size,
            collate_fn=self.valid_cl,
            num_workers=self.args.workers,
            shuffle=False
        )

if __name__ == '__main__':
    # Argument parsing
    parser = ArgumentParser()
    config = yaml.load(open('config_seg.yml').read(), Loader=yaml.Loader)

    args = parser.add_arguments(SegmentationArgs(config=config), dest='options')
    args = parser.parse_args().options

    # Seeding and PL model
    pl.seed_everything(args.seed, workers=True)
    pl_model = SegmentationModel(args)

    # WandB logger
    run_name = 'seg-{}-{}-{}{}-{}-{}'.format(
        args.model_name,
        'pretrained' if args.pretrained else 'scratch',
        args.data_dir.split('/')[-1],
        '-aug' if args.augmentation else '',
        args.head_type,
        args.loss_type
    )

    if args.wandb:
        logger = WandbLogger(project='medical-transformers', name=run_name)
    else:
        logger = TensorBoardLogger(save_dir='logs/{}.log'.format(run_name))
    logger.log_hyperparams(vars(args))

    # Checkpointer
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_dice',
        dirpath='checkpoints',
        every_n_train_steps=0,
        every_n_val_epochs=1,
        filename='{epoch:02d}-{valid_dice:.2f}',
        save_top_k=1,
        mode='max'
    )
    pl_data = SegmentationDataModule(args)
    
    print(args)
    
    # Trainer
    trainer_args = {
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'amp_backend': args.amp_backend if args.amp else None,
        'auto_lr_find': True,
        'callbacks': [checkpoint_callback],
        'gradient_clip_val': args.grad_max_norm if args.grad_clipping else 0.0,
        'gpus': args.gpus if args.distributed else 1, 
        'logger': logger,
        'log_every_n_steps': args.log_interval,
        'max_epochs': args.max_epochs,
        'accelerator': 'ddp' if args.distributed else None,
        'plugins': DDPPlugin(find_unused_parameters=True) if args.distributed else None,
        'precision': 16 if args.amp else 32,
        'val_check_interval': args.val_check_interval,
        'deterministic': True
    }
    # Single GPU training
    if not args.distributed:
        trainer_args.pop('accelerator')
    trainer  = pl.Trainer(**trainer_args)

    # trainer.tune(pl_model, train_loader, valid_loader)
    trainer.fit(pl_model, pl_data)
