from os import confstr
import cv2
import yaml
import wandb
import torch
import numpy as np
import pytorch_lightning as pl
import albumentations as alb

from math import ceil
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import random_split, DataLoader

from torchmetrics import MetricCollection, Accuracy, F1, ConfusionMatrix

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from simple_parsing import ArgumentParser

from arguments import ClassificationArgs
from models.transformer.vit import ViTForClassification
from loader import ClassificationBatchCollector, BreakHisDataset, LABELS, BIN_LABELS


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,   
        args: "Namespace"
    ):
        super().__init__()

        self.args = args
        self.num_classes = args.num_classes

        self.lr = args.learning_rate

        self.model = ViTForClassification(args)
        self.criterion = nn.CrossEntropyLoss()

        # Metric trackers
        metrics = {
            'top1': Accuracy(
                compute_on_step=False,
                dist_sync_on_step=True,
                num_classes=self.num_classes
            ),
            'f1_micro': F1(
                compute_on_step=False,
                dist_sync_on_step=True,
                num_classes=self.num_classes,
                average='micro'
            ),
            'f1_macro': F1(
                compute_on_step=False,
                dist_sync_on_step=True,
                num_classes=self.num_classes,
                average='macro'
            )
        }
        if self.num_classes > 5:
            metrics['top3'] = Accuracy(
                compute_on_step=False,
                dist_sync_on_step=True,
                num_classes=self.num_classes,
                top_k=3
            )
        self.train_metrics = MetricCollection(metrics, prefix='train_')
        metrics['confusion_matrix'] = ConfusionMatrix(
            compute_on_step=False,
            dist_sync_on_step=True,
            num_classes=self.num_classes,
            normalize='true'
        )
        self.valid_metrics = MetricCollection(metrics, prefix='valid_')
        self.matrix = None

    def setup(self, stage):
        if stage == 'fit':
            self.train_steps = len(self.train_dataloader()) * self.args.max_epochs // self.args.accumulate_grad_batches

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        losses = self.criterion(logits, labels)
        self.log('train_loss', losses, on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.train_metrics(logits.softmax(-1), labels)
        return losses


    def training_epoch_end(self, outputs):
        # preds  = torch.cat([i['prediction'].view(-1, i['prediction'].shape[-1]) for i in outputs])
        # labels = torch.cat([i['target'].view(-1) for i in outputs])
        # self.train_metrics(preds, labels)
        scores = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_dict(scores)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        losses = self.criterion(logits, labels)
        self.log('valid_loss', losses, on_step=True, on_epoch=False, sync_dist=True, logger=True)
        self.valid_metrics(logits.softmax(-1), labels)

    def validation_epoch_end(self, outputs):
        scores = self.valid_metrics.compute()
        self.valid_metrics.reset()

        self.matrix = scores.pop('valid_confusion_matrix')
        self.log_dict(scores)

    @staticmethod
    def plot_confusion_matrix(matrix: torch.Tensor, num_classes: int):
        label_dict = {'B': 0, 'M': 1}
        if num_classes == 8:
            label_dict = {k: v for k, v in LABELS.items()}
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.rcParams['figure.dpi'] = 80

        fig, ax = plt.subplots()
        im = ax.imshow(matrix)
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("proportion", rotation=-90, va='bottom')

        keys = list(label_dict.keys())
        ax.set_xticks(np.arange(len(label_dict)))
        ax.set_yticks(np.arange(len(label_dict)))
        ax.set_xticklabels(keys)
        ax.set_yticklabels(keys)
        ax.set_xlabel('Predictions')
        ax.set_ylabel('True Labels')

        for i in range(len(keys)):
            for j in range(len(keys)):
                txt = ax.text(j, i, round(matrix[i,j], 3), ha='center', va='center', color='w')
        ax.set_title("Confusion Matrix")
        return fig

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
        optimizer = torch.optim.AdamW(
            grouped_params, lr=self.args.learning_rate,
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
        return {'optimizer': optimizer, 'lr_scheduler': scheduler} 



class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, args: "Namespace"):
        super().__init__()

        self.args = args

        dataset = BreakHisDataset(
            num_classes=args.num_classes,
            magnification=args.magnification,
            data_path=args.data_dir
        )
        train_len = int(len(dataset) * args.train_split)
        valid_len = len(dataset) - train_len
        self.train_ds, self.valid_ds = random_split(dataset, [train_len, valid_len])
        
        # Transforms
        base_transform = [
            alb.Resize(
                height=args.image_size[0],
                width=args.image_size[1]
            ),
            alb.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
        train_transform = list()
        valid_transform = list()
        if args.augmentation:
            train_transform += [
                alb.RandomScale(),
                alb.Rotate(),
                alb.Flip()
            ]
        train_transform += base_transform
        valid_transform += base_transform

        self.train_cl = ClassificationBatchCollector(
            image_size=args.image_size,
            patch_size=args.patch_size,
            transforms=alb.Compose(train_transform),
            pad_mode=args.pad_mode,
            resize=args.resize,
        )
        self.valid_cl = ClassificationBatchCollector(
            image_size=args.image_size,
            patch_size=args.patch_size,
            transforms=alb.Compose(valid_transform),
            pad_mode=args.pad_mode,
            resize=args.resize
        )

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
    config = yaml.load(open('config_cls.yml').read(), Loader=yaml.Loader)

    args = parser.add_arguments(ClassificationArgs(config=config), dest='options')
    args = parser.parse_args().options

    assert args.magnification in ('40X', '100X', '200X', '400X', 'all')
    assert args.pad_mode in ('constant', 'reflect', 'reflect_101', 'replicate')

    # If resize, resize to given image input size
    input_size = args.image_size
    # Otherwise, pad the image to the smallest tiling
    # Compute the new image size (will be padded)
    height_div = ceil(args.image_size[0] / args.patch_size[0])
    width_div = ceil(args.image_size[1] / args.patch_size[1])
    input_size = (height_div * args.patch_size[0], width_div * args.patch_size[1])
    args.image_size = input_size

    # Seeding and PL model
    pl.seed_everything(args.seed, workers=True)
    pl_model = ClassificationModel(args)

    # WandB logger
    run_name = 'cls-{}-{}-{}-{}-{}{}-{}'.format(
        args.model_name,
        'pretrained' if args.pretrained else 'scratch',
        'resized' if args.resize else 'as-is',
        args.magnification,
        'msb' if args.num_classes < 3 else 'msm',
        '-aug' if args.augmentation else '',
        args.pad_mode
    )

    if args.wandb:
        logger = WandbLogger(project='medical-transformers', name=run_name)
    else:
        logger = TensorBoardLogger(save_dir='logs/{}.log'.format(run_name))
    logger.log_hyperparams(vars(args))

    # Checkpointer
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_top1',
        dirpath='checkpoints',
        every_n_train_steps=0,
        every_n_val_epochs=1,
        filename='{epoch:02d}-{valid_top1:.2f}',
        save_top_k=1,
        mode='max'
    )

    pl_data = ClassificationDataModule(args)
    
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
