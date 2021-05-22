import torch
import torchmetrics
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

class ClassificationModule(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x):
        x = self.encoder.encode(x)[:,0]
        return self.head(x)

class SegmentationModule(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)
    
    def forward(self, x):
        if self.encoder.distil_token:
            x = self.encoder.encode(x)[:,1:-1]
        else:
            x = self.encoder.encode(x)[:,1:]
        return self.head(x)

class ClassificationModel(pl.LightningModule):
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
        self.criterion = nn.CrossEntropyLoss()
        
        # Metric trackers
        train_metrics = \
            {
                'top1': torchmetrics.Accuracy(
                    compute_on_step=False,
                    num_classes=self.num_classes,
                ),
                'f1': torchmetrics.F1(
                    compute_on_step=False,
                    num_classes=self.num_classes
                )
            }
        valid_metrics = \
            {
                'top1': torchmetrics.Accuracy(
                    compute_on_step=False,
                    num_classes=self.num_classes
                ),
                'f1': torchmetrics.F1(
                    compute_on_step=False,
                    num_classes=self.num_classes
                )
            }
        if self.num_classes > 5:
            train_metrics['top3'] = torchmetrics.Accuracy(
                compute_on_step=False,
                num_classes=self.num_classes,
                top_k=3
            )
            valid_metrics['top3'] = torchmetrics.Accuracy(
                compute_on_step=False,
                num_classes=self.num_classes,
                top_k=3
            )
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
        self.train_metrics['top1'].update(outputs['pred'], outputs['label'])
        self.train_metrics['f1'].update(outputs['pred'], outputs['label'])
        if self.num_classes > 5:
            self.train_metrics['top3'].update(outputs['pred'], outputs['label'])
        return outputs['loss']

    def training_epoch_end(self, *args, **kwargs):
        top1_acc = self.train_metrics['top1'].compute()
        f1 = self.train_metrics['f1'].compute()
        self.log('train_top1', top1_acc, on_epoch=True)
        self.log('train_f1', f1, on_epoch=True)
        self.train_metrics['top1'].reset()
        self.train_metrics['f1'].reset()
        if self.num_classes > 5:
            top3_acc = self.train_metrics['top3'].compute()
            self.log('train_top3', top3_acc, on_epoch=True) 
            self.valid_metrics['top3'].reset()           

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        valid_loss = self.criterion(logits, labels)
        logits = F.softmax(logits, dim=-1)
        return {'loss': valid_loss, 'pred': logits, 'label': labels}

    def validation_step_end(self, outputs):
        self.log('valid_loss', outputs['loss'], on_step=True, prog_bar=True, logger=True)
        self.valid_metrics['top1'].update(outputs['pred'], outputs['label'])
        self.valid_metrics['f1'].update(outputs['pred'], outputs['label'])
        if self.num_classes > 5:
            self.valid_metrics['top3'].update(outputs['pred'], outputs['label'])
        return outputs['loss']

    def validation_epoch_end(self, *args, **kwargs):
        top1_acc = self.valid_metrics['top1'].compute()
        f1 = self.valid_metrics['f1'].compute()
        self.log('valid_top1', top1_acc, on_epoch=True)
        self.log('valid_f1', f1, on_epoch=True)
        self.valid_metrics['top1'].reset()
        self.valid_metrics['f1'].reset()
        if self.num_classes > 5:
            top3_acc = self.valid_metrics['top3'].compute()
            self.log('valid_top3', top3_acc, on_epoch=True)     
            self.valid_metrics['top3'].reset()

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
        self.criterion = nn.CrossEntropyLoss()
        
        # Metric trackers
        train_metrics = \
            {
                'top1': torchmetrics.Accuracy(
                    compute_on_step=False,
                    num_classes=self.num_classes,
                )
            }
        valid_metrics = \
            {
                'top1': torchmetrics.Accuracy(
                    compute_on_step=False,
                    num_classes=self.num_classes
                )
            }
        if self.num_classes > 5:
            train_metrics['top3'] = torchmetrics.Accuracy(
                compute_on_step=False,
                num_classes=self.num_classes,
                top_k=3
            )
            valid_metrics['top3'] = torchmetrics.Accuracy(
                compute_on_step=False,
                num_classes=self.num_classes,
                top_k=3
            )
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
        self.train_metrics['top1'].update(outputs['pred'], outputs['label'])
        if self.num_classes > 5:
            self.train_metrics['top3'].update(outputs['pred'], outputs['label'])
        return outputs['loss']

    def training_epoch_end(self, *args, **kwargs):
        top1_acc = self.train_metrics['top1'].compute()
        self.log('train_top1', top1_acc, on_epoch=True)
        self.train_metrics['top1'].reset()
        if self.num_classes > 5:
            top3_acc = self.train_metrics['top3'].compute()
            self.log('train_top3', top3_acc, on_epoch=True) 
            self.valid_metrics['top3'].reset()           

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        valid_loss = self.criterion(logits, labels)
        logits = F.softmax(logits, dim=-1)
        return {'loss': valid_loss, 'pred': logits, 'label': labels}

    def validation_step_end(self, outputs):
        self.log('valid_loss', outputs['loss'], on_step=True, prog_bar=True, logger=True)
        self.valid_metrics['top1'].update(outputs['pred'], outputs['label'])
        if self.num_classes > 5:
            self.valid_metrics['top3'].update(outputs['pred'], outputs['label'])
        return outputs['loss']

    def validation_epoch_end(self, *args, **kwargs):
        top1_acc = self.valid_metrics['top1'].compute()
        self.log('valid_top1', top1_acc, on_epoch=True)
        self.valid_metrics['top1'].reset()
        if self.num_classes > 5:
            top3_acc = self.valid_metrics['top3'].compute()
            self.log('valid_top3', top3_acc, on_epoch=True)     
            self.valid_metrics['top3'].reset()

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
