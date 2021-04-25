import os
import sys
import yaml
import random

import torch
import wandb
import timm
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from tqdm import tqdm

from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Union, Optional
from simple_parsing import ArgumentParser

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from accelerate import Accelerator
from transformers import get_scheduler

from preprocess import FeatureExtractor, BatchCollector, BreakHisDataset, LABELS, BIN_LABELS

@dataclass
class Arguments:
    data_dir: str = field(default='combined')
    magnification: str = field(default='40X')
    augmentation: bool = field(default=False)
    train_split: float = field(default=0.7)

    model_class: str = field(default='vit_base_pytorch16_224')
    pretrained: bool = field(default=False)
    checkpoint: Optional[str] = field(default=None)
    num_classes: int = field(default=8)
    seed: int = field(default=42)

    device: str = field(default='cuda')
    amp: bool = field(default=False)

    num_epochs: int = field(default=3)
    batch_size: int = field(default=1)

    lr: float = field(default=3e-5)
    eps: float = field(default=1e-8)
    beta_1: float = field(default=0.99)
    beta_2: float = field(default=0.999)
    weight_decay: float = field(default=1e-5)
    grad_clipping: float = field(default=False)
    grad_max_norm: float = field(default=1.0)
    grad_accumulation_steps: int = field(default=1)

    scheduler: Optional[str] = field(default=None)
    num_warmup_steps: Optional[int] = field(default=400)

    log_interval: int = field(default=15)
    save_model: bool = field(default=False)

    config: InitVar[Dict] = field(default=dict())

    def __post_init__(self, config: Dict):
        for k, v in config.items():
            try:
                if v is not None:
                    self.__setattr__(k, v)
                # else:
                #     self.__setattr__(k, '')
            except AttributeError:
                raise Exception('No config parameter {} found ...'.format(k))

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def plot_confusion_matrix(logits: "torch.Tensor", labels: "torch.Tensor", num_classes: int):
    label_dict = {'B': 0, 'M': 1}
    if num_classes == 8:
        label_dict = {k: v for k,v in LABELS.items()}

    logits = logits
    preds = torch.argmax(logits, dim=-1).tolist()
    labels = labels.squeeze(-1).tolist()

    matrix = np.zeros((num_classes, num_classes))
    for lbl, pre in zip(labels, preds):
        matrix[lbl, pre] += 1
    matrix /= matrix.sum(axis=1)[:,None]

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
                text = ax.text(j, i, round(matrix[i,j], 3), ha='center', va='center', color='w')

    ax.set_title('Confusion Matrix')
    return fig, ax

def compute_metric(logits: "torch.Tensor", labels: "torch.Tensor", num_classes: int):
    """Computes validation metrics according to number of labels"""
    assert len(logits) == len(labels)
    tot = len(logits)
    if num_classes > 5:
        top3 = torch.topk(logits, k=3, dim=-1)[1]
        top1 = top3[:,:1]

        top3 = torch.sum(top3 == labels).cpu().item()
        top1 = torch.sum(top1 == labels).cpu().item()
        return {'top@3_acc': top3 / tot, 'top@1_acc': top1 / tot}
    else:
        top1 = torch.topk(logits, k=1, dim=-1)[1]
        top1 = torch.sum(top1 == labels).cpu().item()
        return {'top@1_acc': top1 / tot}

def train(
    model: "nn.Module",
    optimizer: "nn.optim.Optimizer",
    scheduler: "nn.optim.lr_scheduler.LRLambda",
    accelerator: "accelerate.Accelerator",
    train_loader: "DataLoader",
    valid_loader: "DataLoader",
    args: "Argument",
    total_steps: int
):
    device = accelerator.device

    # Progress bar
    progress = tqdm(total=total_steps, ascii=True)
    valid_loader = tqdm(valid_loader, ascii=True, leave=False)
    # Separate criterion
    train_criterion = nn.CrossEntropyLoss()
    valid_criterion = nn.CrossEntropyLoss()

    max_acc = -1
    step = 0

    model.to(device)
    
    train_losses = list()
    valid_losses = list()
    for epoch in range(args.num_epochs):
        # Train
        tot, mean_train_loss = 0, 0
        model.train()
        for batch in train_loader:
            current_progress = step / len(train_loader)
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            # Loss computation 
            logits = model(images)
            losses = train_criterion(logits, labels)

            # Loss logging
            item_loss = losses.detach().item()
            tot += len(labels)
            mean_train_loss += (item_loss * len(images))

            losses /= args.grad_accumulation_steps
            accelerator.backward(losses)

            # Gradient clipping (not really used)
            if args.grad_clipping:
                accelerator.clip_grad_norm_(args.grad_max_norm)

            progress.set_description('Epoch {:.3f} Loss: {:.5f}'.format(current_progress, item_loss))
            if not (step+1) % args.log_interval:
                wandb.log({'run_loss': item_loss}, step=step, commit=False)

            # Gradient accumulation for memory saving
            if not (step+1) % args.grad_accumulation_steps:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                # Update progress bar
                progress.update(1)
            step += 1
        train_losses.append(mean_train_loss / tot)
        
        # Validation
        mean_valid_loss, valid_logits, valid_labels = validate(model, valid_loader, valid_criterion, accelerator, step, current_progress, args)
        metric = compute_metric(valid_logits, valid_labels, args.num_classes)
        valid_losses.append(mean_valid_loss)

        # Log Validation Metric
        wandb.log(metric, step=step, commit=False)

        acc = metric['top@1_acc']
        if acc > max_acc:
            max_acc = acc
            
            if args.save_model:
                # Saving model
                save_path = '{}_{}_{}_c{}{}_{}.pt'.format(
                    args.model_class,
                    'pretrained' if args.pretrained else 'scratch',
                    args.magnification,
                    args.num_classes,
                    '_aug' if args.augmentation else '',
                    step
                )
                print('Best model found ... Saving at {} ...'.format(save_path))
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), save_path)

    # Log End Summary Metric
    matrix, _ = plot_confusion_matrix(valid_logits, valid_labels, args.num_classes)
    wandb.log({
        'confusion_matrix': matrix,
        'loss_curves': wandb.plot.line_series(
            xs=[_ for _ in range(len(train_losses))],
            ys=[train_losses, valid_losses],
            keys=['train', 'validation'],
            title='Loss Curves for Epoch',
            xname='epochs'
        )
    })
    wandb.run.summary['best_acc'] = max_acc

@torch.no_grad()
def validate(
    model: "nn.Module",
    loader: "DataLoader",
    criterion: "nn.optim.Optimizer",
    accelerator: "acceleate.Accelerator",
    global_step: int,
    current_progress: float,
    args
):
    device = accelerator.device
    all_logits, all_labels = list(), list()
    
    tot, mean_valid_loss = 0, 0
    model.eval()
    for batch in loader:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(images)

        losses = criterion(logits, labels)
        item_loss = losses.detach().item()
        tot += len(labels)
        mean_valid_loss += (item_loss * len(labels))
        loader.set_description('Validation Loss: {:.5f}'.format(item_loss))

        all_logits.append(accelerator.gather(logits))
        all_labels.append(accelerator.gather(labels.unsqueeze(-1)))
    
    all_logits, all_labels = torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
    return mean_valid_loss / tot, all_logits, all_labels

def main():
    parser = ArgumentParser()
    config = yaml.load(open('config.yml').read(), Loader=yaml.Loader)

    args = parser.add_arguments(Arguments(config=config), dest='options')
    args = parser.parse_args().options

    assert args.magnification in ('40X', '100X', '200X', '400X', 'all')

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Accelerator (syntatic sugar for distributed / fp16 training)
    accelerator = Accelerator(fp16=args.amp, cpu=True if args.device=='cpu' else False)

    # Dataset
    if args.magnification == 'all':
        list_data = [BreakHisDataset(args.num_classes,
                                     args.data_dir,
                                     magnification=mag) for mag in ('100X', '200X', '400X')]
        data = BreakHisDataset(args.num_classes, args.data_dir, magnification='40X')
        for i in list_data:
            data = data + i
    else:
        data = BreakHisDataset(args.num_classes, args.data_dir, args.magnification)

    train_ln = int(len(data) * args.train_split)
    valid_ln = len(data) - train_ln
    train_ds, valid_ds = random_split(data, [train_ln, valid_ln])

    tr_ext, ev_ext = FeatureExtractor(augment=args.augmentation), FeatureExtractor(augment=False)
    tr_collector, ev_collector = BatchCollector(tr_ext), BatchCollector(ev_ext)
    train_loader = DataLoader(train_ds, collate_fn=tr_collector, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_ds, collate_fn=ev_collector, batch_size=args.batch_size)

    # Model 
    model = timm.create_model(args.model_class, pretrained=args.pretrained, num_classes=args.num_classes)

    # Optimizer
    no_decay  = ['bias', 'LayerNorm.weight']
    grouped_params = [
        {
            'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=args.weight_decay)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)

    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint): 
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(args.checkpoint))

    # Scheduler
    update_steps_per_epoch = max(1, len(train_loader) // args.grad_accumulation_steps)
    total_steps = ceil(update_steps_per_epoch * args.num_epochs)
    scheduler = None
    if args.scheduler is not None:
        scheduler = get_scheduler(args.scheduler, optimizer, args.num_warmup_steps, total_steps)

    # Logger: wandb
    print('Total Update Steps: {}'.format(total_steps))
    rank = os.environ.get('RANK', -1)
    if rank == 0 or rank == -1:
        wandb.init(project='vit-domain-adapt', config=args)

        run_name = '{}_{}_{}_c{}{}'.format(
                    args.model_class,
                    'pretrained' if args.pretrained else 'scratch',
                    args.magnification,
                    args.num_classes,
                    '_aug' if args.augmentation else ''
                )
        wandb.run.name = run_name

    # Train
    train(model, optimizer, scheduler, accelerator, train_loader, valid_loader, args, total_steps)

if __name__ == '__main__':
    main()