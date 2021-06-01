
import os
import sys
import yaml
import torch
import wandb
import timm
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt

from math import ceil
from tqdm import tqdm

from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Union, Optional
from simple_parsing import ArgumentParser
from torchmetrics import Accuracy, AverageMeter, F1, ConfusionMatrix, MetricCollection

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from accelerate import Accelerator
from transformers import get_scheduler

from arguments import ClassificationArgs
from loader import ClassificationBatchCollector, BreakHisDataset, LABELS, BIN_LABELS

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
    accelerator: "Accelerator",
    train_loader: "DataLoader",
    valid_loader: "DataLoader",
    args: "Namespace",
    total_steps: int
):  
    mean_train_loss = {'train_loss': AverageMeter(compute_on_step=False)}
    mean_valid_loss = {'valid_loss': AverageMeter(compute_on_step=False)}
    metrics=  {
        'top1': Accuracy(
            compute_on_step=False,
            # dist_sync_on_step=True,
            num_classes=args.num_classes
        ),
        'f1_micro': F1(
            compute_on_step=False,
            # dist_sync_on_step=True,
            num_classes=args.num_classes,
            average='micro'
        ),
        'f1_macro': F1(
            compute_on_step=False,
            # dist_sync_on_step=True,
            num_classes=args.num_classes,
            average='macro'
        )
    }
    if args.num_classes > 5:
        metrics['top3'] = Accuracy(
            compute_on_step=False,
            # dist_sync_on_step=True,
            num_classes=args.num_classes,
            top_k=3
        )
    train_metrics = MetricCollection(metrics, prefix='train_')
    metrics['confusion_matrix'] = ConfusionMatrix(
        compute_on_step=False,
        # dist_sync_on_step=True,
        num_classes=args.num_classes,
        normalize='true'
    )
    valid_metrics = MetricCollection(metrics, prefix='valid_')

    device = accelerator.device

    # Progress bar
    progress = tqdm(total=total_steps, ascii=True)
    valid_loader = tqdm(valid_loader, ascii=True, leave=False)
    
    # Separate criterion
    criterion = nn.CrossEntropyLoss()

    step = 0

    all_logits, all_labels = list(), list()

    model.to(device)
    for epoch in range(args.max_epochs):
        train_losses = list()
        valid_losses = list()
        # Train
        mean_train_loss = 0
        model.train()
        for images, labels in train_loader:
            current_progress = step / len(train_loader)
            images = images.to(device)
            labels = labels.to(device)

            # Loss computation 
            logits = model(images)
            losses = criterion(logits, labels)

            # Loss logging
            item_loss = losses.detach().item()
            mean_train_loss = (item_loss * len(images)) / len(train_loader)

            # Train metrics
            all_logits.append(accelerator.gather(logits))
            all_labels.append(accelerator.gather(labels.unsqueeze(-1)))

            losses /= args.accumulate_grad_batches
            accelerator.backward(losses)

            # Gradient clipping (not really used)
            if args.grad_clipping:
                accelerator.clip_grad_norm_(args.grad_max_norm)

            progress.set_description('Epoch {:.3f} Loss: {:.5f}'.format(current_progress, item_loss))
            if not (step+1) % args.log_interval:
                wandb.log({'run_loss': item_loss}, step=step, commit=False)

            # Gradient accumulation for memory saving
            if not (step+1) % args.accumulate_grad_batches:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # Update progress bar
                progress.update(1)
            step += 1
        train_losses.append(mean_train_loss)

        all_logits, all_labels = torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

        # Validation
        mean_valid_loss,  = validate(model, valid_loader, criterion, accelerator, valid_metrics)
        metric = compute_metric(valid_logits, valid_labels, args.num_classes)
        valid_losses.append(mean_valid_loss)
        wandb.log(metric, step=step, commit=False)

        acc = metric['top@1_acc']
        if acc > max_acc:
            max_acc = acc

            # if args.save_model:
            #     # Saving model
            #     save_path = '{}_{}_{}_c{}_{}.pt'.format(
            #         args.model_class,
            #         'pretrained' if args.pretrained else 'scratch',
            #         args.magnification,
            #         args.class_num,
            #         step
            #     )
            #     print('Best model found ... Saving at {} ...'.format(save_path))
            #     unwrapped_model = accelerator.unwrap_model(model)
            #     accelerator.save(unwrapped_model.state_dict(), save_path)

    
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

@torch.no_grad()
def validate(
    model: "nn.Module",
    loader: "DataLoader",
    criterion: "nn.Module",
    accelerator: "Accelerator",
    step, 
    current_progress,
    args
):
    device = accelerator.device
    all_logits, all_labels = list(), list()

    mean_valid_loss = 0
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        losses = criterion(logits, labels)
        item_loss = losses.detach().item()
        mean_valid_loss += (item_loss * len(labels)) / len(loader)
        loader.set_description('Validation Loss: {:.5f}'.format(item_loss))

        all_logits.append(accelerator.gather(logits))
        all_labels.append(accelerator.gather(labels.unsqueeze(-1)))

    all_logits, all_labels = torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
    return mean_valid_loss, all_logits, all_labels

def main():
    parser = ArgumentParser()
    config = yaml.load(open('config_cls.yml').read(), Loader=yaml.Loader)

    args = parser.add_arguments(ClassificationArgs(config=config), dest='options')
    args = parser.parse_args().options

    # Accelerator (syntatic sugar for distributed / fp16 training)
    accelerator = Accelerator(fp16=args.amp, cpu=False)

    # Dataset
    data = BreakHisDataset(args.num_classes, args.magnification, args.data_dir)

    train_ln = int(len(data) * args.train_split)
    valid_ln = len(data) - train_ln
    train_ds, valid_ds = random_split(data, [train_ln, valid_ln])

    base_transform = [
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
        alb.Resize(
            height=args.image_size[0],
            width=args.image_size[1]
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


    tr_collector = ClassificationBatchCollector(
        image_size=args.image_size,
        patch_size=args.patch_size,
        pad_mode='constant',
        resize=args.resize,
        transforms=alb.Compose(train_transform)
    )
    ev_collector = ClassificationBatchCollector(
        image_size=args.image_size,
        patch_size=args.patch_size,
        pad_mode='constant',
        resize=args.resize,
        transforms=alb.Compose(valid_transform)
    )
    train_loader = DataLoader(
        train_ds,
        collate_fn=tr_collector,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    valid_loader = DataLoader(
        valid_ds,
        collate_fn=ev_collector,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    # Model 
    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=args.num_classes)

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
    optimizer = torch.optim.AdamW(grouped_params, lr=args.learning_rate, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=args.weight_decay)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)

    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint): 
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(args.checkpoint))

    # Scheduler
    update_steps_per_epoch = max(1, len(train_loader) // args.accumulate_grad_batches)
    total_steps = ceil(update_steps_per_epoch * args.max_epochs)
    scheduler = get_scheduler('linear', optimizer, 400, total_steps)

    # Logger: wandb
    print('Total Update Steps: {}'.format(total_steps))
    rank = os.environ.get('RANK', -1)
    if rank == 0 or rank == -1:
        wandb.init(project='medical-transformers', config=args)
    # Train
    train(model, optimizer, scheduler, accelerator, train_loader, valid_loader, args, total_steps)

if __name__ == '__main__':
    main() 