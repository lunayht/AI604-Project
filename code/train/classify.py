import os
import sys
import yaml
import random

import timm
import torch
import wandb
import albumentations as alb

from math import ceil
from tqdm import tqdm

from typing import Dict, List, Union, Optional
from simple_parsing import ArgumentParser

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from accelerate import Accelerator
from transformers import get_scheduler

from trainer import ClassificationArgs
from utils import plot_confusion_matrix, compute_metric, set_seed
from preprocess import FeatureExtractor, BatchCollector, BreakHisDataset, LABELS, BIN_LABELS

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
    best_model = dict()

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

            losses /= args.gradient_accumulation_steps
            accelerator.backward(losses)

            # Gradient clipping (not really used)
            if args.gradient_clipping:
                accelerator.clip_grad_norm_(args.gradient_max_norm)

            progress.set_description('Epoch {:.3f} Loss: {:.5f}'.format(current_progress, item_loss))
            if not (step+1) % args.log_interval:
                wandb.log({'run_loss': item_loss}, step=step, commit=False)

            # Gradient accumulation for memory saving
            if not (step+1) % args.gradient_accumulation_steps:
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

        acc = metric['acc@1']
        if acc > max_acc:
            max_acc = acc
            
            if args.save_model:
                save_path = '{}_{}_{}_c{}{}_{}_{}.pt'.format(
                    args.model_class,
                    'pretrained' if args.pretrained else 'scratch',
                    args.magnification,
                    args.num_classes,
                    '_aug' if args.augmentation else '',
                    args.pad_mode,
                    step
                )
                print('Best model found ... Saving at {} ...'.format(save_path))
                unwrapped_model = accelerator.unwrap_model(model)
                if not args.save_only_best:
                    accelerator.save(unwrapped_model.state_dict(), save_path)
                else:
                    best_model['path'] = save_path
                    best_model['model'] = unwrapped_model.state_dict()

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

    # Save Model
    if args.save_only_best:
        accelerator.save(best_model['model'], best_model['path'])

@torch.no_grad()
def validate(
    model: "nn.Module",
    loader: "DataLoader",
    criterion: "nn.optim.Optimizer",
    accelerator: "accelerate.Accelerator",
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

    args = parser.add_arguments(ClassificationArgs(config=config), dest='options')
    args = parser.parse_args().options

    assert args.magnification in ('40X', '100X', '200X', '400X', 'all')
    assert args.pad_mode in ('constant', 'reflect', 'reflect_101', 'replicate')

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Accelerator (syntactic sugar for distributed / fp16 training)
    accelerator = Accelerator(fp16=args.amp, cpu=True if args.device=='cpu' else False)

    # Compute input size based on image size and patch size
    height_div = args.image_size[0] // args.patch_size[0]
    if args.image_size[0] % args.patch_size[0]:
        height_div += 1
    width_div = args.image_size[1] // args.patch_size[1]
    if args.image_size[1] % args.patch_size[1]:
        width_div += 1
    input_size = (height_div * args.patch_size[0], width_div * args.patch_size[1])

    # Model
    model = timm.create_model(
        args.model_class,
        pretrained=args.pretrained,
        img_size=input_size,
        num_classes=args.num_classes,
        attn_drop_rate=args.attn_drop_rate,
        drop_rate=args.drop_rate
    )

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
    
    transform = None
    if args.augmentation:
        transform = [
            alb.RandomScale(),
            alb.Rotate(),
            alb.Flip()
        ]

    train_ln = int(len(data) * args.train_split)
    valid_ln = len(data) - train_ln
    train_ds, valid_ds = random_split(data, [train_ln, valid_ln])

    tr_ext = FeatureExtractor(
        img_size=args.image_size,
        patch_size=args.patch_size,
        pad_mode=args.pad_mode.upper(),
        augment=args.augmentation,
        transform=transform
    )
    ev_ext = FeatureExtractor(
        img_size=args.image_size,
        patch_size=args.patch_size,
        pad_mode=args.pad_mode.upper(),
        augment=False
    )
    tr_collector = BatchCollector(tr_ext)
    ev_collector = BatchCollector(ev_ext)
    train_loader = DataLoader(train_ds, collate_fn=tr_collector, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_ds, collate_fn=ev_collector, batch_size=args.batch_size)

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
    update_steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
    total_steps = ceil(update_steps_per_epoch * args.num_epochs)
    scheduler = None
    if args.scheduler is not None:
        scheduler = get_scheduler(args.scheduler, optimizer, args.num_warmup_steps, total_steps)

    # Logger: wandb
    print('Total Update Steps: {}'.format(total_steps))
    rank = os.environ.get('RANK', -1)
    if rank == 0 or rank == -1:
        wandb.init(project='vit-domain-adapt', config=args)

        run_name = '{}_{}_{}_c{}{}_{}'.format(
                    args.model_class,
                    'pretrained' if args.pretrained else 'scratch',
                    args.magnification,
                    args.num_classes,
                    '_aug' if args.augmentation else '',
                    args.pad_mode
                )
        wandb.run.name = run_name

    # Train
    train(model, optimizer, scheduler, accelerator, train_loader, valid_loader, args, total_steps)

if __name__ == '__main__':
    main()