import cv2
import yaml
import timm
import albumentations as alb
import pytorch_lightning as pl

from math import ceil
from torch.utils.data import DataLoader, random_split

from simple_parsing import ArgumentParser
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from arguments import ClassificationArgs
from modules import ClassificationModule, ClassificationModel
from preprocess import ClassificationBatchCollector, BreakHisDataset, LABELS, BIN_LABELS

if __name__ == '__main__':
    # Argument parsing
    parser = ArgumentParser()
    config = yaml.load(open('config.yml').read(), Loader=yaml.Loader)

    args = parser.add_arguments(ClassificationArgs(config=config), dest='options')
    args = parser.parse_args().options

    if args.classification:
        assert args.magnification in ('40X', '100X', '200X', '400X', 'all')
        assert args.pad_mode in ('constant', 'reflect', 'reflect_101', 'replicate')

        # If resize, resize to given image input size
        input_size = args.image_size
        # Otherwise, pad the image to the smallest tiling
        if not args.resize:
            # Compute the new image size (will be padded)
            height_div = ceil(args.image_size[0] / args.patch_size[0])
            width_div = ceil(args.image_size[1] / args.patch_size[1])
            input_size = (height_div * args.patch_size[0], width_div * args.patch_size[1])
        
        # Model
        encoder = timm.create_model(
            args.model_name,
            pretrained=args.pretrained,
            num_classes=0,
            drop_rate=args.drop_rate,
            img_size=input_size,
            patch_size=args.patch_size
        )
        model = ClassificationModule(encoder, args.num_classes_classification)

        # Seeding and PL model
        pl.seed_everything(args.seed, workers=True)
        pl_model = ClassificationModel(args, model)

        # WandB logger
        run_name = '{}-{}-{}-{}-c{}{}-{}'.format(
            args.model_name,
            'pretrained' if args.pretrained else 'scratch',
            'resize' if args.resize else 'padded',
            args.magnification,
            args.num_classes,
            '-aug' if args.augmentation else '',
            args.pad_mode
        )

        if args.wandb:
            logger = WandbLogger(project='medical-transformers', name=run_name)
        else:
            logger = TensorBoardLogger(save_dir='logs/{}.log'.format(run_name))
        logger.log_hyperparams(args)

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

        # Data
        ds = BreakHisDataset(
            args.num_classes_classification,
            magnification=args.magnification,
            data_path=args.data_dir
        )
        
        # Transforms
        train_transform = list()
        valid_transform = [
            alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            alb.Resize(height=input_size[0], width=input_size[1])
        ]
        if args.augmentation:
            train_transform += [alb.ShiftScaleRotate(border_mode=cv2.BORDER_REFLECT)]
        train_transform += [
            alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
            alb.Resize(height=input_size[0], width=input_size[1])
        ]
        
        train_len = int(len(ds) * args.train_split)
        valid_len = len(ds) - train_len
        train_ds, valid_ds = random_split(ds, [train_len, valid_len])
        
        train_cl = ClassificationBatchCollector(
            image_size=input_size,
            patch_size=args.patch_size,
            pad_mode=args.pad_mode,
            do_resize=args.resize,
            do_augment=args.augmentation,
            transforms=alb.Compose(train_transform)
        )
        valid_cl = ClassificationBatchCollector(
            image_size=input_size,
            patch_size=args.patch_size,
            pad_mode=args.pad_mode,
            do_resize=args.resize,
            do_augment=False,
            transforms=alb.Compose(valid_transform)
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            collate_fn=train_cl,
            pin_memory=True,
            shuffle=True
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            collate_fn=valid_cl,
            pin_memory=True,
            shuffle=False
        )
        
        # Trainer
        trainer_args = {
            'accelerator': 'dp',
            'accumulate_grad_batches': args.accumulate_grad_batches,
            'amp_backend': 'native',
            'auto_lr_find': True,
            'callbacks': [checkpoint_callback],
            'gradient_clip_val': args.grad_max_norm if args.grad_clipping else 0.0,
            'gpus': args.gpus, 
            'logger': logger,
            'log_every_n_steps': args.log_interval,
            'max_epochs': args.max_epochs,
            'accelerator': 'ddp',
            'plugins': DDPPlugin(find_unused_parameters=False),
            'precision': 16 if args.amp else 32,
            'val_check_interval': args.val_check_interval,
            'deterministic': True
        }
        trainer  = pl.Trainer(**trainer_args)

        # trainer.tune(pl_model, train_loader, valid_loader)
        trainer.fit(pl_model, train_loader, valid_loader)
    
    if args.segmentation:
        pass
