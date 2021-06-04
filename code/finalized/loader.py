import os
import glob

import cv2
import torch
import numpy as np
import albumentations as alb

from functools import lru_cache
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

cv2.setNumThreads(0)

"""
    NOTE: The classes can be put into sub-classes.

    MALIGNANT = \
        [
            ('DC', 'ductal_carcinoma'),
            ('LC', 'lobular_carcinoma'),
            ('MC', 'mucinous_carcinoma'),
            ('PC', 'papillary_carcinoma')
        ]
    BENIGN = \
        [
            ('A', 'adenosis'),
            ('F', 'fibroadenoma'),
            ('PT', 'phyllodes_tumor'),
            ('TA', 'tubular_adenoma')
        ]
"""

CLASSES = \
    [
        ('DC', 'ductal_carcinoma'),
        ('LC', 'lobular_carcinoma'),
        ('MC', 'mucinous_carcinoma'),
        ('PC', 'papillary_carcinoma'),
        ('A', 'adenosis'),
        ('F', 'fibroadenoma'),
        ('PT', 'phyllodes_tumor'),
        ('TA', 'tubular_adenoma')
    ]

# Label map
LABELS = {k[0]: v for v, k in enumerate(CLASSES)}
# 0 if BENIGN, 1 if MALIGNANT
BIN_LABELS = {k[0]: 1 if k[0] in ('DC', 'LC', 'MC', 'PC') else 0 for v, k in enumerate(CLASSES)}

class BreakHisDataset(Dataset):
    def __init__(
        self,
        num_classes: int,
        magnification: str,
        data_path: Optional[str]='./data/breakhis',
    ):
        super().__init__()

        self.num_classes = num_classes
        self.magnification = magnification

        # Reads file names (will be read in at io_cache)
        self.files = list()
        if self.magnification == 'all':
            for i in ('40X', '100X', '200X', '400X'):
                paths = os.path.join(data_path, i, '*.png')
                self.files += glob.glob(paths)
        else:
            paths = os.path.join(data_path, self.magnification, '*.png')
            self.files += glob.glob(paths)

        assert len(self.files) > 0, "No image files found."

        # Function for getting the label
        if num_classes == 8:
            self.get_label = lambda x: LABELS.get(x.split('_')[2].split('-')[0])
        else:
            self.get_label = lambda x: BIN_LABELS.get(x.split('_')[2].split('-')[0])
       
    def __len__(self):
        return len(self.files)

    @lru_cache(maxsize=None)
    def cache(self, file):
        """Disk read/write with lru caching for reducing NAS communication overhead"""
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = self.get_label(file)
        return img, lbl

    def __getitem__(self, idx):
        f = self.files[idx]
        return self.cache(f)

class ClassificationBatchCollector:
    """Batch collector for classification using in DataLoader.collate_fn"""
    def __init__(
        self,
        image_size : Tuple[int, int], # This is input size for model
        patch_size : Tuple[int, int],
        transforms : "alb.Transform",
        pad_mode : Optional[str]="constant",
        resize   : Optional[bool]=False
    ):
        self.i_h, self.i_w = image_size
        self.p_h, self.p_w = patch_size
        self.transforms = transforms
        self.pad_mode = getattr(cv2, 'BORDER_{}'.format(pad_mode.upper()))
        self.resize = resize

        assert self.pad_mode in (0, 1, 2, 3, 4)
        
    def __call__(self, batch):
        images, labels = zip(*batch)
        if not self.resize:
            images = [self.padding(image) for image in images]
        images = [self.transforms(image=image)['image'] for image in images]
        images = [image.transpose(2, 0, 1) for image in images]
        return torch.tensor(images), torch.tensor(labels)
        # return torch.tensor(images)

    def padding(self, image: "np.ndarray") -> "np.ndarray":
        """Pads image with predefined border type"""
        h, w, _ = image.shape
        h_to_pad = self.i_h - h
        w_to_pad = self.i_w - w

        top, lft = h_to_pad // 2, w_to_pad // 2
        bot, rht = h_to_pad - top, w_to_pad - lft

        padded = cv2.copyMakeBorder(
            image, 
            top, bot, lft, rht,
            self.pad_mode
        )
        return padded

class CrowdsourcingDataset(Dataset):
    def __init__(self, data_path: Optional[str]='dataset/crowdsourcing/patch-512-reflect'):
        super().__init__()

        img_path = os.path.join(data_path, 'images', '*.png')
        msk_path = os.path.join(data_path, 'masks', '*.png')
        self.image_files = glob.glob(img_path)
        self.mask_files  = glob.glob(msk_path)

        assert len(self.image_files) > 0, "No image files found."
        assert all([
            os.path.basename(i) == os.path.basename(j) \
                for i,j in zip(self.image_files, self.mask_files)
        ]), "Image and mask files do not match."

    @lru_cache(maxsize=None)
    def cache(self, img, msk):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
        return img, msk
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img, msk = self.image_files[idx], self.mask_files[idx]
        return self.cache(img, msk)

class SegmentationBatchCollector:
    def __init__(self, augmentation: "alb.Transform"=None, num_classes: int=22, pad_id: int=100):
        self.augmentation = augmentation
        self.normalize = alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        self.num_classes = num_classes
        self.pad_id = pad_id

    def __call__(self, batch):
        images, masks = zip(*batch)
        # Same augmentation done on image and mask
        if self.augmentation is not None:
            augmented = [self.augmentation(image=image, mask=mask) for image, mask in batch]
            images = [aug['image'] for aug in augmented]
            masks  = [aug['mask'] for aug in augmented]
        images = [self.normalize(image=image)['image'] for image in images]
        images = [image.transpose(2, 0, 1) for image in images]
        # images = [cv2.resize(image, (512, 512)) for image in images]
        # masks  = [cv2.resize(mask, (512, 512)) for mask in masks]
        images, masks = torch.tensor(images), torch.tensor(masks)
        return images, masks.long()
