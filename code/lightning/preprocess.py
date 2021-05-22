import os
import glob

import cv2
import torch
import numpy as np

from functools import lru_cache
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

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

        # Function for getting the label
        if num_classes == 8:
            self.get_label = lambda x: LABELS.get(x.split('_')[2].split('-')[0])
        else:
            self.get_label = lambda x: BIN_LABELS.get(x.split('_')[2].split('-')[0])

    @lru_cache(maxsize=None)
    def io_cache(self, f):
        """Reads images from disk with LRU caching"""
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img / 255.0, dtype=np.float32)
        lbl = self.get_label(f)
        return img, lbl
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img, lbl = self.io_cache(f)
        return img, lbl

class ClassificationBatchCollector:
    """Batch collector for classification using in DataLoader.collate_fn"""
    def __init__(
        self,
        patch_size  : Tuple[int, int],
        image_mean  : List[float]=[0.5, 0.5, 0.5],
        image_std   : List[float]=[0.5, 0.5, 0.5],
        pad_mode    : Optional[str]="constant",
        do_normalize: Optional[bool]=True,
        do_augment  : Optional[bool]=False,
        augmentation: Optional[List["Transform"]]=None
    ):
        self.p_h, self.p_w = patch_size
        self.image_mean = np.array(image_mean)
        self.image_std  = np.array(image_std)

        self.pad_mode = getattr(cv2, 'BORDER_{}'.format(pad_mode.upper()))
        assert self.pad_mode in (0, 1, 2, 3, 4)
        self.do_normalize = do_normalize
        self.do_augment = do_augment
        self.augmentation = augmentation

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = [self.padding(image) for image in images]
        assert len(set([i.shape for i in images])) == 1
        if self.do_normalize:
            images = [self.normalize(image=image)  for image in images]
        if self.do_augment:
            images = [self.augmentation(image=image)['image'] for image in images]
        images = [image.transpose(2, 0, 1) for image in images]

        return torch.tensor(images), torch.tensor(labels)
        
    def padding(self, image: "np.ndarray") -> "np.ndarray":
        """Pads image with predefined border type"""
        h, w, _ = image.shape
        h_to_pad = self.p_h - (h % self.p_h)
        w_to_pad = self.p_w - (w % self.p_w)

        top, lft = h_to_pad // 2, w_to_pad // 2
        bot, rht = h_to_pad - top, w_to_pad - lft

        padded = cv2.copyMakeBorder(
            image, 
            top, bot, lft, rht,
            self.pad_mode
        )
        return padded

    def normalize(self, image: 'np.ndarray') -> 'np.ndarray':
        mean = self.image_mean.astype(image.dtype)
        std  = self.image_std.astype(image.dtype)
        if image.ndim == 3 and image.shape[2] in [1,3]:
            return (image - mean[None, None, :]) / std[None, None, :]
        else:
            return (image - mean) / std