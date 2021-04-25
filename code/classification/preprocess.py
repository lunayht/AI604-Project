import os
import glob
import random

import torch
import albumentations as alb
import numpy as np

from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union

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

LABELS = {k[0]: v for v, k in enumerate(CLASSES)}

# 0 if BENIGN, 1 if MALIGNANT
BIN_LABELS = {k[0]: 1 if k[0] in ('DC', 'LC', 'MC', 'PC') else 0 for v, k in enumerate(CLASSES)}


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

class BreakHisDataset(Dataset):
    def __init__(self, num_classes: int, data_path: str='./combined', magnification: str='40X', files: Optional[List]=None):
        super().__init__()

        self.magnification = magnification
        self.num_classes = num_classes

        if files is not None:
            self.files = files
        else:
            paths = os.path.join(data_path, magnification, '*.png')
            self.files = glob.glob(paths)
        
        # Function for getting the label
        if num_classes == 8:
            self.get_label = lambda x: LABELS.get(x.split('_')[2].split('-')[0])
        else:
            self.get_label = lambda x: BIN_LABELS.get(x.split('_')[2].split('-')[0])
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        # np.ndarray
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # int
        lbl = self.get_label(f)
        
        return \
            {
                'images': np.array(img / 255.0, dtype=np.float32),
                'labels': lbl
            }

    def __add__(self, other):
        assert self.num_classes == other.num_classes
        self.files.extend(other.files)
        return self

class BatchCollector:
    """
        Batch collector for using in DataLoader.collate_fn.
    """
    def __init__(self, extractor: "FeatureExtractor"):
        self.extractor = extractor

    def __call__(self, batch):
        images = [_['images'] for _ in batch]
        labels = [_['labels'] for _ in batch]

        return \
            {
                'images': self.extractor(images),
                'labels': torch.tensor(labels)
            }

class FeatureExtractor:
    """
        Feature Extractor class for preparing image features.

        Modified the implementation of FeatureExtractor and
        ImageFeatureExtractionMixin by the HuggingFace Team.

        https://github.com/huggingface/transformers/tree/master/src/transformers
    """
    def __init__(
        self,
        img_mean: List[float]=None,
        img_std: List[float]=None,
        normalize: bool=True,
        augment: bool=False,
        resize: bool=True,
        size=224,
        transform=None,
        **kwargs
    ):
        self.img_mean = [0.5, 0.5, 0.5]
        self.img_std  = [0.5, 0.5, 0.5]
        self.do_normalize = normalize
        self.size = size
        self.transform = alb.Resize(224,224)
        if transform is not None:
            self.transform = transform

    def __call__(
        self, 
        images: Union[List["Image.Image"], Tuple["Image.Image"]]
    ) -> "torch.Tensor":
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.img_mean, std=self.img_std) for image in images]
        images = [self.transform(image=image)['image'] for image in images]
        images = [image.transpose(2, 0, 1) for image in images]
        return torch.tensor(images)
    
    def to_image(
        self,
        image: Union["np.ndarray", "torch.Tensor"],
        rescale: Optional[bool]=True
    ) -> "Image.Image":
        """
            Converts `np.ndarray`, `torch.Tensor` to PIL Image for 
            visualization.
        """
        assert isinstance(image, torch.Tensor) or isinstance(image, np.ndarray)

        if isinstance(image, torch.Tensor):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in [1,3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return Image.fromarray(image)
        return image

    def normalize(
        self,
        image: "np.ndarray",
        mean: Union[List[float], "np.ndarray", "torch.Tensor"],
        std: Union[List[float], "np.ndarray", "torch.Tensor"]
    ) -> "np.ndarray":
        assert isinstance(image, np.ndarray)

        if not isinstance(mean, np.ndarray):
            mean = np.array(mean).astype(image.dtype)
        if not isinstance(std, np.ndarray):
            std = np.array(std).astype(image.dtype)
        elif isinstance(image, torch.Tensor):
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std)
        if image.ndim == 3 and image.shape[2] in [1,3]:
            return (image - mean[None, None, :]) / std[None, None, :]
        else:
            return (image - mean) / std