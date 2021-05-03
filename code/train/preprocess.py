import os
import glob
import random

import cv2
import torch
import numpy as np
import albumentations as alb

from tqdm import tqdm
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
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        img_mean: List[float]=[0.5, 0.5, 0.5],
        img_std: List[float]=[0.5, 0.5, 0.5],
        pad_mode: str='CONSTANT',
        do_normalize: bool=True,
        do_augment: bool=False,
        transform: List["alb.Transform"]=None,
        **kwargs
    ):
        # Compute padding
        height, width = img_size
        self.p_height, self.p_width = patch_size
        self.border_type = getattr(cv2, 'BORDER_{}'.format(pad_mode))

        self.img_mean = img_mean
        self.img_std = img_std

        self.pad_mode = pad_mode
        self.do_normalize = do_normalize
        self.do_augment = do_augment

        if transform is not None:
            self.transform = alb.Compose(transform)

    def __call__(
        self, 
        images: Union[List["np.ndarray"], Tuple["np.ndarray"]]
    ) -> "torch.Tensor":
        images = [self.pad_image(image) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.img_mean, std=self.img_std) for image in images]
        if self.do_augment:
            images = [self.transform(image=image)['image'] for image in images]
        images = [image.transpose(2, 0, 1) for image in images]
        return torch.tensor(images)

    def pad_image(self, image):
        h, w, _ = image.shape
        h_to_pad = self.p_height - (h % self.p_height)
        w_to_pad = self.p_width - (w % self.p_width)

        top = h_to_pad // 2
        bottom = h_to_pad - top
        left = w_to_pad // 2
        right = w_to_pad - left
        padded = cv2.copyMakeBorder(
            image, 
            top, bottom, left, right,
            self.border_type
        )
        return padded
    
    # def to_image(
    #     self,
    #     image: Union["np.ndarray", "torch.Tensor"],
    #     rescale: Optional[bool]=True
    # ) -> "Image.Image":
    #     """
    #         Converts `np.ndarray`, `torch.Tensor` to PIL Image for 
    #         visualization.
    #     """
    #     assert isinstance(image, torch.Tensor) or isinstance(image, np.ndarray)

    #     if isinstance(image, torch.Tensor):
    #         image = image.numpy()

    #     if isinstance(image, np.ndarray):
    #         if image.ndim == 3 and image.shape[0] in [1,3]:
    #             image = image.transpose(1, 2, 0)
    #         if rescale:
    #             image = image * 255
    #         image = image.astype(np.uint8)
    #         return Image.fromarray(image)
    #     return image

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