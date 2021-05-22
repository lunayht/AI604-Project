"""
Load Crowdsourcing Dataset:
    E.g.
    mean = [0.60258, 0.44218, 0.57057]
    std = [0.30052, 0.27571, 0.27647]
    TRAIN_IMAGE_PATH = "Patches/train_512/images/"
    TRAIN_MASK_PATH = "Patches/train_512/masks/"
    X_train = os.listdir(TRAIN_IMAGE_PATH)
    train_set = CrowdsourcingDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, X_train, mean, std)
"""

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class CrowdsourcingDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx], 0)
        img = Image.fromarray(img)

        t = T.Compose(
            [T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(self.mean, self.std)]
        )
        img = t(img)
        mask = torch.from_numpy(mask).long()  # to tensor
        return img, mask
