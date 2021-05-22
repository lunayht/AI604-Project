#!/usr/bin/env python3
import cv2
import numpy as np
import os

from PIL import Image
from torchvision.transforms.functional import pad
from typing import Tuple

IMAGES_PATH = "dataset/images"  # Original Images Folder
MASKS_PATH = "dataset/masks"  # Original Masks Folder
IMAGE_PATCHES = "dataset/patches/images"  # Destination (Images)
MASK_PATCHES = "dataset/patches/masks"  # Destination (Masks)
PATCH_SIZE = 512


def create_folder(foldername: str):
    """Create folder to store patches"""
    if not os.path.isdir(foldername):
        os.makedirs(foldername)


def compute_pixel_diff(ori_size: int, patch_size: int = 1024) -> int:
    """Compute pixel different for padding purpose"""
    return patch_size - (ori_size % patch_size)


def compute_padding_size(pixels: int) -> Tuple[int, int]:
    """Compute padding size for each side (left, right) or (top, bottom)"""
    pad_size = pixels // 2
    return (pad_size, pad_size) if (pixels % 2 == 0) else (pad_size, pad_size + 1)


def get_padding(img: np.ndarray, patch_size: int = 1024) -> Tuple[int, int, int, int]:
    """Get padding size for four sides"""
    diff_height = compute_pixel_diff(img.shape[0], patch_size)
    diff_width = compute_pixel_diff(img.shape[1], patch_size)
    horizontal_pad = compute_padding_size(diff_height)
    vertical_pad = compute_padding_size(diff_width)
    return (vertical_pad[0], horizontal_pad[0], vertical_pad[1], horizontal_pad[1])


def pad_image(img: np.ndarray, patch_size: int = 1024) -> np.ndarray:
    """Pad image/ mask according to padding size"""
    padding = get_padding(img, patch_size)
    return np.array(pad(Image.fromarray(img), padding))


def generate_image_patches(img: np.ndarray, patch_size: int = 1024):
    """Generate patches for each original image/ mask"""
    for y in range(0, img.shape[0], patch_size):
        y_end = y + patch_size
        for x in range(0, img.shape[1], patch_size):
            x_end = x + patch_size
            yield img[y:y_end, x:x_end]


if __name__ == "__main__":
    create_folder(IMAGE_PATCHES)
    create_folder(MASK_PATCHES)

    filelist = os.listdir(IMAGES_PATH)
    for count, filename in enumerate(filelist):
        img = cv2.imread(os.path.join(IMAGES_PATH, filename))
        mask = cv2.imread(os.path.join(MASKS_PATH, filename), 0)
        new_img = pad_image(img, PATCH_SIZE)
        new_mask = pad_image(mask, PATCH_SIZE)
        idx = 0
        for img_patch, mask_patch in zip(
            generate_image_patches(new_img, PATCH_SIZE),
            generate_image_patches(new_mask, PATCH_SIZE),
        ):
            outside_roi = np.all(mask_patch == 0)
            if not outside_roi:
                savename = filename.split(".png")[0] + "_" + str(idx) + ".png"
                cv2.imwrite(os.path.join(IMAGE_PATCHES, savename), img_patch)
                cv2.imwrite(os.path.join(MASK_PATCHES, savename), mask_patch)
                idx += 1
        print(f"Done {round(count/len(filelist)*100, 2)}% : {filename} ({idx} patches)")
