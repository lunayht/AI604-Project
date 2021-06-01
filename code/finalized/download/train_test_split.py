#!/usr/bin/env python3
"""
Split Crowdsourcing dataset to train, test and validation set
"""
import numpy as np
import os
import shutil

from typing import List, Tuple

ORIGINAL_PATH = "dataset/crowdsourcing/patches/"
TARGET_PATH = "dataset/crowdsourcing/"
SPLIT_TYPE = ["train/", "test/"]
TEST_RATIO = 0.2

def create_folder(foldername: str):
    """Create folder to store patches"""
    if not os.path.isdir(foldername):
        os.makedirs(foldername)

def move_files(target_folder: str, all_filenames: List[str]):
    data_types = ["images/", "masks/"]
    path = TARGET_PATH + target_folder
    for data in data_types:
        for filename in all_filenames:
            shutil.copy(ORIGINAL_PATH + data + filename, path + data)
        print(f"Done for {ORIGINAL_PATH + data}: moved to {path + data}")


def shuffle_split(all_files: str) -> Tuple[List, List]:
    np.random.shuffle(all_files)
    train_filenames, test_filenames = np.split(
        np.array(all_files), [int(len(all_files) * (1 - TEST_RATIO))]
    )
    return (train_filenames, test_filenames)


def start():
    for data_type in SPLIT_TYPE:
        create_folder(TARGET_PATH + data_type + "images/")
        create_folder(TARGET_PATH + data_type + "masks/")

    imgs = os.listdir(ORIGINAL_PATH + "images/")
    masks = os.listdir(ORIGINAL_PATH + "masks/")
    assert len(imgs) == len(
        masks
    ), "Number of image patches and mask patches should be equal"

    (train_filenames, test_filenames) = shuffle_split(imgs)
    print(
        f"Total: {len(imgs)}\nTrain: {len(train_filenames)} ; Test: {len(test_filenames)}"
    )

    move_files(target_folder="test/", all_filenames=test_filenames)
    move_files(target_folder="train/", all_filenames=train_filenames)

    DATA_PATH = "data/"

    for dt in SPLIT_TYPE:
        img_path = os.listdir(DATA_PATH + dt + "images/")
        mask_path = os.listdir(DATA_PATH + dt + "masks/")
        assert len(img_path) == len(
            mask_path
        ), "Number of images and masks should be equal"


if __name__ == "__main__":
    start()