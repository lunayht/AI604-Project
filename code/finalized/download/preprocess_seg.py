import os
import cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from patchify import patchify
from sklearn.model_selection import train_test_split

def extract_patches(image_list, mask_src, image_src, mask_dst, image_dst, patch_size):
    """Reads image and mask then extract patches and disregard patches outside of RoI"""
    for im in tqdm(image_list):
        img = cv2.imread(os.path.join(image_src, im))
        msk = cv2.imread(os.path.join(mask_src, im), 0)

        img_patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
        msk_patches = patchify(msk, (patch_size, patch_size), step=patch_size)
        img_patches = img_patches.reshape((-1, patch_size, patch_size, 3))
        msk_patches = msk_patches.reshape((-1, patch_size, patch_size))
        # Step=256 for patch_sie patches means no overlap
        for i in range(img_patches.shape[0]):
            mask_patch = msk_patches[i]
            unique  = np.unique(mask_patch)
            outside = np.all(mask_patch == 0)
            if not outside and (len(unique) > 1):
                img_patch = img_patches[i]
                filename = im.split(".png")[0] + "_" + str(i) + ".png"
                cv2.imwrite(os.path.join(image_dst, filename), img_patch)
                cv2.imwrite(os.path.join(mask_dst, filename), mask_patch)

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':
    df = pd.read_csv('download/pathdata.csv')
    train_df, test_df= train_test_split(df, test_size=0.2, random_state=42)
    
    image_path = 'dataset/crowdsourcing/images/'
    mask_path  = 'dataset/crowdsourcing/masks/'
    
    create_folder('dataset/crowdsourcing/patches')
    create_folder('dataset/crowdsourcing/patches/train')
    create_folder('dataset/crowdsourcing/patches/train/masks')
    create_folder('dataset/crowdsourcing/patches/train/images')

    create_folder('dataset/crowdsourcing/patches/test')
    create_folder('dataset/crowdsourcing/patches/test/masks')
    create_folder('dataset/crowdsourcing/patches/test/images')

    # Process train saet
    image_list = train_df.images_path.values.tolist()
    mask_src   = mask_path
    image_src  = image_path
    mask_dst   = 'dataset/crowdsourcing/patches/train/masks/'
    image_dst  = 'dataset/crowdsourcing/patches/train/images/'
    patch_size = 512
    extract_patches(image_list, mask_src, image_src, mask_dst, image_dst, patch_size)

    # Process test set
    image_list = test_df.images_path.values.tolist()
    mask_src   = mask_path
    image_src  = image_path
    mask_dst   = 'dataset/crowdsourcing/patches/test/masks/'
    image_dst  = 'dataset/crowdsourcing/patches/test/images/'
    patch_size = 512
    extract_patches(image_list, mask_src, image_src, mask_dst, image_dst, patch_size)