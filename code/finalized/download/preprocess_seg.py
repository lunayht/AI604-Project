import os
import cv2
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from math import log2
from patchify import patchify
from collections import defaultdict
from sklearn.model_selection import train_test_split

"""
    We follow the method 

    outside_roi	0
    tumor	                1   ->   tumor
    stroma	                2   ->   stroma
    lymphocytic_infiltrate	3   ->   inflammatory infiltrates
    necrosis_or_debris	    4   ->   necrosis
    glandular_secretions	5   ->   other
    blood	                6   ->   other
    exclude	                7   ->   other
    metaplasia_NOS	        8   ->   other
    fat	                    9   ->   other
    plasma_cells	        10  ->   inflammatory infiltrates
    other_immune_infiltrate	11  ->   inflammatory infiltrates
    mucoid_material	        12  ->   other
    normal_acinus_or_duct	13  ->   other
    lymphatics	            14  ->   other
    undetermined	        15  ->   other
    nerve	                16  ->   other
    skin_adnexa	            17  ->   other
    blood_vessel	        18  ->   other
    angioinvasion	        19  ->   tumor
    dcis	                20  ->   tumor
    other	                21  ->   other
"""

CLASS_MAP = \
    {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 5,
        7: 5,
        8: 5,
        9: 5,
        10: 3,
        11: 3,
        12: 5,
        13: 5,
        14: 5,
        15: 5,
        16: 5,
        17: 5,
        18: 5,
        19: 1,
        20: 1,
        21: 5
    }

def replace_classes(mask):
    new_mask = np.copy(mask)
    for k, v in CLASS_MAP.items():
        new_mask[mask == k] = v
    return new_mask

def extract_patches(image_list, mask_src, image_src, mask_dst, image_dst, patch_size):
    """Reads image and mask then extract patches and disregard patches outside of RoI"""
    class_counts = defaultdict(lambda: 0)
    skipped = 0
    total = 0
    for im in tqdm(image_list):
        img = cv2.imread(os.path.join(image_src, im))
        msk = cv2.imread(os.path.join(mask_src, im), 0)
        
        assert (img.shape[0] == msk.shape[0]) \
            and (img.shape[1] == msk.shape[1]), "Mismatch!"

        img_patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
        msk_patches = patchify(msk, (patch_size, patch_size), step=patch_size)
        img_patches = img_patches.reshape((-1, patch_size, patch_size, 3))
        msk_patches = msk_patches.reshape((-1, patch_size, patch_size))
        # Step = 256 for patch size means no overlap
        for i in range(img_patches.shape[0]):
            # Replace class labels
            mask_patch = replace_classes(msk_patches[i])
            unique, counts = np.unique(mask_patch, return_counts=True)
            # If outside of RoI takes > 90% and there is only 1 class, ignore the patch.
            outside = np.mean(mask_patch == 0) > 0.9
            if outside and (len(unique) < 2):
                skipped += 1
                continue
            for x, y in enumerate(unique):
                class_counts[y] += counts[x].item()
            img_patch = img_patches[i]
            filename = im.split(".png")[0] + "_" + str(i) + ".png"
            cv2.imwrite(os.path.join(image_dst, filename), img_patch)
            cv2.imwrite(os.path.join(mask_dst, filename), mask_patch)
            total += 1
    print('Skipped: {} / {}'.format(skipped, total))
    return class_counts

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--patch_size', '-p',
        type=int, default=512
    )
    parser.add_argument(
        '--test_split', '-s',
       type=float, default=0.2
    )
    args = parser.parse_args()

    assert args.test_split < 0.5, "Test split should be less than half of the dataset ..."
    assert int(log2(args.patch_size)) == log2(args.patch_size), \
        "Patch size should be power of 2 (necessary for fitting with ViT patch size and some other U-Net like architectures) ..."

    df = pd.read_csv('download/path_data.csv')
    train_df, test_df= train_test_split(df, test_size=args.test_split, random_state=42)

    image_path = 'dataset/crowdsourcing/images/'
    mask_path  = 'dataset/crowdsourcing/masks/'
    
    create_folder('dataset/crowdsourcing/patches-{}'.format(args.patch_size))

    create_folder('dataset/crowdsourcing/patches-{}/train'.format(args.patch_size))
    create_folder('dataset/crowdsourcing/patches-{}/train/masks'.format(args.patch_size))
    create_folder('dataset/crowdsourcing/patches-{}/train/images'.format(args.patch_size))

    create_folder('dataset/crowdsourcing/patches-{}/test'.format(args.patch_size))
    create_folder('dataset/crowdsourcing/patches-{}/test/masks'.format(args.patch_size))
    create_folder('dataset/crowdsourcing/patches-{}/test/images'.format(args.patch_size))

    # Process train set
    image_list = train_df.images_path.values.tolist()
    mask_src   = mask_path
    image_src  = image_path
    mask_dst   = 'dataset/crowdsourcing/patches-{}/train/masks/'.format(args.patch_size)
    image_dst  = 'dataset/crowdsourcing/patches-{}/train/images/'.format(args.patch_size)
    patch_size = args.patch_size
    train_cnt  = extract_patches(image_list, mask_src, image_src, mask_dst, image_dst, patch_size)

    # Save train class counts (used for weighted CE)
    f = open("dataset/crowdsourcing/patches-{}/train/class_weights.txt".format(args.patch_size), "a")
    f.write(str(train_cnt))
    f.close()

    # Process test set
    image_list = test_df.images_path.values.tolist()
    mask_src   = mask_path
    image_src  = image_path
    mask_dst   = 'dataset/crowdsourcing/patches-{}/test/masks/'.format(args.patch_size)
    image_dst  = 'dataset/crowdsourcing/patches-{}/test/images/'.format(args.patch_size)
    patch_size = args.patch_size
    extract_patches(image_list, mask_src, image_src, mask_dst, image_dst, patch_size)