#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_pair.py

This script performs registration and preprocessing of Whole Slide Images (WSIs).
It includes foreground detection, bounding box cropping, and registration between
reference (HE) and target (IHC) images, with outputs of cropped WSIs, thumbnails,
and masks for subsequent analysis.

Author: [Your Name]
Date: 2025-06-12
License: MIT
"""

import os
import argparse
import numpy as np
from PIL import Image

# Project-specific imports
import abandon_idx_list as shared_inf
import utils_add
import utils

from utils import (
    foreground_detection_model,
    get_foreground,
    get_bbox,
    simple_bbox_crop,
    ref_conv_input,
    trs_conv_input,
    register,
)


def get_args_parser():
    """
    Parse command-line arguments for WSI registration.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="WSI Pair Registration Pipeline")

    # Input / Output configuration
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index of WSIs to process."
    )
    parser.add_argument(
        "--end_idx", type=int, default=99,
        help="End index of WSIs to process."
    )
    parser.add_argument(
        "--processed_dir", type=str,
        default=r"F:\13_Data\03_Pathology\z05_ANHIR\01_custom\registration\lunglobe\20250612",
        help="Directory for saving processed registration results."
    )
    parser.add_argument(
        "--dataset_dir", type=str,
        default=r"F:\13_Data\03_Pathology\z05_ANHIR\00_original\lung-lobes_4\scale-100pc",
        help="Directory of original dataset containing WSI images."
    )
    parser.add_argument(
        "--prefix", type=str, default="4",
        help="Prefix for naming output files (e.g., kidney_5 → prefix=5)."
    )

    # Reference / Registration configuration
    parser.add_argument(
        "--reference", type=str, default="HE",
        help="Reference image type for registration (default: HE)."
    )

    # Foreground detection configuration
    parser.add_argument(
        "--foreground_model_path", type=str, default="foreground_detect.pt",
        metavar="", help="Path to the pretrained foreground detection model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, metavar="",
        help="Batch size for foreground detection."
    )
    parser.add_argument(
        "--white_thr", type=int, default=230, metavar="",
        help="Threshold for white pixel intensity."
    )
    parser.add_argument(
        "--black_thr", type=int, default=20, metavar="",
        help="Threshold for black pixel intensity."
    )

    # Mask configuration
    parser.add_argument(
        "--mask_size", type=int, default=256, metavar="",
        help="Size of the mask used for cropping."
    )
    parser.add_argument(
        "--mask_thr", type=float, default=0.25, metavar="",
        help="Threshold for mask ratio filtering."
    )
    parser.add_argument(
        "--downsample", type=int, default=32, metavar="",
        help="Downsample factor for WSIs."
    )

    return parser.parse_args()


def check_finished(ref_path, trs_path, save_wsi_dir, mask_dir, mask_size):
    """
    Check if the current pair has already been processed.

    Parameters
    ----------
    ref_path : str
        Path to the reference image.
    trs_path : str
        Path to the target (IHC) image.
    save_wsi_dir : str
        Directory for saving cropped WSIs.
    mask_dir : str
        Directory for saving mask files.
    mask_size : int
        Size of the mask.

    Returns
    -------
    bool
        True if all outputs exist, False otherwise.
    """
    ref_name = os.path.splitext(os.path.basename(ref_path))[0]
    trs_name = os.path.splitext(os.path.basename(trs_path))[0]
    idx = ref_name.split("_")[0]

    return all([
        os.path.exists(os.path.join(save_wsi_dir, f"{ref_name}.png")),
        os.path.exists(os.path.join(save_wsi_dir, f"{trs_name}.png")),
        os.path.exists(os.path.join(mask_dir, f"{idx}_{mask_size}.npy")),
    ])


def create_all_dir_new(processed_dir):
    """
    Create all required directories for saving intermediate and final results.

    Parameters
    ----------
    processed_dir : str
        Path to the root processed directory.
    """
    utils_add.check_and_create_folder("./temp")
    utils_add.check_and_create_folder(os.path.join(processed_dir, "cropped_wsi"))
    utils_add.check_and_create_folder(os.path.join(processed_dir, "thumbnail"))
    utils_add.check_and_create_folder(os.path.join(processed_dir, "mask_npy"))


def get_directories(path):
    """
    Get all directories under a specified path.

    Parameters
    ----------
    path : str
        Input directory path.

    Returns
    -------
    list of str
        List of directory names under the given path.
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


if __name__ == "__main__":
    # Parse input arguments
    args = get_args_parser()

    start_idx, end_idx = args.start_idx, args.end_idx
    processed_dir, dataset_dir = args.processed_dir, args.dataset_dir
    save_wsi_dir = os.path.join(processed_dir, "cropped_wsi")

    # Foreground detection parameters
    batch_size = args.batch_size
    down_sample = args.downsample
    white_thr, black_thr = args.white_thr, args.black_thr
    foreground_model_path = args.foreground_model_path

    # Initialize model and helper lists
    model = foreground_detection_model(foreground_model_path)
    failed_list, already_finished_list, lack_tiff_list = [], [], []
    abandon_idx_list = shared_inf.abandon_idx_list

    # Create result directories
    create_all_dir_new(processed_dir)

    # Load dataset image names
    img_names = os.listdir(dataset_dir)
    prefix, ref_name = args.prefix, args.reference
    mask_size, mask_thr = args.mask_size, args.mask_thr

    # Identify reference and target paths
    ref_path, trs_paths = "", []
    for img_name in img_names:
        full_path = os.path.join(dataset_dir, img_name)
        if ref_name in img_name:
            ref_path = full_path
        else:
            trs_paths.append(full_path)

    # Foreground detection
    foreground_kwargs = {
        "batch_size": batch_size,
        "white_thr": white_thr,
        "black_thr": black_thr,
        "downsample": down_sample,
    }

    cache_path = f"./temp/{prefix}_{ref_name}.npy"
    if os.path.exists(cache_path):
        foreground = np.load(cache_path)
    else:
        foreground = get_foreground(model, ref_path, **foreground_kwargs)
        np.save(cache_path, foreground)

    utils.show_image(foreground, "Original Foreground Mask")

    # Bounding box cropping
    cropped_mask, bbox = simple_bbox_crop(foreground)
    utils.show_image(cropped_mask, "Cropped Foreground Mask")

    print("\tLoading Reference Image...")
    ref_conv = ref_conv_input(ref_path, bbox, foreground, down_sample)

    # Prepare saving directories
    thumbnail_dir = os.path.join(processed_dir, "thumbnail")
    wsi_dir = os.path.join(processed_dir, "cropped_wsi")
    mask_dir = os.path.join(processed_dir, "mask_npy")

    for d in [thumbnail_dir, wsi_dir, mask_dir]:
        utils_add.check_and_create_folder(d)

    # Save reference image
    ref_tn_path, ref_tn_mask_divide_path, mask = utils_add.save_ref_no_rotate(
        ref_path, prefix, down_sample, mask_size, mask_thr, bbox, cropped_mask,
        thumbnail_dir, wsi_dir, mask_dir
    )

    # Process target (IHC) images
    divide_trs_tn_paths, divide_trs_mask_paths = [], []
    for trs_path in trs_paths:
        trs_name = os.path.splitext(os.path.basename(trs_path))[0]
        divide_trs_tn_path = os.path.join(thumbnail_dir, f"{prefix}_{trs_name}_divide.png")
        divide_trs_mask_path = os.path.join(thumbnail_dir, f"{prefix}_{trs_name}_mask{mask_size}_divide.png")

        divide_trs_tn_paths.append(divide_trs_tn_path)
        divide_trs_mask_paths.append(divide_trs_mask_path)

        trs_conv = trs_conv_input(trs_path)

        print("\tRegistering Images...")
        data = register(ref_conv, trs_conv)

        trs_ori_path = os.path.join(wsi_dir, f"{prefix}_{trs_name}.png")
        trs_tn_path = utils_add.save_trs_rotate(
            trs_path, prefix, data, down_sample, mask_size, bbox,
            thumbnail_dir, wsi_dir
        )

        # Generate dividers
        stride = mask_size // 16
        _ = utils_add.add_divider(trs_tn_path, stride, divide_trs_tn_path)

        # Apply mask and save
        trs_img = Image.open(trs_tn_path)
        trs_mask = utils_add.apply_mask(trs_img, mask, stride)
        trs_mask_path = os.path.join(thumbnail_dir, f"{prefix}_{trs_name}_mask{mask_size}.png")
        trs_mask.save(trs_mask_path)

        _ = utils_add.add_divider(trs_mask_path, stride, divide_trs_mask_path)

    # Concatenate reference and target images
    divide_ref_tn = utils_add.add_divider(ref_tn_path, stride, ref_tn_path[:-4] + "_divide.png")
    for i, trs_path in enumerate(trs_paths):
        trs_name = os.path.splitext(os.path.basename(trs_path))[0]
        utils_add.concatenate_images(
            os.path.join(thumbnail_dir, f"{prefix}_{ref_name}_{trs_name}.png"),
            image1=divide_ref_tn, image2=None, image2_path=divide_trs_tn_paths[i]
        )
        utils_add.concatenate_images(
            os.path.join(thumbnail_dir, f"{prefix}_{ref_name}_{trs_name}_contrast.png"),
            image1=None, image2=None,
            image1_path=ref_tn_mask_divide_path, image2_path=divide_trs_mask_paths[i]
        )

    # Concatenate all images for overview
    utils_add.concatenate_many_images(
        os.path.join(thumbnail_dir, f"{prefix}_whole.png"),
        image1=divide_ref_tn, images=None, images_path=divide_trs_tn_paths
    )
    utils_add.concatenate_many_images(
        os.path.join(thumbnail_dir, f"{prefix}_whole_contrast.png"),
        image1=None, images=None,
        image1_path=ref_tn_mask_divide_path, images_path=divide_trs_mask_paths
    )

    print("Processing complete.")
