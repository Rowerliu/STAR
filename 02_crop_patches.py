#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_pair_patches.py

This script crops multi-stain Whole Slide Images (WSIs) into patches for training.
It uses precomputed binary masks to select tissue regions, ensuring that only
foreground regions are cropped and saved.

Workflow
--------
1. Parse arguments (input/output directories, WSI indices, patch parameters).
2. Skip abandoned indices (from shared config).
3. For each WSI and each stain type (e.g., HE, MAS, PAS, PASM):
   - Load the cropped WSI and corresponding mask.
   - Extract patches with given patch size and stride.
   - Save patches into the specified folder.

Dependencies
------------
- pyvips (for efficient WSI handling)
- Pillow (for image manipulation)
- numpy (for mask loading)

Author: [Your Name]
Date: 2025-06-12
License: MIT
"""

import os
import glob
import argparse

import numpy as np
import PIL.Image
import pyvips

# Project-specific
import abandon_idx_list as shared_inf


# ---------------------------
# Argument Parser
# ---------------------------
def get_args_parser():
    """
    Parse command-line arguments for WSI patch extraction.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="WSI Patch Extraction Pipeline")

    # Index range
    parser.add_argument("--start_idx", type=int, default=1,
                        help="Start index of WSIs to process.")
    parser.add_argument("--end_idx", type=int, default=5,
                        help="End index of WSIs to process.")

    # Quality filtering
    parser.add_argument("--black_pixel_ratio_threshold", type=float, default=0.001,
                        help="Discard patches if black pixel ratio exceeds this threshold.")

    # Output directories
    parser.add_argument("--save_patch_dir", type=str,
                        default=r"F:\13_Data\03_Pathology\z05_ANHIR\01_custom\registration\kidney\20250612_v5\patches",
                        help="Directory to save extracted patches.")
    parser.add_argument("--save_dir_cls", type=list, default=["HE", "MAS", "PAS", "PASM"],
                        help="List of stain classes to process.")

    # Input directories
    parser.add_argument("--wsi_dir", type=str,
                        default=r"F:\13_Data\03_Pathology\z05_ANHIR\01_custom\registration\kidney\20250612_v5\cropped_wsi",
                        help="Directory containing cropped WSIs (PNG).")
    parser.add_argument("--mask_dir", type=str,
                        default=r"F:\13_Data\03_Pathology\z05_ANHIR\01_custom\registration\kidney\20250612_v5\mask_npy",
                        help="Directory containing binary mask files (.npy).")

    # Patch parameters
    parser.add_argument("--mask_size", type=int, default=256, metavar="",
                        help="Size of the mask unit.")
    parser.add_argument("--crop_patch_size", type=int, default=256,
                        help="Size of each cropped patch.")
    parser.add_argument("--crop_overlap", type=int, default=0,
                        help="Overlap size between adjacent patches.")

    return parser.parse_args()


# ---------------------------
# Utility Functions
# ---------------------------
def count_black_prop(image_tensor):
    """
    Count the proportion of black pixels in a patch.

    Parameters
    ----------
    image_tensor : torch.Tensor or np.ndarray
        Patch image in tensor format (expected to have .numpy()).

    Returns
    -------
    float
        Proportion of black pixels (0-1).
    """
    image = PIL.Image.fromarray(image_tensor.numpy())
    width, height = image.size

    black_count = sum(
        1 for i in range(width) for j in range(height)
        if image.getpixel((i, j)) == (0, 0, 0)
    )

    return black_count / (width * height)


def crop_image(image_path, patch_save_dir, patch_size, stride, mask, cls):
    """
    Crop patches from a given WSI using mask filtering.

    Parameters
    ----------
    image_path : str
        Path to the WSI PNG image.
    patch_save_dir : str
        Directory to save cropped patches.
    patch_size : int
        Size of each cropped patch (in pixels).
    stride : int
        Stride between adjacent patches.
    mask : np.ndarray
        Binary mask array indicating valid regions (0=background, 1=foreground).
    cls : str
        Stain class (e.g., HE, MAS, PAS).

    Returns
    -------
    None
    """
    image = pyvips.Image.new_from_file(image_path)
    wsi_idx = int(os.path.basename(image_path).split(".")[0].split("_")[0])
    width, height = image.width, image.height

    total_patches, skipped_patches = 0, 0
    for j in range(0, height - patch_size + 1, stride):
        for i in range(0, width - patch_size + 1, stride):

            # Skip background tiles
            if mask[j // patch_size][i // patch_size] == 0:
                skipped_patches += 1
                continue

            # Crop and save
            patch = image.crop(i, j, patch_size, patch_size)
            patch_name = f"{cls}_{wsi_idx:04d}_Y{j // stride:03d}_X{i // stride:03d}.jpg"
            patch.write_to_file(os.path.join(patch_save_dir, patch_name))

            total_patches += 1

    print(f"\tExtracted {total_patches} patches for {cls}, skipped {skipped_patches} background patches.")


# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    args = get_args_parser()

    # Parameters
    start_idx, end_idx = args.start_idx, args.end_idx
    save_patch_dir_base = args.save_patch_dir
    wsi_dir, mask_dir = args.wsi_dir, args.mask_dir
    mask_size = args.mask_size
    folder_list = args.save_dir_cls
    ref_name = folder_list[0]  # reference class (usually "HE")

    # Tracking lists
    failed_list, lack_png_list, lack_mask_list = [], [], []
    abandon_idx_list = shared_inf.abandon_idx_list

    # Determine indices to process
    idx_list = [i for i in range(start_idx, end_idx + 1) if i not in abandon_idx_list]

    print("\n" + "-" * 40)
    print(f"Processing WSIs from {start_idx} to {end_idx}")
    print(f"Abandoned indices: {abandon_idx_list}")
    print(f"Final list of WSIs to process ({len(idx_list)}): {idx_list}")

    # Process each WSI
    for idx in idx_list:
        print(f"\nProcessing WSI {idx}")

        for cls in folder_list:
            print("\t----------------------------")
            save_patch_dir = os.path.join(save_patch_dir_base, cls)
            os.makedirs(save_patch_dir, exist_ok=True)

            # Skip if already processed
            if glob.glob(os.path.join(save_patch_dir, f"{cls}_{idx:04d}_*")):
                print(f"\tWSI {idx} ({cls}) already processed, skipping.")
                continue

            # Check WSI existence
            wsi_path = os.path.join(wsi_dir, f"{idx}_{cls}.png")
            if not os.path.exists(wsi_path):
                print(f"\tMissing PNG for WSI {idx}, class {cls}. Skipping.")
                failed_list.append(idx)
                lack_png_list.append(idx)
                continue

            # Check mask existence
            mask_path = os.path.join(mask_dir, f"{idx}_{ref_name}_{mask_size}.npy")
            if not os.path.exists(mask_path):
                print(f"\tMissing mask for WSI {idx}. Skipping.")
                failed_list.append(idx)
                lack_mask_list.append(idx)
                continue
            else:
                mask = np.load(mask_path)

            # Perform cropping
            patch_size = args.crop_patch_size
            stride = patch_size - args.crop_overlap
            crop_image(wsi_path, save_patch_dir, patch_size, stride, mask, cls)

    # Final summary
    total_files = len(idx_list) * len(folder_list)
    print("\n" + "-" * 50)
    print(f"Processed WSIs from {start_idx} to {end_idx}")
    print(f"Expected files: {total_files}, Success: {total_files - len(failed_list)}, Failures: {len(failed_list)}")
    print(f"Failed WSI indices: {failed_list}")
    print(f"Missing PNG indices: {lack_png_list}")
    print(f"Missing Mask indices: {lack_mask_list}")
