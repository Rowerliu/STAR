#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_refine.py

Manual refinement tool for aligning HE and IHC WSIs based on thumbnails.

This script is intended for interactive usage: the user visually adjusts the
relative position (x, y offsets) of IHC thumbnails to better align with HE
thumbnails. The refined alignment is then applied to the full-resolution WSI
images and saved for downstream tasks.

Notes
-----
- No defensive programming is implemented (as this is a manual refinement tool).
- One adjustment step corresponds to 64 pixels in the original WSI, which
  equals 4 pixels at 16x downsampled thumbnails.
- Positive move_x / move_y = move right / down; negative = move left / up.

Dependencies
------------
- pyvips
- OpenCV
- Pillow
- numpy

Author: [Your Name]
Date: 2025-06-12
License: MIT
"""

import os
import glob
import shutil
import cv2
import numpy as np
from PIL import Image

import abandon_idx_list as shared_inf

# VIPS library path (update if needed)
vipshome = r"F:\DSW\solve_env_bug\vips-dev-8.14\bin"
os.environ["PATH"] = vipshome + ";" + os.environ["PATH"]
import pyvips


# ---------------------------
# Parameters (modify as needed)
# ---------------------------
start_idx = 20
end_idx = 25

png_dir_base = r"..\processed_train\20250305\cropped_wsi"
tn_dir_base = r"..\processed_train\20250305\thumbnail"
mask_dir_base = r"..\processed_train\20250305\mask_npy"
save_refine_png_dir_base = r"..\processed_train\20250305\refine\cropped_wsi"
save_refine_tn_dir_base = r"..\processed_train\20250305\refine\thumbnail"


# ---------------------------
# Utility Functions
# ---------------------------
def apply_mask(image, mask, stride):
    """
    Apply a binary mask to an image (set background pixels to black).

    Parameters
    ----------
    image : PIL.Image
        16x downsampled image.
    mask : np.ndarray
        32x downsampled binary mask.
    stride : int
        Number of pixels in image corresponding to one pixel in mask.

    Returns
    -------
    PIL.Image
        Image with background masked out (black).
    """
    img_np = np.array(image)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0 and stride * i < img_np.shape[0] and stride * j < img_np.shape[1]:
                img_np[stride * i:min(stride * (i + 1), img_np.shape[0]),
                       stride * j:min(stride * (j + 1), img_np.shape[1])] = (0, 0, 0)

    return Image.fromarray(np.uint8(img_np))


def pil_to_cv2(pil_image):
    """Convert PIL.Image to OpenCV BGR image."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert OpenCV BGR image to PIL.Image."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def add_divider(image, stride, new_image_path=""):
    """
    Overlay grid lines on an image for alignment guidance.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    stride : int
        Grid line spacing (in pixels).
    new_image_path : str, optional
        If provided, save image with dividers to this path.

    Returns
    -------
    PIL.Image
        Image with grid lines drawn.
    """
    img = pil_to_cv2(image)
    height, width, _ = img.shape

    # Vertical lines
    for x in range(stride, width, stride):
        color = (0, 255, 0) if abs(x - width / 2) < stride else (0, 0, 255)
        cv2.line(img, (x, 0), (x, height), color, 2)

    # Horizontal lines
    for y in range(stride, height, stride):
        color = (0, 255, 0) if abs(y - height / 2) < stride else (0, 0, 255)
        cv2.line(img, (0, y), (width, y), color, 2)

    if new_image_path:
        cv2.imwrite(new_image_path, img)

    return cv2_to_pil(img)


def add_mask_and_divide(image, mask_1024):
    """Apply mask and overlay dividers."""
    return add_divider(apply_mask(image, mask_1024, 64), 64)


def concatenate_image(image1, image2, flag_show=True):
    """
    Concatenate two PIL images horizontally.

    Parameters
    ----------
    image1 : PIL.Image
    image2 : PIL.Image
    flag_show : bool
        If True, display the concatenated image.

    Returns
    -------
    PIL.Image
        Concatenated image.
    """
    width1, height1 = image1.size
    width2, height2 = image2.size
    assert height1 == height2

    new_image = Image.new("RGB", (width1 + width2, max(height1, height2)))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))

    if flag_show:
        new_image.show()
    return new_image


def move_pic(image, move_x, move_y):
    """
    Shift an image by (move_x, move_y) with black padding.

    Parameters
    ----------
    image : PIL.Image
    move_x : int
        Shift in x-direction (positive = right).
    move_y : int
        Shift in y-direction (positive = down).

    Returns
    -------
    PIL.Image
        Shifted image with black padding.
    """
    width, height = image.size
    new_image = Image.new("RGB", (width, height), color=(0, 0, 0))

    for x in range(width):
        for y in range(height):
            new_x, new_y = x - move_x, y - move_y
            if 0 <= new_x < width and 0 <= new_y < height:
                new_image.putpixel((x, y), image.getpixel((new_x, new_y)))

    return new_image


def move_and_save_big_png(vips_image_path, move_x, move_y, save_refine_png_dir):
    """
    Apply translation to large WSI (PNG) using pyvips and save.

    Parameters
    ----------
    vips_image_path : str
        Path to input WSI PNG.
    move_x : int
        Shift in x-direction (in pixels).
    move_y : int
        Shift in y-direction (in pixels).
    save_refine_png_dir : str
        Directory to save refined PNG.
    """
    vips_image = pyvips.Image.new_from_file(vips_image_path)
    width, height = vips_image.width, vips_image.height

    left = 0 if move_x > 0 else -move_x
    top = 0 if move_y > 0 else -move_y
    crop_width, crop_height = width - abs(move_x), height - abs(move_y)

    vips_image = vips_image.crop(left, top, crop_width, crop_height)
    padded_image = vips_image.embed(
        move_x if move_x > 0 else 0,
        move_y if move_y > 0 else 0,
        width, height, extend="black"
    )

    out_path = os.path.join(save_refine_png_dir, os.path.basename(vips_image_path))
    padded_image.write_to_file(out_path)


def save_pic(idx, he_png_path, ihc_png_path, move_x, move_y,
             final_con_mask_tn, save_refine_tn_dir, save_refine_png_dir,
             final_con_th=None):
    """
    Save refined thumbnails and WSIs.

    Parameters
    ----------
    idx : int
        WSI index.
    he_png_path : str
        Path to HE PNG.
    ihc_png_path : str
        Path to IHC PNG.
    move_x : int
        X-direction shift (unit = 64 pixels).
    move_y : int
        Y-direction shift (unit = 64 pixels).
    final_con_mask_tn : PIL.Image
        Final concatenated masked thumbnail.
    save_refine_tn_dir : str
        Directory to save refined thumbnails.
    save_refine_png_dir : str
        Directory to save refined WSIs.
    final_con_th : PIL.Image, optional
        Final concatenated thumbnail without mask.
    """
    os.makedirs(save_refine_tn_dir, exist_ok=True)
    os.makedirs(save_refine_png_dir, exist_ok=True)

    final_con_mask_tn.save(os.path.join(save_refine_tn_dir, f"{idx}_new_mask.png"))
    if final_con_th:
        final_con_th.save(os.path.join(save_refine_tn_dir, f"{idx}_new.png"))

    shutil.copy2(he_png_path, os.path.join(save_refine_png_dir, os.path.basename(he_png_path)))
    move_and_save_big_png(ihc_png_path, 64 * move_x, 64 * move_y, save_refine_png_dir)


# ---------------------------
# Main Interactive Loop
# ---------------------------
if __name__ == "__main__":
    print("Manual refinement tool initialized.")
    print("Note: 1 unit = 64 pixels in WSI = 4 pixels at 16x thumbnail.")
    print("Positive offset: right/down; Negative: left/up.")

    folder_list = [
        ("HE_ER", "ER"),
        ("HE_HER2", "HER2"),
        ("HE_KI67", "KI67"),
        ("HE_PGR", "PGR"),
    ]
    idx_list = [i for i in range(start_idx, end_idx + 1)]

    # Exclude abandoned indices
    for abandon_idx in shared_inf.abandon_idx_list:
        if abandon_idx in idx_list:
            idx_list.remove(abandon_idx)

    # Process each WSI
    for idx in idx_list:
        print(f"\nProcessing WSI {idx}")
        for he_ihc, ihc_kind in folder_list:

            th_dir = os.path.join(tn_dir_base, he_ihc)
            png_dir = os.path.join(png_dir_base, he_ihc)
            mask_dir = os.path.join(mask_dir_base, he_ihc)
            save_refine_tn_dir = os.path.join(save_refine_tn_dir_base, he_ihc)
            save_refine_png_dir = os.path.join(save_refine_png_dir_base, he_ihc)

            th_path_list = glob.glob(os.path.join(th_dir, f"{idx}_*_z0.png"))
            ori_path_list = glob.glob(os.path.join(png_dir, f"{idx}_*_z0.png"))
            assert (len(th_path_list) == 2 and len(ori_path_list) == 2) or (len(th_path_list) == 0 and len(ori_path_list) == 0)
            if len(th_path_list) == 0:
                continue

            print("\t---------------")

            he_th_path = th_path_list[0] if "_HE_" in th_path_list[0] else th_path_list[1]
            ihc_th_path = th_path_list[1] if "_HE_" in th_path_list[0] else th_path_list[0]
            he_png_path = ori_path_list[0] if "_HE_" in ori_path_list[0] else ori_path_list[1]
            ihc_png_path = ori_path_list[1] if "_HE_" in ori_path_list[0] else ori_path_list[0]

            mask_1024 = np.load(os.path.join(mask_dir, f"{idx}_1024.npy"))
            he_th, ihc_th = Image.open(he_th_path), Image.open(ihc_th_path)

            ori_th = concatenate_image(add_divider(he_th, 64), add_divider(ihc_th, 64))

            while True:
                move_x = int(input("\tShift in x-direction (units of 1=64px): "))
                move_y = int(input("\tShift in y-direction (units of 1=64px): "))

                new_ihc_th = move_pic(ihc_th, 4 * move_x, 4 * move_y)

                final_con_th = concatenate_image(he_th, new_ihc_th, False)
                final_con_mask_tn = concatenate_image(
                    add_mask_and_divide(he_th, mask_1024),
                    add_mask_and_divide(new_ihc_th, mask_1024)
                )

                finished = input("\tIs alignment finished? (1=yes, other=no): ") == "1"
                if finished:
                    print("\tSaving refined images...")
                    save_pic(idx, he_png_path, ihc_png_path, move_x, move_y,
                             final_con_mask_tn, save_refine_tn_dir, save_refine_png_dir, final_con_th)
                    print("\tSaved successfully.")
                    break
