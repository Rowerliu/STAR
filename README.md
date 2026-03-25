# STAR

STAR is a fast and robust rigid registration framework for serial histopathological images. This repository provides a practical pipeline for:

1. Automatic rigid registration between one reference slide and multiple target slides
2. Tissue-aware cropping of aligned whole slide images
3. Mask-guided paired patch extraction for downstream training or analysis
4. Optional manual refinement based on thumbnails

The code is suitable for serial pathology slides with relatively consistent morphology and is currently organized as a simple three-stage workflow:

- `01_pair.py`: automatic registration and result export
- `02_crop_patches.py`: crop paired patches from registered slides
- `03_refine.py`: manual refinement for difficult cases

The original README was very brief. This document is intended to help new users understand the code structure, prepare data correctly, and run the pipeline quickly.

## 1. Repository Structure

```text
STAR/
|-- 01_pair.py
|-- 02_crop_patches.py
|-- 03_refine.py
|-- utils.py
|-- utils_add.py
|-- wsi_dataset.py
|-- abandon_idx_list.py
|-- README.md
```

Main file roles:

- `01_pair.py`
  - Loads one reference image and several target images from a case folder
  - Detects foreground on the reference image
  - Crops the common tissue region
  - Estimates rigid transform for each target image
  - Saves cropped WSIs, thumbnails, masks, and visualization panels
- `02_crop_patches.py`
  - Uses the mask generated from the reference slide
  - Crops aligned patches from each registered stain
  - Saves patches by stain category
- `03_refine.py`
  - Provides an interactive manual correction tool
  - Lets the user shift target thumbnails relative to the reference
  - Applies the same correction to the full cropped PNG
- `utils.py`
  - Foreground detection
  - Bounding-box extraction
  - Reference/target preprocessing
  - Rotation + translation search for rigid registration
- `utils_add.py`
  - Save registered WSIs and thumbnails
  - Build masks at different scales
  - Draw grid overlays and generate comparison images
- `wsi_dataset.py`
  - Patch dataset used during foreground detection inference
- `abandon_idx_list.py`
  - Used to skip problematic case indices

## 2. Pipeline Overview

The practical workflow is:

1. Prepare one case folder containing one reference slide and one or more target slides
2. Run `01_pair.py` to complete automatic rigid registration
3. Check the thumbnails in `processed_dir/thumbnail`
4. If needed, run `03_refine.py` for manual correction
5. Run `02_crop_patches.py` to crop training or analysis patches

Conceptually, the pipeline works as follows:

1. A foreground classifier identifies tissue regions on the reference slide
2. A foreground bounding box is extracted and used as the common crop region
3. The reference image is converted into a grayscale template
4. Each target image is converted into a grayscale response map
5. Registration is performed by coarse-to-fine rotation search plus translation search
6. Registered full-resolution crops and thumbnails are exported
7. A binary tissue mask derived from the reference slide is reused for all aligned slides

This means the pipeline assumes:

- registration is rigid, not deformable
- one slide in each case acts as the reference
- all target slides belong to the same tissue section series and overlap substantially with the reference

## 3. Environment and Dependencies

From the imports in the source code, the main dependencies are:

- Python 3.9 or newer is recommended
- `numpy`
- `pillow`
- `opencv-python`
- `matplotlib`
- `pandas`
- `tqdm`
- `torch`
- `torchvision`
- `tifffile`
- `pyvips`

Suggested installation:

```bash
pip install numpy pillow opencv-python matplotlib pandas tqdm tifffile pyvips
pip install torch torchvision
```

### 3.1 GPU Requirement

Foreground detection and registration currently use CUDA directly in several places, for example:

- the foreground classification model is moved to GPU with `.cuda()`
- tensors used during registration are also moved to GPU

So the current code is written with NVIDIA GPU execution in mind. If you want CPU-only support, you will need to modify the code.

### 3.2 pyvips Requirement

`02_crop_patches.py` and `03_refine.py` rely on `pyvips`.

On Windows, in addition to installing the Python package, you usually also need the native libvips binaries and must ensure they are visible in `PATH`.

Note:

- `03_refine.py` currently hardcodes a local `vipshome` path:
  - `F:\DSW\solve_env_bug\vips-dev-8.14\bin`
- You will need to change this path to your own local libvips installation before running the refinement tool.

### 3.3 Foreground Model Checkpoint

`01_pair.py` requires a pretrained foreground detection model:

- default file name: `foreground_detect.pt`

The original repository README provided the following download link:

- <https://drive.google.com/file/d/1V_0IsgIKjEV2VxWYclyAhCfCoJ2dXMDs/view?usp=sharing>

Place the downloaded checkpoint in the repository root, or pass its path through `--foreground_model_path`.

## 4. Data Organization

### 4.1 Input Organization for `01_pair.py`

`01_pair.py` expects `dataset_dir` to point to one case folder containing:

- one reference image whose file name contains the reference keyword, default `HE`
- one or more target images whose file names do not contain that keyword

Example:

```text
case_0001/
|-- 1_HE.tif
|-- 1_MAS.tif
|-- 1_PAS.tif
|-- 1_PASM.tif
```

With this folder:

- `--dataset_dir case_0001`
- `--reference HE`
- `--prefix 1`

the script will treat `1_HE.tif` as reference and the others as target slides.

### 4.2 Supported Image Formats

From the code, the pipeline directly reads:

- `.tif`
- `.tiff`
- common image formats readable by OpenCV such as `.png`, `.jpg`

For TIFF files, reading is handled with `tifffile.imread`.

### 4.3 Naming Convention

The saved file names depend heavily on `prefix` and the original base names.

For best compatibility, it is recommended that:

- each case has a unique integer-like prefix such as `1`, `2`, `3`
- stain names are included in the original file names, such as `1_HE`, `1_MAS`, `1_PAS`

This matches the expectations in both `01_pair.py` and `02_crop_patches.py`.

## 5. Stage 1: Automatic Registration

Run:

```bash
python 01_pair.py \
  --dataset_dir /path/to/case_0001 \
  --processed_dir /path/to/output/case_0001_result \
  --prefix 1 \
  --reference HE \
  --foreground_model_path foreground_detect.pt
```

### 5.1 Important Arguments

- `--dataset_dir`
  - Input case folder containing one reference slide and several target slides
- `--processed_dir`
  - Root folder for all outputs
- `--prefix`
  - Case identifier used in saved file names
- `--reference`
  - Keyword used to locate the reference slide, default is `HE`
- `--foreground_model_path`
  - Path to the pretrained tissue/foreground classifier
- `--batch_size`
  - Batch size for foreground inference
- `--white_thr`
  - White threshold used in candidate patch filtering
- `--black_thr`
  - Black threshold used in candidate patch filtering
- `--mask_size`
  - Mask cell size used for later patch cropping, default `256`
- `--mask_thr`
  - Threshold for deciding whether a mask cell is valid foreground
- `--downsample`
  - Downsample ratio used during preprocessing and registration, default `32`

### 5.2 What `01_pair.py` Produces

The script creates the following directories under `processed_dir`:

```text
processed_dir/
|-- cropped_wsi/
|-- thumbnail/
|-- mask_npy/
```

Typical outputs:

#### `cropped_wsi/`

- cropped reference PNG
- cropped and rotated target PNGs

Example:

```text
cropped_wsi/
|-- 1_HE.png
|-- 1_MAS.png
|-- 1_PAS.png
|-- 1_PASM.png
```

#### `thumbnail/`

- thumbnail of the cropped reference
- thumbnail of each registered target
- masked thumbnails
- grid-overlaid thumbnails
- side-by-side comparison images
- whole-case montage images

Typical examples:

- `1_HE.png`
- `1_HE_mask256.png`
- `1_HE_mask256_divide.png`
- `1_MAS.png`
- `1_MAS_mask256.png`
- `1_MAS_divide.png`
- `1_MAS_mask256_divide.png`
- `1_HE_MAS.png`
- `1_HE_MAS_contrast.png`
- `1_whole.png`
- `1_whole_contrast.png`

#### `mask_npy/`

- reference-derived binary mask reused for all stains

Example:

- `1_HE_256.npy`

### 5.3 How Registration Is Implemented

Based on the source code, the rigid registration strategy is:

1. Generate a reference foreground mask using a ResNet18 foreground classifier
2. Crop the foreground bounding box
3. Convert reference and target slides into grayscale representations
4. Perform coarse rotation search from `0` to `359` degrees with step `10`
5. For each rotation, search translation by 2D convolution
6. Around the best coarse angle, perform fine search with angle step `1`
7. Use the best angle and translation to crop and rotate the target slide

This is a rigid registration framework, so it is most appropriate when:

- section deformation is limited
- inter-slice morphology is still similar
- stain differences are strong but global tissue shape is preserved

## 6. Stage 2: Patch Cropping

After automatic registration is satisfactory, run:

```bash
python 02_crop_patches.py \
  --start_idx 1 \
  --end_idx 1 \
  --wsi_dir /path/to/output/case_0001_result/cropped_wsi \
  --mask_dir /path/to/output/case_0001_result/mask_npy \
  --save_patch_dir /path/to/output/case_0001_result/patches
```

### 6.1 Important Arguments

- `--wsi_dir`
  - Folder containing registered PNGs from `01_pair.py`
- `--mask_dir`
  - Folder containing `.npy` masks from `01_pair.py`
- `--save_patch_dir`
  - Folder where patches will be written
- `--save_dir_cls`
  - Stain list, default `["HE", "MAS", "PAS", "PASM"]`
- `--mask_size`
  - Must match the mask size used in `01_pair.py`
- `--crop_patch_size`
  - Patch size, default `256`
- `--crop_overlap`
  - Overlap between adjacent patches
- `--start_idx`, `--end_idx`
  - Case index range

### 6.2 Patch Naming

Patch names are saved as:

```text
{cls}_{wsi_idx:04d}_Y{row:03d}_X{col:03d}.jpg
```

For example:

```text
HE_0001_Y000_X000.jpg
PAS_0001_Y005_X012.jpg
```

### 6.3 Output Structure

```text
patches/
|-- HE/
|-- MAS/
|-- PAS/
|-- PASM/
```

The mask from the reference slide is used to skip background regions. Only mask-positive areas are cropped.

### 6.4 Important Note About `mask_size` and `patch_size`

The patch cropping code indexes the mask with:

```python
mask[j // patch_size][i // patch_size]
```

In practice, the safest setting is:

- `mask_size == crop_patch_size`

For example:

- `mask_size = 256`
- `crop_patch_size = 256`

This matches the default design of the code and avoids indexing mismatch between the saved mask grid and the cropped patch grid.

## 7. Stage 3: Manual Refinement

`03_refine.py` is an interactive post-processing tool for cases where automatic registration is not good enough.

### 7.1 Before Running

This script currently contains several hardcoded settings and is not yet a general command-line tool. You need to edit the script first:

- `start_idx`
- `end_idx`
- `png_dir_base`
- `tn_dir_base`
- `mask_dir_base`
- `save_refine_png_dir_base`
- `save_refine_tn_dir_base`
- `vipshome`
- `folder_list`

In particular, `folder_list` is currently written for a breast IHC-style setting:

```python
folder_list = [
    ("HE_ER", "ER"),
    ("HE_HER2", "HER2"),
    ("HE_KI67", "KI67"),
    ("HE_PGR", "PGR"),
]
```

So if you are working with another stain set, you must replace it with your own pairing names.

### 7.2 What It Does

For each case:

1. Load the reference and target thumbnails
2. Show grid-overlaid comparisons
3. Let the user input `move_x` and `move_y`
4. Apply the same shift to the full-resolution cropped PNG
5. Save refined thumbnails and refined large PNGs

### 7.3 Coordinate Meaning

From the code comments:

- one manual step corresponds to `64` pixels in the cropped WSI
- that equals `4` pixels in the thumbnail
- positive `move_x` means shift right
- positive `move_y` means shift down

This tool adjusts translation only. It does not re-estimate rotation.

## 8. Common Usage Example

Suppose one case folder contains:

```text
example_case/
|-- 1_HE.tif
|-- 1_MAS.tif
|-- 1_PAS.tif
|-- 1_PASM.tif
```

### Step 1: Automatic registration

```bash
python 01_pair.py \
  --dataset_dir /data/example_case \
  --processed_dir /data/example_case_processed \
  --prefix 1 \
  --reference HE \
  --mask_size 256 \
  --mask_thr 0.25 \
  --downsample 32 \
  --foreground_model_path foreground_detect.pt
```

### Step 2: Check results

Open and inspect:

- `thumbnail/1_whole.png`
- `thumbnail/1_whole_contrast.png`
- `thumbnail/1_HE_MAS.png`
- `thumbnail/1_HE_PAS.png`
- `thumbnail/1_HE_PASM.png`

If alignment looks poor for a target slide, use `03_refine.py`.

### Step 3: Crop paired patches

```bash
python 02_crop_patches.py \
  --start_idx 1 \
  --end_idx 1 \
  --wsi_dir /data/example_case_processed/cropped_wsi \
  --mask_dir /data/example_case_processed/mask_npy \
  --save_patch_dir /data/example_case_processed/patches \
  --mask_size 256 \
  --crop_patch_size 256 \
  --crop_overlap 0
```

## 9. Practical Notes and Caveats

Please pay attention to the following details when using this repository:

- The code is currently closer to a research prototype than a packaged software release
- Several scripts still contain dataset-specific default paths
- GPU is assumed in the current implementation
- `03_refine.py` is partially hardcoded for a specific stain configuration
- The scripts use `matplotlib` image display in multiple places, which may pop up figures during execution
- The saved intermediate cache is written to `./temp`
- The foreground mask is derived only from the reference slide and then reused across all target slides
- The registration is rigid only; severe tissue deformation may require manual refinement or a deformable method

## 10. Recommended Workflow for New Users

If you are using the repository for the first time, this order is recommended:

1. Prepare one small case first instead of batch-processing many cases
2. Verify that `foreground_detect.pt` can be loaded correctly
3. Run `01_pair.py` on a single case
4. Check `thumbnail/*contrast.png` and `thumbnail/*whole*.png`
5. If needed, correct difficult cases with `03_refine.py`
6. Only after visual confirmation, run `02_crop_patches.py`
7. Keep `mask_size` equal to `crop_patch_size` unless you have verified another setting carefully

## 11. Citation

If you use this codebase in your research, please consider citing the following papers:

```bibtex
@article{liu2025star,
  title={STAR: A Fast and Robust Rigid Registration Framework for Serial Histopathological Images},
  author={Liu, Zeyu and Ding, Shengwei},
  journal={arXiv preprint arXiv:2509.02952},
  year={2025}
}
```

```bibtex
@article{liu2025stainexpert,
  title={StainExpert: A Unified Multi-Expert Diffusion Framework for Multi-Target Pathological Stain Translation},
  author={Liu, Zeyu and He, Yufang and Zhang, Tianyi and Ma, Chenbin and Song, Fan and Wu, Huijie and Cai, Ruxin and Guo, Haoran and Zhang, Haonan and Wen, Bo and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```

## 12. Future Improvements

For better usability, the following would be valuable next steps:

- convert `03_refine.py` into a proper command-line script
- add a `requirements.txt` or `environment.yml`
- add CPU/GPU device selection
- add batch-level dataset organization examples
- add quantitative evaluation scripts
- support deformable registration as an optional refinement stage

