import os
import glob
import argparse

import PIL
import numpy as np

from src_for_train_set import abandon_idx_list_train_set as shared_inf

# 这个要根据情况修改
vipshome = r'F:\DSW\solve_env_bug\vips-dev-8.14\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips


def get_args_parser():
    parser = argparse.ArgumentParser()
    # 要处理的wsi的编号起止
    parser.add_argument('--start_idx', type=int, default=1)
    parser.add_argument('--end_idx', type=int, default=5)
    # 当黑色像素占比超过此值时抛弃
    parser.add_argument('--black_pixel_ratio_threshold', type=float, default=0.001)
    # 保存切后的Multi stain patch的文件夹地址
    parser.add_argument('--save_patch_dir', type=str, default=r"F:\13_Data\03_Pathology\z05_ANHIR\01_custom\registration\kidney\20250612_v5\patches")
    parser.add_argument('--save_dir_cls', type=list, default=["HE", "MAS", "PAS", "PASM"])
    # WSI关键区域的png
    parser.add_argument('--wsi_dir', type=str, default=r"F:\13_Data\03_Pathology\z05_ANHIR\01_custom\registration\kidney\20250612_v5\cropped_wsi")
    parser.add_argument('--mask_dir', type=str, default=r"F:\13_Data\03_Pathology\z05_ANHIR\01_custom\registration\kidney\20250612_v5\mask_npy")
    parser.add_argument('--mask_size', help='size of mask unit', type=int, default=256, metavar='')
    parser.add_argument('--crop_patch_size', type=int, default=256)
    parser.add_argument('--crop_overlap', type=int, default=0)
    return parser.parse_args()


def count_black_prop(image):
    image = PIL.Image.fromarray(image.numpy())
    width, height = image.size

    RGBCounter = 0
    for i in range(width):
        for j in range(height):
            if (image.getpixel((i, j)) == (0, 0, 0)):
                RGBCounter += 1

    prop = RGBCounter / (width * height)
    return prop


def crop_image(image_path, patch_save_dir, patch_size, stride, mask, cls):
    '''
    Args:
        added_mask: 用于裁剪 IHC 时记录黑色区域位置，在裁剪 HE 时忽略这些区域

    Returns:
        新的 added_mask，用于后续 HE 裁剪的参考。
    '''

    image = pyvips.Image.new_from_file(image_path)
    wsi_idx = int(image_path.split(os.sep)[-1].split('.')[0].split('_')[0])
    width, height = image.width, image.height

    num = 0
    abandon_num = 0
    for j in range(0, height - patch_size + 1, stride):
        for i in range(0, width - patch_size + 1, stride):
            # 背景要跳过
            if mask[j // patch_size][i // patch_size] == 0:
                continue

            # 切割
            patch = image.crop(i, j, patch_size, patch_size)

            patch.write_to_file(
                os.path.join(patch_save_dir, f"{cls}_{wsi_idx:04d}_Y{j // stride:03d}_X{i // stride:03d}.jpg"))
            num += 1

    print(f"\t切了 {num} 个 {cls} 的 patch，抛弃了 {abandon_num} 个 patch")


if __name__ == '__main__':
    # 获取args的信息
    args = get_args_parser()

    black_pixel_ratio_threshold = args.black_pixel_ratio_threshold
    start_idx = args.start_idx
    end_idx = args.end_idx

    save_patch_dir_base = args.save_patch_dir
    wsi_dir = args.wsi_dir
    mask_dir = args.mask_dir
    mask_size = args.mask_size

    # 基础信息
    failed_list = []
    lack_png_list = []
    lack_mask_list = []
    abandon_idx_list = shared_inf.abandon_idx_list

    # 确定要处理的wsi，删去抛弃不处理的wsi。最后idx_list里面存的就是要处理wsi的idx
    idx_list = [i for i in range(start_idx, end_idx + 1)]
    for abandon_idx in abandon_idx_list:
        if abandon_idx in idx_list:
            idx_list.remove(abandon_idx)

    print("\n\n" + 5 * "-----------------------")
    print(f"要处理从{start_idx}到{end_idx}的wsi")
    print(f"抛弃不处理的idx_list为:{abandon_idx_list}")
    print(f"最终要处理的wsi_idx有{len(idx_list)}个")
    print(f"最终要处理的wsi_idx_list为{idx_list}")

    folder_list = args.save_dir_cls
    ref_name = folder_list[0]

    # 准备开始处理
    for idx in idx_list:
        print(f"\n开始处理{idx}号wsi")

        for cls in folder_list:
        # transform多类图片分别进行处理
            # 找出文件存储的路径
            print("\t----------------------------")
            save_patch_dir = os.path.join(save_patch_dir_base, cls)
            os.makedirs(save_patch_dir, exist_ok=True)

            # 检查是否已经处理过了
            if len(glob.glob(os.path.join(save_patch_dir, f"{idx}_{cls}*"))) > 0:
                print(f"\t{idx}号wsi已经处理过了")
                continue

            # 检查wsi是否存在
            wsi_path = os.path.join(wsi_dir, f"{idx}_{cls}.png")
            if not os.path.exists(wsi_path):
                print(f"\t{idx}号wsi，处理{cls}时，缺失png文件，跳过")
                failed_list.append(idx)
                lack_png_list.append(idx)
                continue

            # 检查mask文件是否存在
            mask = None
            mask_path = os.path.join(mask_dir, f"{idx}_{ref_name}_{mask_size}.npy")
            if not os.path.exists(mask_path):
                print(f"\t{idx}号wsi缺失mask信息，跳过")
                failed_list.append(idx)
                lack_png_list.append(idx)
                continue
            else:
                mask = np.load(os.path.join(mask_dir, f"{idx}_{ref_name}_{mask_size}.npy"))

            # 正式进行裁剪
            patch_size = args.crop_patch_size
            stride = patch_size - args.crop_overlap
            crop_image(wsi_path, save_patch_dir, patch_size, stride, mask, cls)


    # 最后输出放弃的和失败的
    file_amount = len(idx_list) * len(folder_list)
    print("\n\n" + 10 * "-------------")
    print(f"要处理从{start_idx}到{end_idx}的wsi")
    print(f"共要处理{file_amount}个文件，成功了{file_amount - len(failed_list)}个，失败了{len(failed_list)}个")
    print(f"失败的wsi的idx为：{failed_list}")
    print(f"缺少png的idx为：{lack_png_list}")
    print(f"缺少mask的idx为：{lack_mask_list}")
