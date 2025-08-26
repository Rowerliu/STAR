import numpy as np
import os
import argparse
from PIL import Image

import abandon_idx_list as shared_inf
import utils_add
import utils

from utils import foreground_detection_model
from utils import get_foreground
from utils import get_bbox, simple_bbox_crop
from utils import ref_conv_input
from utils import trs_conv_input
from utils import register


def get_args_parser():
    parser = argparse.ArgumentParser()
    # 要处理的wsi的编号起止
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=99)
    # 输出配准WSI
    parser.add_argument('--processed_dir', type=str, default=r"F:\13_Data\03_Pathology\z05_ANHIR\01_custom\registration\lunglobe\20250612")
    # 原始数据集
    parser.add_argument('--dataset_dir', type=str, default=r"F:\13_Data\03_Pathology\z05_ANHIR\00_original\lung-lobes_4\scale-100pc")
    # 原始数据集
    parser.add_argument('--prefix', type=str, default=r"4")  # e.g. (kidney_5), prefix = 5

    # 配准目标
    parser.add_argument('--reference', type=str, default="HE")
    # 前后景分割
    help_str = 'path for foreground detection model'
    parser.add_argument('--foreground_model_path', help=help_str, type=str, default=r'foreground_detect.pt', metavar='')
    parser.add_argument('--batch_size', help='batch_size', type=int, default=1, metavar='')
    parser.add_argument('--white_thr', help='Threshold for white pixel intensity', type=int, default=230, metavar='')
    parser.add_argument('--black_thr', help='Threshold for balck pixel intensity', type=int, default=20, metavar='')

    parser.add_argument('--mask_size', help='Threshold for mask ratio', type=int, default=256, metavar='')
    parser.add_argument('--mask_thr', help='Threshold for mask ratio', type=float, default=0.25, metavar='')
    parser.add_argument('--downsample', help='Downsample factor', type=int, default=32, metavar='')
    return parser.parse_args()


def check_finished(ref_path, trs_path, save_wsi_dir, mask_dir, mask_size):
    ref_name = ref_path.split(os.sep)[-1].split('.')[0]
    trs_name = trs_path.split(os.sep)[-1].split('.')[0]
    idx = ref_name.split('_')[0]
    if os.path.exists(os.path.join(save_wsi_dir, f"{ref_name}.png")) and \
            os.path.exists(os.path.join(save_wsi_dir, f"{trs_name}.png")) and \
            os.path.exists(os.path.join(mask_dir, f"{idx}_{mask_size}.npy")):
        return True
    return False


def create_all_dir_new(processed_dir):
    utils_add.check_and_create_folder(os.path.join("./temp"))
    utils_add.check_and_create_folder(os.path.join(processed_dir, "cropped_wsi"))
    utils_add.check_and_create_folder(os.path.join(processed_dir, "thumbnail"))
    utils_add.check_and_create_folder(os.path.join(processed_dir, "mask_npy"))


def get_directories(path):
    """返回指定路径下的所有目录"""
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


if __name__ == '__main__':
    args = get_args_parser()

    start_idx = args.start_idx
    end_idx = args.end_idx
    processed_dir = args.processed_dir
    save_wsi_dir = os.path.join(processed_dir, "cropped_wsi")
    dataset_dir = args.dataset_dir
    # ----------------------------
    batch_size = args.batch_size
    down_sample = args.downsample
    white_thr = args.white_thr
    black_thr = args.black_thr
    foreground_model_path = args.foreground_model_path

    # 其他基本信息
    model = foreground_detection_model(foreground_model_path)  # 加载获取前景的模型
    failed_list = []
    already_finished_list = []
    lack_tiff_list = []
    abandon_idx_list = shared_inf.abandon_idx_list

    # 建好保存结果的文件夹
    create_all_dir_new(processed_dir)

    img_names = os.listdir(dataset_dir)
    prefix = args.prefix
    ref_name = args.reference
    mask_size = args.mask_size
    mask_thr = args.mask_thr

    ref_path = ''
    trs_paths = []
    for img_name in img_names:
        if ref_name in img_name:
            ref_path = os.path.join(dataset_dir, img_name)
        else:
            trs_paths.append(os.path.join(dataset_dir, img_name))

    foreground_kwargs = {'batch_size': batch_size, 'white_thr': white_thr, 'black_thr': black_thr, 'downsample': down_sample}

    if os.path.exists(f"./temp/{prefix}_{ref_name}.npy"):
        foreground = np.load(f"./temp/{prefix}_{ref_name}.npy")
    else:
        foreground = get_foreground(model, ref_path, batch_size=batch_size, white_thr=white_thr,
                                    black_thr=black_thr, downsample=down_sample)
        np.save(f"./temp/{prefix}_{ref_name}.npy", foreground)

    utils.show_image(foreground, "原始的前景mask")

    # bbox表示裁剪后的图像在原图上的范围
    # cropped_mask, bbox = get_bbox(foreground)
    cropped_mask, bbox = simple_bbox_crop(foreground)
    utils.show_image(cropped_mask, "裁剪后的前景mask")

    # he图像要同步进行预处理，找到有组织的部分。 ihc图像只要读取就行了
    print("\t正在读取图像")
    ref_conv = ref_conv_input(ref_path, bbox, foreground, down_sample)

    # 获取相应的存储路径
    thumbnail_dir = os.path.join(processed_dir, "thumbnail")
    wsi_dir = os.path.join(processed_dir, "cropped_wsi")
    mask_dir = os.path.join(processed_dir, "mask_npy")

    utils_add.check_and_create_folder(thumbnail_dir)
    utils_add.check_and_create_folder(wsi_dir)
    utils_add.check_and_create_folder(mask_dir)

    ref_tn_path, ref_tn_mask_divide_path, mask = utils_add.save_ref_no_rotate(ref_path, prefix, down_sample,
                                                                            mask_size, mask_thr, bbox,
                                                                            cropped_mask,
                                                                            thumbnail_dir, wsi_dir,
                                                                            mask_dir)
    divide_trs_tn_paths = []
    divide_trs_mask_paths = []
    # 正式开始一个个处理
    for trs_path in trs_paths:
        trs_name = trs_path.split(os.sep)[-1].split('.')[0]
        divide_trs_tn_path = os.path.join(thumbnail_dir, f"{prefix}_{trs_name}_divide.png")
        divide_trs_tn_paths.append(divide_trs_tn_path)

        divide_trs_mask_path = os.path.join(thumbnail_dir, f"{prefix}_{trs_name}_mask{mask_size}_divide.png")
        divide_trs_mask_paths.append(divide_trs_mask_path)

        trs_conv = trs_conv_input(trs_path)

        # 配对两张图像，data里面是一个三元元组，(x, y, theta)
        print("\t正在配准")
        data = register(ref_conv, trs_conv)

        # 保存裁剪后的he和ihc图像，在保存ihc时也会根据情况修改mask文件
        # 涉及到了文件的读取
        trs_ori_path = os.path.join(wsi_dir, f"{prefix}_{trs_name}.png")
        trs_tn_path = utils_add.save_trs_rotate(trs_path, prefix, data, down_sample, mask_size, bbox, thumbnail_dir, wsi_dir)

        # 生成分割线
        stride = mask_size // 16
        _ = utils_add.add_divider(trs_tn_path, stride, divide_trs_tn_path)

        # 给IHC添加分割线和mask，同时和HE拼接，便于肉眼识别
        trs_img = Image.open(trs_tn_path)
        trs_mask = utils_add.apply_mask(trs_img, mask, stride)
        trs_mask_path = os.path.join(thumbnail_dir, f"{prefix}_{trs_name}_mask{mask_size}.png")
        trs_mask.save(trs_mask_path)

        trs_mask_divide = utils_add.add_divider(trs_mask_path, stride, divide_trs_mask_path)

    # 统一合成
    divide_ref_tn = utils_add.add_divider(ref_tn_path, stride, ref_tn_path[:-4] + '_divide.png')
    for i in range(len(divide_trs_tn_paths)):
        trs_path = trs_paths[i]
        trs_name = trs_path.split(os.sep)[-1].split('.')[0]
        # ihc_kind, _ = get_ihc_kind_and_he_file_name(ihc_path_list[i])
        utils_add.concatenate_images(os.path.join(thumbnail_dir, f"{prefix}_{ref_name}_{trs_name}.png"),
                                   image1=divide_ref_tn, image2=None,
                                   image2_path=divide_trs_tn_paths[i])
        utils_add.concatenate_images(output_path=os.path.join(thumbnail_dir, f"{prefix}_{ref_name}_{trs_name}_contrast.png"),
                                   image1=None, image2=None,
                                   image1_path=ref_tn_mask_divide_path,
                                   image2_path=divide_trs_mask_paths[i])

    utils_add.concatenate_many_images(os.path.join(thumbnail_dir, f"{prefix}_whole.png"),
                                    image1=divide_ref_tn, images=None,
                                    images_path=divide_trs_tn_paths)
    utils_add.concatenate_many_images(output_path=os.path.join(thumbnail_dir, f"{prefix}_whole_contrast.png"),
                                    image1=None, images=None,
                                    image1_path=ref_tn_mask_divide_path,
                                    images_path=divide_trs_mask_paths)
    # num_img_processed += 1

    print('done')

