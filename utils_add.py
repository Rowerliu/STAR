import time
import os
import cv2
import torch
import tifffile
import numpy as np
import utils

from itertools import product
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
show_img_flag = True


def check_and_create_folder(path):
    # 检查路径是否存在
    if not os.path.exists(path):
        try:
            # 递归创建文件夹
            os.makedirs(path)
            print(f"文件夹 {path} 创建成功")
        except OSError as e:
            print(f"创建文件夹 {path} 失败：{str(e)}")
    else:
        print(f"文件夹 {path} 已存在")


def rotate_img(pic, theta):
    """
    旋转图像并返回旋转后的结果。
    如果图像尺寸过大（超过指定阈值），先缩放图像，以确保图像尺寸均小于 SHRT_MAX。

    Args:
        pic : numpy.ndarray，输入图像
        theta : float，旋转角度（单位：度）
    """
    # 定义一个安全阈值，低于 SHRT_MAX (通常32767) 以避免内部处理时尺寸超限
    MAX_DIM = 32000  # 可根据实际情况调整，比如30000或更低

    # 原始图像尺寸
    orig_h, orig_w = pic.shape[:2]

    # 记录缩放因子，默认不缩放
    scale_factor = 1.0

    # 如果图像任一边长超过安全阈值，则按比例缩小图像
    if orig_h >= MAX_DIM or orig_w >= MAX_DIM:
        scale_factor = min(MAX_DIM / orig_h, MAX_DIM / orig_w)
        new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
        print(f"图像尺寸过大，缩放因子: {scale_factor:.3f}, 新尺寸: ({new_w}, {new_h})")
        pic = cv2.resize(pic, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 旋转模板的中心点坐标 c，通过将 he_template 的宽度和高度各自除以 2 来计算。
    c = pic.shape[1] // 2, pic.shape[0] // 2
    pic = pic.astype('uint8')

    # 计算给定中心点 c、旋转角度 theta 和缩放因子为 1.0 的旋转矩阵 M。 维度为 [2, 3]
    # 应用旋转矩阵 M 到 he_template。he_template.shape[1] 表示图像宽度，he_template.shape[0] 表示图像高度
    # rotated是旋转后的图像
    M = cv2.getRotationMatrix2D(c, theta, 1.0)
    rotated = cv2.warpAffine(
        pic, M, (pic.shape[1], pic.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # 白色填充
    )

    # 如果进行了缩放，则将旋转后的图像恢复到原始尺寸
    if scale_factor < 1.0:
        rotated = cv2.resize(rotated, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    return rotated


# 连接两个图像
def concatenate_images(output_path, image1=None, image2=None, image1_path="", image2_path="", ):
    # 打开两个图像
    image1 = Image.open(image1_path) if image1 is None else Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.open(image2_path) if image2 is None else Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    # 获取两个图像的尺寸
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 计算新图像的尺寸
    new_width = width1 + width2
    new_height = max(height1, height2)

    # 创建一个新的画布，大小为两个图像拼接后的尺寸
    new_image = Image.new('RGB', (new_width, new_height))

    # 在新画布上粘贴图像
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))

    # 保存拼接后的图像
    new_image.save(output_path)
    print("\t两个图像已成功拼接，保存为:", output_path)


# 连接多个图像
def concatenate_many_images(output_path, image1=None, images=None, image1_path="", images_path=[]):
    # 打开两个图像
    image1 = Image.open(image1_path) if image1 is None else Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    if images is None:
        images = []
        for image_path in images_path:
            images.append(Image.open(image_path))

    # 获取两个图像的尺寸
    width1, height1 = image1.size
    new_width = width1
    new_height = height1
    widths = [width1]
    for image in images:
        width, height = image.size
        widths.append(width + widths[-1])
        new_width = new_width + width
        new_height = max(new_height, height)

    # 创建一个新的画布，大小为两个图像拼接后的尺寸
    new_image = Image.new('RGB', (new_width, new_height))

    # 在新画布上粘贴图像
    new_image.paste(image1, (0, 0))
    for i, image in enumerate(images):
        new_image.paste(image, (widths[i], 0))

    # 保存拼接后的图像
    new_image.save(output_path)
    print("\t两个图像已成功拼接，保存为:", output_path)


def downsample_mask(mask_32, downsample_factor, abandon_threshold=0.8):
    a, b = mask_32.shape
    # 加1是为了映射 边缘部分，边缘部分一定抛弃，置为0
    new_mask = np.zeros((a // downsample_factor + 1, b // downsample_factor + 1), dtype=int)

    for i in range(a // downsample_factor):
        for j in range(b // downsample_factor):
            region = mask_32[i * downsample_factor:(i + 1) * downsample_factor,
                     j * downsample_factor:(j + 1) * downsample_factor]
            ones_count = np.count_nonzero(region == 1)
            if ones_count / (downsample_factor * downsample_factor) > abandon_threshold:
                new_mask[i, j] = 1
    return new_mask


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=5, gap_length=5):
    # 计算总长度
    dist = ((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2) ** 0.5
    dashes = int(dist // (dash_length + gap_length))
    for i in range(dashes):
        start_frac = (i * (dash_length + gap_length)) / dist
        end_frac = ((i * (dash_length + gap_length)) + dash_length) / dist
        start_point = (
            int(pt1[0] + (pt2[0] - pt1[0]) * start_frac),
            int(pt1[1] + (pt2[1] - pt1[1]) * start_frac),
        )
        end_point = (
            int(pt1[0] + (pt2[0] - pt1[0]) * end_frac),
            int(pt1[1] + (pt2[1] - pt1[1]) * end_frac),
        )
        cv2.line(img, start_point, end_point, color, thickness)


def add_divider(image_path, stride, new_image_path=""):
    # 读取图像
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # 分割线参数
    line_color = (0, 0, 0)  # 黑色
    line_thickness = 1
    dash_length = 4
    gap_length = 4

    # 绘制垂直虚线
    for x in range(stride, width, stride):
        draw_dashed_line(img, (x, 0), (x, height), line_color, line_thickness, dash_length, gap_length)

    # 绘制水平虚线
    for y in range(stride, height, stride):
        draw_dashed_line(img, (0, y), (width, y), line_color, line_thickness, dash_length, gap_length)

    # 保存图像
    if new_image_path != "":
        cv2.imwrite(new_image_path, img)

    return img


def add_divider_old(image_path, stride, new_image_path=""):
    # stride表示每多少像素加一条分割线
    img = cv2.imread(image_path)
    # if img is None:
    #     raise ValueError(f"Failed to read image from {image_path}")
    # # Convert BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape

    # 绘制垂直分割线
    for x in range(stride, width, stride):
        cv2.line(img, (x, 0), (x, height), (0, 0, 255), 2)  # 使用红色分割线，线宽为2
        if abs(x - width / 2) < stride:
            cv2.line(img, (x, 0), (x, height), (0, 255, 0), 2)

    # 绘制水平分割线
    for y in range(stride, height, stride):
        cv2.line(img, (0, y), (width, y), (0, 0, 255), 2)  # 使用红色分割线，线宽为2
        if abs(y - height / 2) < stride:
            cv2.line(img, (0, y), (width, y), (0, 255, 0), 2)

    # 保存图像
    if new_image_path != "":
        cv2.imwrite(new_image_path, img)
        # print("\t\t分割线已成功添加，并保存为:", new_image_path)

    return img


def apply_mask(image, mask, stride):
    '''
    Modifying an image using information from a mask

    :param image: 16倍降采样的图像
    :param mask: 32倍降采样的mask
    :param stride: mask图像中每一像素对应image中像素的个数
    :return: 处理后的图像，将背景全部置为黑色
    '''
    # 将Image对象转换为NumPy数组
    img_np = np.array(image)

    # 将指定区域的像素值置为黑色
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0 and stride * i < img_np.shape[0] and stride * j < img_np.shape[1]:
                img_np[stride * i: min(stride * (i + 1), img_np.shape[0]),
                stride * j: min(stride * (j + 1), img_np.shape[1])] \
                    = (0, 0, 0)

    # 创建新的Image对象
    new_image = Image.fromarray(np.uint8(img_np))
    return new_image


def fix_image(image, real_x, real_y):
    '''
    修复生成出的ihc的png图像，只有在real_X或者real_y为负时生效

    :param image:
    :param real_x: 在ihc原图中裁剪时左上角点的x坐标
    :param real_y: 在ihc原图中裁剪时左上角点的y坐标
    :return:
    '''

    expandx = abs(real_x) if real_x < 0 else 0
    expandy = abs(real_y) if real_y < 0 else 0
    if expandx == 0 and expandy == 0:
        return image

    # 创建新的图像，然后将原图像粘贴到新图像中。从而实现对左侧和上侧的黑色像素填充
    new_width = image.width + expandx
    new_height = image.height + expandy
    expanded_image = Image.new("RGB", (new_width, new_height), "white")
    expanded_image.paste(image, (expandx, expandy))
    return expanded_image


def save_trs(trs_path, prefix, place_inf, down_sample, box, thumbnail_dir, wsi_dir):
    # ! 代码中的down_sample为32，但是此处保存的缩略图为了好看是缩放的16倍
    start_time = time.time()
    pic_name = trs_path.split(os.sep)[-1].split('.')[0]
    print(f"\t正在保存IHC: {prefix}_{pic_name}")

    # Prepare the image (ihc)
    ext = os.path.splitext(trs_path)[1].lower()
    if ext == '.tif' or ext == '.tiff':
        wsi_array = tifffile.imread(trs_path)
    else:
        wsi_array = cv2.imread(trs_path)
        if wsi_array is None:
            raise ValueError(f"Failed to read image from {trs_path}")
        # Convert BGR to RGB
        wsi_array = cv2.cvtColor(wsi_array, cv2.COLOR_BGR2RGB)
    # 如果 filtered_patches 等后续函数依赖 PIL 接口，则转换为 PIL Image
    wsi = Image.fromarray(wsi_array)

    w, h = wsi.size
    wsi_ori = wsi.copy()
    wsi.thumbnail((w // (down_sample / 2), h // (down_sample / 2)))
    wsi_tn = wsi

    # 确定要裁剪的区域
    # 注意b-a对应的是高！！！
    a, b, c, d = box
    ori_height = (b - a) * down_sample
    ori_width = (d - c) * down_sample
    y_, x_, theta = place_inf  # 表示ihc横纵坐标，HE图像的旋转角度
    real_y = y_ * down_sample
    real_x = x_ * down_sample

    # 生成缩略图
    cropped_thumbnail = Image.fromarray(
        np.array(wsi_tn)[max(2 * y_, 0): 2 * (y_ + b - a), max(2 * x_, 0): 2 * (x_ + d - c)])
    utils.show_image(cropped_thumbnail, "填充像素前")
    cropped_thumbnail = fix_image(cropped_thumbnail, 2 * x_, 2 * y_)
    utils.show_image(cropped_thumbnail, "填充像素后")
    cropped_thumbnail.save(os.path.join(thumbnail_dir, f"{pic_name}.png"))

    # 生成原图大小的png  。这有一个前提要求，那就是原始图片的MPP一致，此数据集保证了
    # 这里可能会出现一个bug，那就是目标区间超出了范围，出现这个问题就直接不管了
    if real_y + ori_height >= h or real_x + ori_width >= w:
        print("\t此图像到处时发现范围超出了原图大小")

    ori_image = Image.fromarray(
        np.array(wsi_ori)[max(real_y, 0):real_y + ori_height, max(real_x, 0): real_x + ori_width])
    ori_image = fix_image(ori_image, real_x, real_y)  # 可能出现real_x或者real_y<0的情况
    ori_image.save(os.path.join(wsi_dir, f"{prefix}_{pic_name}.png"))

    end_time = time.time()
    print(f"\t\t已保存ihc图像，用时：{end_time - start_time:.2f}秒")
    return os.path.join(thumbnail_dir, f"{prefix}_{pic_name}.png")


def save_trs(he_path, prefix, place_inf, down_sample, box, mask, thumbnail_dir, wsi_dir, mask_dir):
    # 代码中的down_sample为32，但是此处保存的缩略图为了好看是缩放的16倍
    start_time = time.time()
    pic_name = he_path.split(os.sep)[-1].split('.')[0]
    print(f"\t正在保存HE图: {prefix}_{pic_name}")

    # Prepare the image (he)
    ext = os.path.splitext(he_path)[1].lower()
    if ext == '.tif' or ext == '.tiff':
        wsi_array = tifffile.imread(he_path)
    else:
        wsi_array = cv2.imread(he_path)
        if wsi_array is None:
            raise ValueError(f"Failed to read image from {he_path}")
        # Convert BGR to RGB
        wsi_array = cv2.cvtColor(wsi_array, cv2.COLOR_BGR2RGB)
    # 如果 filtered_patches 等后续函数依赖 PIL 接口，则转换为 PIL Image
    wsi = Image.fromarray(wsi_array)

    w, h = wsi.size
    wsi_ori = wsi.copy()
    wsi.thumbnail((w // (down_sample / 2), h // (down_sample / 2)))
    wsi_tn = wsi

    # 确定要裁剪的区域
    a, b, c, d = box
    width_on_wsi = (b - a) * down_sample
    height_on_wsi = (d - c) * down_sample
    x_, y_, theta = place_inf  # 表示ihc横纵坐标，HE图像的旋转角度
    real_x = a * down_sample
    real_y = c * down_sample
    print(f"\t\t旋转了{theta}度")

    # 生成mask信息
    k = np.ones((7, 7))
    mask = cv2.dilate(mask, k, 1)
    mask = rotate_img(mask, theta)
    utils.show_image(mask, "旋转后的mask")

    # 降采样并保存mask
    mask1024 = downsample_mask(mask, 32)  # 本来就是32缩略图，再降采样32，总1024倍
    np.save(os.path.join(mask_dir, f"{prefix}_{pic_name.split('_')[0]}_1024.npy"), mask1024)
    utils.show_image(mask1024, "降采样1024倍的mask")

    # 保存旋转后的图像
    rotated_tb = Image.fromarray(rotate_img(np.array(wsi_tn)[a * 2:b * 2, c * 2:d * 2], theta))
    rotated_tb.save(os.path.join(thumbnail_dir, f"{prefix}_{pic_name}.png"))

    rotated_ori = Image.fromarray(
        rotate_img(np.array(wsi_ori)[real_x:real_x + width_on_wsi, real_y:real_y + height_on_wsi], theta))
    rotated_ori.save(os.path.join(wsi_dir, f"{prefix}_{pic_name}.png"))

    # 保存加上mask信息的图像
    mask32_tn = apply_mask(rotated_tb, mask, 2)
    mask32_tn.save(os.path.join(thumbnail_dir, f"{prefix}_{pic_name}_mask32.png"))
    utils.show_image(mask32_tn, "使用mask后的图像")

    mask1024_tn = apply_mask(rotated_tb, mask1024, 64)
    mask1024_tn.save(os.path.join(thumbnail_dir, f"{prefix}_{pic_name}_mask1024.png"))
    utils.show_image(mask1024_tn, "使用1024倍mask的图像")

    add_divider(image_path=os.path.join(thumbnail_dir, f"{prefix}_{pic_name}_mask1024.png"),
                stride=64,
                new_image_path=os.path.join(thumbnail_dir, f"{prefix}_{pic_name}_mask1024_divide.png"))

    end_time = time.time()
    print(f"\t\t已保存he图像，用时：{end_time - start_time:.2f}秒")

    return os.path.join(thumbnail_dir, f"{prefix}_{pic_name}.png"), os.path.join(thumbnail_dir,
                                                                        f"{prefix}_{pic_name}_mask1024_divide.png"), mask1024


def save_trs_rotate(trs_path, prefix, place_inf, down_sample, mask_size, box, thumbnail_dir, wsi_dir):
    # ! 代码中的down_sample为32，但是此处保存的缩略图为了好看是缩放的16倍
    start_time = time.time()
    pic_name = trs_path.split(os.sep)[-1].split('.')[0]
    print(f"\tSaving transform image: {prefix}_{pic_name}")

    # Prepare the image (ihc)
    ext = os.path.splitext(trs_path)[1].lower()
    if ext == '.tif' or ext == '.tiff':
        wsi_array = tifffile.imread(trs_path)
    else:
        wsi_array = cv2.imread(trs_path)
        if wsi_array is None:
            raise ValueError(f"Failed to read image from {trs_path}")
        # Convert BGR to RGB
        wsi_array = cv2.cvtColor(wsi_array, cv2.COLOR_BGR2RGB)
    # 如果 filtered_patches 等后续函数依赖 PIL 接口，则转换为 PIL Image
    wsi = Image.fromarray(wsi_array)

    w, h = wsi.size
    wsi_ori = wsi.copy()
    wsi.thumbnail((w // (down_sample / 2), h // (down_sample / 2)))
    wsi_tn = wsi

    # 确定要裁剪的区域
    # 注意b-a对应的是高！！！
    a, b, c, d = box
    ori_height = (b - a) * down_sample
    ori_width = (d - c) * down_sample
    y_, x_, theta = place_inf  # 表示ihc横纵坐标，HE图像的旋转角度
    print(f"\t\t旋转了{theta}度，需要逆向旋转{theta}度")
    real_y = y_ * down_sample
    real_x = x_ * down_sample

    # 生成缩略图
    cropped_thumbnail = Image.fromarray(
        np.array(wsi_tn)[max(2 * y_, 0): 2 * (y_ + b - a), max(2 * x_, 0): 2 * (x_ + d - c)])
    utils.show_image(cropped_thumbnail, "填充像素前")
    cropped_thumbnail = fix_image(cropped_thumbnail, 2 * x_, 2 * y_)
    utils.show_image(cropped_thumbnail, "填充像素后")
    # 旋转-theta
    cropped_thumbnail = Image.fromarray(
        rotate_img(np.array(cropped_thumbnail), -theta)
    )
    cropped_thumbnail.save(os.path.join(thumbnail_dir, f"{prefix}_{pic_name}.png"))

    # 生成原图大小的png。这有一个前提要求，那就是原始图片的MPP一致，此数据集保证了
    # 这里可能会出现一个bug，那就是目标区间超出了范围，出现这个问题就直接不管了
    if real_y + ori_height >= h or real_x + ori_width >= w:
        print("\t此图像到处时发现范围超出了原图大小")

    ori_image = Image.fromarray(
        np.array(wsi_ori)[max(real_y, 0):real_y + ori_height, max(real_x, 0): real_x + ori_width])
    ori_image = fix_image(ori_image, real_x, real_y)  # 可能出现real_x或者real_y<0的情况
    # 旋转-theta
    ori_image = Image.fromarray(
        rotate_img(np.array(ori_image), -theta)
    )
    ori_image.save(os.path.join(wsi_dir, f"{prefix}_{pic_name}.png"))

    end_time = time.time()
    print(f"\t\t已保存ihc图像，用时：{end_time - start_time:.2f}秒")
    return os.path.join(thumbnail_dir, f"{prefix}_{pic_name}.png")


def save_ref_no_rotate(he_path, prefix, down_sample, mask_size, mask_thr, box, mask, thumbnail_dir, wsi_dir, mask_dir):
    # 代码中的down_sample为32，但是此处保存的缩略图为了好看是缩放的16倍
    start_time = time.time()
    pic_name = he_path.split(os.sep)[-1].split('.')[0]
    print(f"\t正在保存HE图: {prefix}_{pic_name}")

    # Prepare the image (he)
    ext = os.path.splitext(he_path)[1].lower()
    if ext == '.tif' or ext == '.tiff':
        wsi_array = tifffile.imread(he_path)
    else:
        wsi_array = cv2.imread(he_path)
        if wsi_array is None:
            raise ValueError(f"Failed to read image from {he_path}")
        # Convert BGR to RGB
        wsi_array = cv2.cvtColor(wsi_array, cv2.COLOR_BGR2RGB)
    # 如果 filtered_patches 等后续函数依赖 PIL 接口，则转换为 PIL Image
    wsi = Image.fromarray(wsi_array)

    w, h = wsi.size
    wsi_ori = wsi.copy()
    wsi.thumbnail((w // (down_sample / 2), h // (down_sample / 2)))
    wsi_tn = wsi

    # 确定要裁剪的区域
    a, b, c, d = box
    width_on_wsi = (b - a) * down_sample
    height_on_wsi = (d - c) * down_sample
    # x_, y_, theta = place_inf  # 表示ihc横纵坐标，HE图像的旋转角度
    real_x = a * down_sample
    real_y = c * down_sample
    # print(f"\t\t旋转了{theta}度")

    # 生成mask信息
    k = np.ones((7, 7))
    mask = cv2.dilate(mask, k, 1)
    # mask = rotate_img(mask, theta)
    utils.show_image(mask, "mask")

    stride = int(mask_size // 32)
    # 降采样并保存mask
    mask = downsample_mask(mask, stride, mask_thr)  # 本来就是32缩略图，再降采样32，总1024倍
    np.save(os.path.join(mask_dir, f"{prefix}_{pic_name.split('_')[0]}_{mask_size}.npy"), mask)
    utils.show_image(mask, f"降采样{stride}倍的mask")

    # 保存图像
    tb = Image.fromarray(np.array(wsi_tn)[a * 2:b * 2, c * 2:d * 2])
    tb.save(os.path.join(thumbnail_dir, f"{prefix}_{pic_name}.png"))

    ori = Image.fromarray(
        np.array(wsi_ori)[real_x:real_x + width_on_wsi, real_y:real_y + height_on_wsi])
    ori.save(os.path.join(wsi_dir, f"{prefix}_{pic_name}.png"))

    # 保存加上mask信息的图像
    stride = stride*2
    mask_tn = apply_mask(tb, mask, stride)
    mask_tn.save(os.path.join(thumbnail_dir, f"{prefix}_{pic_name}_mask{mask_size}.png"))
    utils.show_image(mask_tn, f"使用{mask_size}倍mask的图像")

    add_divider(image_path=os.path.join(thumbnail_dir, f"{prefix}_{pic_name}_mask{mask_size}.png"), stride=stride,
                new_image_path=os.path.join(thumbnail_dir, f"{prefix}_{pic_name}_mask{mask_size}_divide.png"))

    end_time = time.time()
    print(f"\t\t已保存he图像，用时：{end_time - start_time:.2f}秒")

    return os.path.join(thumbnail_dir, f"{prefix}_{pic_name}.png"), os.path.join(thumbnail_dir, f"{prefix}_{pic_name}_mask{mask_size}_divide.png"), mask


def scale_around_point(image, cx, cy, sx, sy):
    """以指定点为中心缩放图像，保持原图尺寸（缩放后填充或裁剪）"""
    h, w = image.shape[:2]
    M = np.array([
        [sx, 0, cx * (1 - sx)],  # 平移量确保中心点固定
        [0, sy, cy * (1 - sy)]
    ], dtype=np.float32)
    scaled_image = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return scaled_image


def scale_cycle(image, sx_list, sy_list, cx=None, cy=None):
    """遍历所有sx和sy的组合，返回字典：键为(sx, sy)，值为缩放后的图像"""
    # 默认使用图像中心
    if cx is None or cy is None:
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
    # 生成所有sx和sy的笛卡尔积组合
    scale_pairs = list(product(sx_list, sy_list))
    # 存储结果字典
    result_dict = {}
    for sx, sy in scale_pairs:
        scaled_image = scale_around_point(image, cx, cy, sx, sy)
        result_dict[(sx, sy)] = scaled_image
    return result_dict


def max_dot_scale(he, scale_dict):
    """
    基于图像点积的多尺度匹配（要求he和缩放后的ihc尺寸相同）
    Args:
        he: 输入图像 [H, W]（Tensor）
        scale_dict: 缩放后的ihc图像字典，键为(sx, sy)，值为图像 [H, W]（Tensor或numpy）
        stride: 保留参数（实际未使用，仅为兼容接口）
    Returns:
        (max_value, sx, sy)
    """
    max_value = -float('inf')

    # 确保he为Tensor格式
    if not isinstance(he, torch.Tensor):
        he = torch.from_numpy(he).float()

    # 遍历所有缩放组合
    for (sx, sy), scaled_image in scale_dict.items():
        # 将ihc图像转为Tensor
        if not isinstance(scaled_image, torch.Tensor):
            ihc_scaled = torch.from_numpy(scaled_image).float()
        else:
            ihc_scaled = scaled_image

        # 计算点积（要求尺寸完全一致）
        assert he.shape == ihc_scaled.shape, "HE和IHC缩放图尺寸必须相同"
        he_mean = he.mean(dim=(0, 1), keepdim=True)
        he_std = he.std(dim=(0, 1), keepdim=True)
        he = (he - he_mean) / he_std

        ihc_mean = ihc_scaled.mean(dim=(0, 1), keepdim=True)
        ihc_std = ihc_scaled.std(dim=(0, 1), keepdim=True)
        ihc = (ihc_scaled - ihc_mean) / ihc_std

        dot_product = torch.sum(he * ihc) / (torch.norm(he) * torch.norm(ihc))
        # 更新最大值记录
        if dot_product > max_value:
            max_value = dot_product
            reg_data = (scaled_image, sx, sy)

    # 返回结果（x,y无意义，固定为0）
    return reg_data


def max_conv_scale(he, scale_dict, stride):
    """
    基于缩放因子字典的卷积最大值搜索
    Args:
        he: 输入图像 (Tensor, shape: [C, H, W])
        scale_dict: 缩放图像字典，键为(sx, sy)，值为缩放后的图像 (Tensor)
        stride: 卷积步长
    Returns:
        reg_data: (max_value, sx, sy, x, y)
    """
    max_value = -9999999
    reg_data = None

    # 遍历所有缩放因子组合
    for (sx, sy), scaled_image in scale_dict.items():
        # 将缩放后的图像转换为Tensor作为卷积核
        weight = torch.from_numpy(scaled_image).float()  # 假设输入是numpy数组
        weight = weight.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, kH, kW]

        # 处理输入图像维度
        input_ = he.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

        # 动态调整kernel尺寸（如果kernel大于input）
        input_h, input_w = input_.shape[-2:]
        kernel_h, kernel_w = weight.shape[-2:]

        # 如果kernel尺寸超过input，则下采样kernel
        if kernel_h > input_h or kernel_w > input_w:
            scale_factor_h = (input_h - 1) / kernel_h
            scale_factor_w = (input_w - 1) / kernel_w
            scale_factor = min(scale_factor_h, scale_factor_w)
            new_h = max(1, int(kernel_h * scale_factor))
            new_w = max(1, int(kernel_w * scale_factor))

            weight = torch.nn.functional.interpolate(
                weight,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )

        # 执行卷积
        out = torch.nn.functional.conv2d(
            input_,
            weight,
            stride=stride,
            padding=0  # 根据需求调整padding
        )

        # 更新最大值记录
        current_max = out.max().item()
        if current_max > max_value:
            max_value = current_max
            reg_data = (scaled_image, sx, sy)

    return reg_data


def register_scale(he, ihc):
    # # 读取图像并转换为灰度
    # he = cv2.cvtColor(cv2.imread(he_dir), cv2.COLOR_BGR2GRAY)
    # ihc = cv2.cvtColor(cv2.imread(ihc_dir), cv2.COLOR_BGR2GRAY)

    # 粗配准阶段 -------------------------------------------------
    sx_coarse = [0.5, 0.7, 1.0, 1.3, 1.6]
    sy_coarse = [0.5, 0.7, 1.0, 1.3, 1.6]
    scale_dict_coarse = scale_cycle(ihc, sx_coarse, sy_coarse)

    # 执行粗搜索
    scaled_ihc, best_sx, best_sy = max_dot_scale(
        torch.from_numpy(he).float(),
        scale_dict_coarse
    )
    utils.show_image(scaled_ihc, 'scaled_ihc')

    # 细配准阶段 -------------------------------------------------
    # 生成精细缩放因子
    sx_fine = np.linspace(max(0.3, best_sx - 0.2), best_sx + 0.2, 5)
    sy_fine = np.linspace(max(0.3, best_sy - 0.2), best_sy + 0.2, 5)

    scale_dict_fine = scale_cycle(ihc, sx_fine, sy_fine)

    # 执行精细搜索
    scaled_ihc_fianl, sx_final, sy_final = max_dot_scale(
        torch.from_numpy(he).float(),
        scale_dict_fine
    )
    utils.show_image(scaled_ihc_fianl, 'scaled_ihc_final')

    return sx_final, sy_final
