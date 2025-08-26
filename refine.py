# 这个函数用于手工对齐HE和IHC
# 基本的逻辑是人肉眼尝试对其缩略图
# 这个文件就没有防御性编程了，仅仅实现最基本的功能
import glob
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import abandon_idx_list as shared_inf

# ! 这部分需要根据实际情况修改
vipshome = r'F:\DSW\solve_env_bug\vips-dev-8.14\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips

# 自行设置参数
start_idx = 20
end_idx = 25

png_dir_base = "..\\processed_train\\20250305\\cropped_wsi"
tn_dir_base = "..\\processed_train\\20250305\\thumbnail"
mask_dir_base = "..\\processed_train\\20250305\\mask_npy"
save_refine_png_dir_base = "..\\processed_train\\20250305\\refine\\cropped_wsi"
save_refine_tn_dir_base = "..\\processed_train\\20250305\\refine\\thumbnail"


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


def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def add_divider(image, stride, new_image_path=""):
    # stride表示每多少像素加一条分割线
    img = pil_to_cv2(image)
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

    return cv2_to_pil(img)


def add_mask_and_divide(image1, mask_1024):
    image1 = apply_mask(image1, mask_1024, 64)
    return add_divider(image1, 64)


def concatenate_image(image1, image2, flag_show=True):
    # 将两张图象拼起来展示出来，输入的都是PIL图像
    # 获取两个图像的尺寸
    width1, height1 = image1.size
    width2, height2 = image2.size
    new_width = width1 + width2
    assert height1 == height2
    new_height = max(height1, height2)

    # 创建一个新的画布，大小为两个图像拼接后的尺寸
    new_image = Image.new('RGB', (new_width, new_height))

    # 在新画布上粘贴图像
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))

    # 保存拼接后的图像
    # new_image = add_vertical_line(new_image)
    if flag_show:
        new_image.show()
    return new_image


def move_pic(image, move_x, move_y):
    width, height = image.size
    new_image = Image.new('RGB', (width, height), color=(0, 0, 0))

    for x in range(width):
        for y in range(height):
            new_x = x
            new_y = y
            new_x -= move_x
            new_y -= move_y
            if new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
                new_image.putpixel((x, y), image.getpixel((new_x, new_y)))

    return new_image


def move_and_save_big_png(vips_image_path, move_x, move_y, save_refine_png_dir):
    # 仅仅用于处理原图大小的png文件，因为其太大了PIL没法处理
    vips_image = pyvips.Image.new_from_file(vips_image_path)
    width, height = vips_image.width, vips_image.height

    left = 0 if move_x > 0 else -1 * move_x
    top = 0 if move_y > 0 else -1 * move_y
    crop_width = width - abs(move_x)
    crop_height = height - abs(move_y)
    vips_image = vips_image.crop(left, top, crop_width, crop_height)

    # 在周围添加黑色像素
    padded_image = vips_image.embed(move_x if move_x > 0 else 0, move_y if move_y > 0 else 0, width, height, extend='black')
    padded_image.write_to_file(os.path.join(save_refine_png_dir, vips_image_path.split(os.sep)[-1]))
    # new_img = vips_to_pil(padded_image)
    # new_img.show()


def save_pic(idx, he_png_path, ihc_png_path, move_x, move_y, final_con_mask_tn, save_refine_tn_dir, save_refine_png_dir, final_con_th=None):
    # 保存缩略图图像，2个，原始
    final_con_mask_tn.save(os.path.join(save_refine_tn_dir, f"{idx}_new_mask.png"))
    if final_con_th != None:
        final_con_th.save(os.path.join(save_refine_tn_dir, f"{idx}_new.png"))

    # 复制原始HE的png图像
    shutil.copy2(he_png_path, os.path.join(save_refine_png_dir, he_png_path.split(os.sep)[-1]))

    # 处理IHC的png图像并保存
    move_and_save_big_png(ihc_png_path, 64 * move_x, 64 * move_y, save_refine_png_dir)
    return True


if __name__ == "__main__":
    print("1对应原图中的64像素，对应16倍降采样的4像素。分割线中一个框对应的是16")
    print("为负表示向左，为正表示向右")

    folder_list = [("HE_ER", "ER"), ("HE_HER2", "HER2"), ("HE_KI67", "KI67"), ("HE_PGR", "PGR")]
    idx_list = [i for i in range(start_idx, end_idx + 1)]

    # Abandon low-quality samples
    abandon_idx_list = shared_inf.abandon_idx_list
    for abandon_idx in abandon_idx_list:
        if abandon_idx in idx_list:
            idx_list.remove(abandon_idx)

    for idx in idx_list:
        print(f"开始处理{idx}号wsi")

        # 4种类别，分别处理
        for he_ihc, ihc_kind in folder_list:
            # 设置全新的路径
            th_dir = os.path.join(tn_dir_base, he_ihc)
            png_dir = os.path.join(png_dir_base, he_ihc)
            mask_dir = os.path.join(mask_dir_base, he_ihc)
            save_refine_tn_dir = os.path.join(save_refine_tn_dir_base, he_ihc)
            save_refine_png_dir = os.path.join(save_refine_png_dir_base, he_ihc)

            # 获取路径，原始图像的缩略图和原图，不带分割线和mask
            th_path_list = glob.glob(os.path.join(th_dir, f"{idx}_*_z0.png"))
            ori_path_list = glob.glob(os.path.join(png_dir, f"{idx}_*_z0.png"))
            assert (len(th_path_list) == 2 and len(ori_path_list) == 2) or (len(th_path_list) == 0 and len(ori_path_list) == 0)
            if len(th_path_list) == 0 and len(ori_path_list) == 0:
                continue

            print("\t---------------")

            he_th_path = th_path_list[0] if "_HE_" in th_path_list[0] else th_path_list[1]
            ihc_th_path = th_path_list[1] if "_HE_" in th_path_list[0] else th_path_list[0]
            he_png_path = ori_path_list[0] if "_HE_" in ori_path_list[0] else ori_path_list[1]
            ihc_png_path = ori_path_list[1] if "_HE_" in ori_path_list[0] else ori_path_list[0]

            # 读取相关文件
            mask_1024 = np.load(os.path.join(mask_dir, f"{idx}_1024.npy"))
            he_th = Image.open(he_th_path)
            ihc_th = Image.open(ihc_th_path)

            # 开始切之前展示一下便于后续切
            ori_th = concatenate_image(add_divider(he_th, 64), add_divider(ihc_th, 64))
            while True:
                # 1对应原图中的64像素，对应16倍降采样的4像素。分割线中一个框对应的是16
                # 为负表示向左，为正表示向右
                move_x = int(input("\tx方向上移动："))
                move_y = int(input("\ty方向上移动："))
                new_ihc_th = move_pic(ihc_th, 4 * move_x, 4 * move_y)

                # 展示原始对比和加了mask的对比
                final_con_th = None
                final_con_th = concatenate_image(he_th, new_ihc_th, False)
                final_con_mask_tn = concatenate_image(add_mask_and_divide(he_th, mask_1024),
                                                      add_mask_and_divide(new_ihc_th, mask_1024))

                finished = input("\t是否完成任务，1表示完成：") == "1"
                if finished:
                    # 保存图像
                    print("\t开始保存图像")
                    save_pic(idx, he_png_path, ihc_png_path, move_x, move_y, final_con_mask_tn,save_refine_tn_dir, save_refine_png_dir, final_con_th)
                    print("\t保存完毕")
                    break
