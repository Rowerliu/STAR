import os.path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageOps
from PIL import ImageFilter
import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
from PIL import Image
from wsi_dataset import WSIDataset
# from src_for_train_set.wsi_dataset_train import WSIDataset

import tifffile

temp_dir = '..\\temp'
debug_flag = True
pad_in_ihcPad = 0


# 即时处理可视化
def show_image(ndarray_image, image_name=''):
    # pass
    plt.rcParams['font.family'] = 'SimHei'  # 指定要使用的中文字体，比如SimHei
    plt.rcParams['axes.unicode_minus'] = False
    plt.imshow(ndarray_image)
    plt.title(image_name)
    plt.axis('off')  # 可选：隐藏坐标轴
    plt.show()  # 显示图像


def base_transforms():
    transforms_ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transforms_


# 调用代码：df = filtered_patches(wsi, 64, 230, 20)，后面3个都是常数
def filtered_patches(wsi, stride, white_thr, black_thr):
    # 缩小WSI为缩略图，缩小程度由参数stride控制
    w, h = wsi.size

    # 因为Image.thumbnail()是直接对Image对象进行操作，所以需要进行深拷贝
    wsi_copy = wsi.copy()
    wsi_copy.thumbnail((w // stride, h // stride))

    # 将PIL.Image转为ndarry，大小为(h,w)，第1，2维的顺序与Image完全不同
    thumb = np.array(wsi_copy.convert('L'))

    # 对缩略图进行灰度阈值过滤，得到满足条件的像素组成的布尔数组
    arr = np.logical_and(thumb < white_thr, thumb > black_thr)

    # 创建空的DataFrame
    df = pd.DataFrame(columns=['dim1', 'dim2'])

    # 根据满足条件的像素索引计算图像块的中心坐标，并赋值给DataFrame的'dim1'和'dim2'列
    # 注意此处的(dim1, dim2)对应了(w，h)
    df['dim1'], df['dim2'] = stride * np.where(arr)[1] + (stride // 2), stride * np.where(arr)[0] + (stride // 2)
    return df


def foreground_detection_model(foreground_model_path):
    model = models.resnet18().cuda()
    model.fc = torch.nn.Linear(512, 2).cuda()
    model.load_state_dict(torch.load(foreground_model_path))
    model.eval()
    return model


def get_foreground(model, wsi_path, batch_size=512, white_thr=230, black_thr=20, stride=64, downsample=32, wsi=None):
    # wsi = OpenSlide(wsi_path)  todo 需要测试，替换了之前的读取形式，包括filtered_patches
    # wsi = Image.open(wsi_path)
    # 使用 tifffile 读取大尺寸 TIFF 文件，开启内存映射以节省内存
    if wsi is not None:
        wsi_array = wsi
    else:
        ext = os.path.splitext(wsi_path)[1].lower()
        if ext == '.tif' or ext == '.tiff':
            wsi_array = tifffile.imread(wsi_path)
        else:
            wsi_array = cv2.imread(wsi_path)
            if wsi_array is None:
                raise ValueError(f"Failed to read image from {wsi_path}")
            # Convert BGR to RGB
            wsi_array = cv2.cvtColor(wsi_array, cv2.COLOR_BGR2RGB)

    # Ensure array is in uint8 format for PIL
    if wsi_array.dtype != np.uint8:
        wsi_array = (wsi_array / wsi_array.max() * 255).astype(np.uint8)

    # 如果 filtered_patches 等后续函数依赖 PIL 接口，则转换为 PIL Image
    wsi = Image.fromarray(wsi_array)
    df = filtered_patches(wsi, stride, white_thr, black_thr)

    print(f'\tin get_foreground, dataloader batches = {1 + (len(df) // batch_size)}')
    dataset = WSIDataset(df, wsi, base_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = np.zeros(len(dataset))
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=1 + (len(df) // batch_size)):
            out_ = model(data.cuda())
            preds[batch_size * i: batch_size * i + data.shape[0]] = torch.argmax(out_, axis=1).cpu().numpy()
    df['pred'] = preds.astype(int)

    # 再生成缩略图
    w, h = wsi.size
    tn = wsi.copy()
    tn.thumbnail((w // downsample, h // downsample))
    tn = tn.convert('L')

    foreground = np.zeros_like(tn)
    df = df.loc[df['pred'] == 1].copy()
    df['dim1'] = df['dim1'] // downsample
    df['dim2'] = df['dim2'] // downsample
    # 使用上述取出的图像块坐标作为索引，在全零数组（foreground）中标记对应位置为1，表示对应位置是前景。
    foreground[df.values.T[:2].astype(int)[1], df.values.T[:2].astype(int)[0]] = 1

    return foreground


def bbox_helper(density, bins):
    dmap = []
    for i in range(0, len(density), len(density) // bins):
        dmap.append(np.sum(density[i: i + len(density) // bins]))
    for item in range(dmap.index(sorted(dmap)[-5]), len(dmap)):
        if dmap[item] < sorted(dmap)[-5] // 20: break
    end = item + 1 if item < len(density) - 1 else item
    for item in range(dmap.index(sorted(dmap)[-5]), 0, -1):
        if dmap[item] < sorted(dmap)[-5] // 20: break
    start = item - 1 if item > 1 else item
    return start, end


def get_bbox_primary(foreground):
    # 在宽度方向上进行裁剪，获取区间 [r, l]
    density_x = np.sum(foreground, axis=0)
    bins = 100
    start_x, end_x = bbox_helper(density_x, bins)
    r, l = (len(density_x) // bins) * start_x, (len(density_x) // bins) * end_x
    foreground = foreground[:, r:l]  # 这里裁剪了一次了

    # 在高度方向上进行裁剪，获取区间 [t, b]
    density_y = np.sum(foreground, axis=1)
    start_y, end_y = bbox_helper(density_y, bins)
    t, b = (len(density_y) // bins) * start_y, (len(density_y) // bins) * end_y

    # 第一个为彻底最终的mask图像，后面的为图像的范围，表示为foreground[t:b, r:l]
    return foreground[t:b, :], (t, b, r, l)


def get_bbox_secondory(arr):
    r = np.where((np.cumsum(np.sum(arr, axis=0)) / np.sum(arr)) > 0.05)[0][0]
    l = np.where((np.cumsum(np.sum(arr, axis=0)) / np.sum(arr)) < 0.95)[-1][-1]

    t = np.where((np.cumsum(np.sum(arr, axis=1)) / np.sum(arr)) > 0.05)[0][0]
    b = np.where((np.cumsum(np.sum(arr, axis=1)) / np.sum(arr)) < 0.95)[-1][-1]

    return arr[t:b, r:l], (t, b, r, l)


def get_bbox(arr):
    bbox = get_bbox_primary(arr)
    # bbox是一个tuple，大小为2
    # bbox[0]是裁剪后的mask
    # bbox[1]是裁剪的范围，对应bbox[0]所处的位置区间
    if np.sum(bbox[0]) / np.sum(arr) > 0.8:
        # 计算裁减过后的前景mask图像 占 原始mask图像前景 的 比值
        # 如果此占比高于80%，直接以此作为裁剪后的mask
        return bbox

    bbox_sec = get_bbox_secondory(arr)
    if np.sum(bbox[0]) / np.sum(arr) > np.sum(bbox_sec[0]) / np.sum(arr):
        # 前面的框的效果还更好，所以返回之前的结果bbox
        return bbox
    return bbox_sec


def simple_bbox_crop(mask):
    """直接根据非零像素的边界裁剪图像"""
    # 水平方向：找到左右边界
    rows = np.any(mask, axis=0)
    cols = np.any(mask, axis=1)

    # 找到有效区域的边界
    left, right = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, mask.shape[1] - 1)
    top, bottom = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, mask.shape[0] - 1)

    # 返回裁剪后的图像和坐标 (t, b, l, r)
    return mask[top:bottom + 1, left:right + 1], (top, bottom + 1, left, right + 1)


def ref_conv_input(ref_path, box, fg, stride):
    # 创建一个7x7的数组k，其中元素全为1。此数组用于进行膨胀操作
    k = np.ones((7, 7))
    # 使用OpenCV的dilate函数对前景图像fg进行膨胀操作
    fg = cv2.dilate(fg, k, 1)
    # 创建一个与前景图像形状相同的布尔数组bg，其中元素为True表示为背景，False表示为前景
    bg = fg == 0

    # Prepare reference image
    ext = os.path.splitext(ref_path)[1].lower()
    if ext == '.tif' or ext == '.tiff':
        wsi_array = tifffile.imread(ref_path)
    else:
        wsi_array = cv2.imread(ref_path)
        if wsi_array is None:
            raise ValueError(f"Failed to read image from {ref_path}")
        # Convert BGR to RGB
        wsi_array = cv2.cvtColor(wsi_array, cv2.COLOR_BGR2RGB)
    # 如果 filtered_patches 等后续函数依赖 PIL 接口，则转换为 PIL Image
    wsi = Image.fromarray(wsi_array)

    w, h = wsi.size
    # stride的取值就是args.downsample，保持与取mask时的方法一致
    wsi.thumbnail((w // stride, h // stride))
    tn = wsi.convert("L")

    if debug_flag:
        tn.save(os.path.join(temp_dir, "ref_conv_input_1_tn_origin_downsample.png"))

    # 将包围框box的四个坐标值分别赋值给变量a，b，c，d
    a, b, c, d = box
    bg = bg[a:b, c:d]
    # 从缩放后的灰度图像tn中提取与包围框对应的区域，并通过NumPy数组转换为PIL图像对象
    image = Image.fromarray(np.array(tn)[a:b, c:d])
    if debug_flag:
        tnn = wsi.copy()
        ori_image = Image.fromarray(np.array(tnn)[a:b, c:d])
        ori_image.save(os.path.join(temp_dir, "ref_conv_input_2_cropped_wsi_down_image.png"))

        # if True:
        #     # 得到的是原图大小的图像
        #     tnn = wsi.get_thumbnail((w, h))
        #     ori_image = Image.fromarray(np.array(tnn)[a*stride:b*stride, c*stride:d*stride])
        #     ori_image.save(os.path.join(temp_dir, "save_he_cropped_wsi_ori_image.png"))

        image.save(os.path.join(temp_dir, "ref_conv_input_3_cropped_image.png"))

    # 对提取的图像进行直方图均衡化操作，然后将像素值小于50的部分生成一个布尔遮罩（mask）
    mask = np.array(ImageOps.equalize(image, mask=None)) < 50
    # 对图像进行反转操作，使其逆转
    image = ImageOps.invert(image)

    if debug_flag:
        image.save(os.path.join(temp_dir, "ref_conv_input_3_inverse_image.png"))

    image = np.array(image)
    # 使用遮罩将图像中与背景相关的部分设置为255（白色）
    image[mask] = 255

    min_, max_ = np.min(image), np.max(image)
    # 对图像进行归一化操作，将像素值缩放到0~1的范围内
    image = (image - min_) / (max_ - min_)
    image = (image + 1) / 2
    image = 255 * image
    # 将背景对应的像素值设置为0（黑色）
    image[bg] = 0
    # 将图像数组的数据类型转换为无符号8位整数（uint8）
    image = image.astype('uint8')
    return image


def trs_conv_input(trs_path, thumbnail=True):

    # Prepare transform image
    if thumbnail:
        # wsi = Image.open(mod_path)
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
        wsi.thumbnail((w // 32, h // 32))
        tn = wsi.convert("L")

        if debug_flag:
            tn.save(os.path.join(temp_dir, "ihc_conv_input_1_tn_downSample.png"))
    else:
        tn = Image.open(trs_path).convert("L")

    tn = np.array(tn)
    # tn[tn < 30] = 255：将tn中灰度值小于30的像素点设置为255（白色）
    tn[tn < 30] = 255
    tn = Image.fromarray(tn)
    if debug_flag and thumbnail:
        tn.save(os.path.join(temp_dir, "ihc_conv_input_2_change_background_white_image.png"))

    # 应用模糊滤波器对图像进行模糊处理
    tn = tn.filter(ImageFilter.BLUR)
    if debug_flag and thumbnail:
        tn.save(os.path.join(temp_dir, "ihc_conv_input_3_mohu_image.png"))

    tn = np.array(tn)
    # 对图像数组中的每个像素值执行255减法操作，即将像素值反转，得到亮度反转后的图像
    trs = 255.0 - tn
    # 将ihc转换为浮点型数据
    trs = trs.astype(float)
    # 返回的是0~255
    return trs


# 接受一个HE模板图像作为参数，并返回一个填充后的HE模板图像
# 主要目的是对HE模板图像进行边界填充，以使图像的宽高比例保持一致，并在边界周围添加一定数量的填充
def pad_he(he_template):
    if he_template.shape[0] % 2 == 1:
        pad = 1
        he_template = cv2.copyMakeBorder((he_template).astype('uint8'), pad, 0, 0, 0, 0)
    if he_template.shape[1] % 2 == 1:
        pad = 1
        he_template = cv2.copyMakeBorder((he_template).astype('uint8'), 0, 0, 0, pad, 0)

    if he_template.shape[0] > he_template.shape[1]:
        pad = (he_template.shape[0] - he_template.shape[1]) // 2
        he_template = cv2.copyMakeBorder((he_template).astype('uint8'), 0, 0, pad, pad, 0)

    if he_template.shape[0] < he_template.shape[1]:
        pad = (he_template.shape[1] - he_template.shape[0]) // 2
        he_template = cv2.copyMakeBorder((he_template).astype('uint8'), pad, pad, 0, 0, 0)

    pad = max(list(he_template.shape)) // 4
    he_template = cv2.copyMakeBorder((he_template).astype('uint8'), pad, pad, pad, pad, 0)
    return he_template


def pad_ihc(ihc, he_template):
    # 就是在四周加上pad大小的像素
    pad = max(he_template.shape) // 2
    global pad_in_ihcPad
    pad_in_ihcPad = 0  # 目前是无用的
    # pad_in_ihcPad = pad
    transform_ = transforms.Pad([pad, pad, pad, pad])
    ihc = Image.fromarray(ihc.astype('uint8'))
    # 暂时不进行pad操作
    # ihc = transform_(ihc)
    ihc = np.array(ihc)
    ihc = torch.Tensor(ihc).cuda()
    return ihc


def rotation_matrix(he_template, astride, start, end):
    # 旋转模板的中心点坐标 c，通过将 he_template 的宽度和高度各自除以 2 来计算。
    c = he_template.shape[1] // 2, he_template.shape[0] // 2
    # 计算旋转矩阵的平面数量 num_planes，它表示在指定范围内的旋转角度之间的平面数量。计算方法是将 end 减去 start，然后除以步长 astride
    num_planes = (end - start) // astride

    rot_matrix = torch.zeros((num_planes, he_template.shape[0], he_template.shape[1]), requires_grad=False).cuda()
    he_template = he_template.astype('uint8')

    for plane in range(num_planes):
        # 计算当前循环迭代中的旋转角度 theta，就是当前的索引 * 步长
        theta = start + (plane * astride)
        # 计算给定中心点 c、旋转角度 theta 和缩放因子为 1.0 的旋转矩阵 M。维度为 [2, 3]
        M = cv2.getRotationMatrix2D(c, theta, 1.0)
        # 这一行代码应用旋转矩阵 M 到 he_template，使用 cv2.warpAffine 函数生成旋转后的图像数组 rotated。
        # he_template.shape[1] 表示图像宽度，he_template.shape[0] 表示图像高度
        # rotated是旋转后的图像
        rotated = cv2.warpAffine(he_template, M, (he_template.shape[1], he_template.shape[0]))
        rot_matrix[plane] = torch.Tensor(rotated.astype(float))
        # torch.save(rot_matrix, 'rot_matrix_new.pt')
    return rot_matrix


def max_conv(ihc, rot_matrix, stride):
    max_ = -9999999
    reg_data = None
    for angle in range(rot_matrix.shape[0]):
        input_ = ihc[(None,) * 2]
        # rot_matrix[angle]用于提取某一旋转角度下的HE_template图像，此时实际对应的旋转角度是10*angle
        # (None,) * 2用于在原先的tensor前面拓展维度，拓展两次
        weight = rot_matrix[angle][(None,) * 2]

        # 获取输入图像和模板的空间尺寸
        input_h, input_w = input_.shape[-2:]
        kernel_h, kernel_w = weight.shape[-2:]
        if kernel_h >= input_h or kernel_w >= input_w:
            scale_factor_h = (input_h - 1) / kernel_h
            scale_factor_w = (input_w - 1) / kernel_w
            scale_factor = min(scale_factor_h, scale_factor_w)
            new_h = max(1, int(kernel_h * scale_factor))
            new_w = max(1, int(kernel_w * scale_factor))
            # 使用双线性插值缩放 kernel，注意 weight 的 shape 为 (1,1,kH,kW)
            weight = torch.nn.functional.interpolate(weight, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # out_ 被赋值为使用 torch.nn.functional.conv2d 函数进行卷积操作的输出结果，其中输入为 input_，权重为 weight，并且指定了步长为 stride
        out_ = torch.nn.functional.conv2d(input_, weight, stride=stride)

        if int(torch.max(out_)) > max_:
            max_ = torch.max(out_)
            argmax = torch.where(out_ == max_)
            reg_data = stride * int(argmax[2][0]), stride * int(argmax[3][0]), angle

    return reg_data


# note 真正负责配准的函数
def register(he_template, ihc):
    # 对HE图像进行填充 ! 现在决定不做这个事
    # if debug_flag:
    #     show_image(he_template, "pad前的HE图像")
    # # he_template = pad_he(he_template)
    # if debug_flag:
    #     show_image(he_template, "pad后的HE图像")

    # 角度和步长的初始值
    astride = 10
    stride = 10
    # rot_matrix 生成的是一个列表，里面每一个元素都是一张图，为he_template旋转后的结果
    rot_matrix = rotation_matrix(he_template, astride, 0, 360)

    # ---------------------------------
    if debug_flag:
        show_image(ihc, "pad前的IHC图像")
    # 对IHC图像进行填充
    ihc = pad_ihc(ihc, he_template)
    if debug_flag:
        show_image(ihc.cpu().numpy(), "pad后的IHC图像")

    # 通过卷积操作在IHC图像中搜索与HE模板图像最相似的区域，并返回该区域在IHC图像中的位置（x坐标和y坐标）和旋转角度。
    x_strided, y_strided, angle_strided = max_conv(ihc, rot_matrix, stride)
    angle_strided = astride * angle_strided
    if debug_flag:
        pad = max(he_template.shape) // 2
        show_image(ihc[x_strided:x_strided + 2 * pad, y_strided:y_strided + 2 * pad].cpu().numpy(), "粗配准后的IHC图像")

    # 更新角度和步长的值，以较小的步长进行进一步的配准
    astride = 1
    stride = 1

    # 生成更细致的旋转后的HE图像，旋转矩阵的角度在 angle_strided - 10到 angle_strided + 10度之间，用于在IHC图像中更细致地寻找与HE模板图像最佳匹配的区域
    rot_matrix = rotation_matrix(he_template, astride, angle_strided - 10, angle_strided + 10)

    # 对ihc进行裁剪，仅仅获取 关键位置和其周围 的图像
    pad = max(he_template.shape) // 2
    debug_ihcc = ihc.clone()
    ihc = ihc[max(x_strided - 50, 0):x_strided + 50 + (2 * pad), max(y_strided - 50, 0):y_strided + 50 + (2 * pad)]

    # 展示第一次粗配准后得到的图像，也是第二次细配准时输入的ihc
    if debug_flag:
        show_image(ihc.cpu().numpy(), "粗配准后得到的图像，也是第二次细配准时输入的ihc")

    # 在精细的情况下进行进一步的配准
    x_, y_, angle_ = max_conv(ihc, rot_matrix, stride)
    # 修正x_, y_, theta的信息，要加上之前的粗略情况下的信息
    x_ = max(x_strided - 50, 0) + x_
    y_ = max(y_strided - 50, 0) + y_
    if debug_flag:
        debug_ihc = debug_ihcc[max(x_, 0): x_ + pad * 2, max(y_, 0):y_ + pad * 2]
        show_image(ihc.cpu().numpy(), "输入的IHC图像")
        show_image(debug_ihcc.cpu().numpy(), "pad后的IHC")
        show_image(debug_ihc.cpu().numpy(), "细配准后的IHC图像")

    theta = angle_strided + angle_ - 10
    return x_, y_, theta
