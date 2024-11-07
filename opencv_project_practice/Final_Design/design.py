import cv2
import preprocess
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1. 数据集准备
dataset = np.load('data_test_tube_X.npy')
label = np.load('data_test_tube_y.npy')[1:]
template_tube = dataset[0]
data_list = dataset[1:]
i_shape = template_tube.shape


# 2. 产品倾斜角度估计
def estimate_rotation_angle(src):
    # 在这里实现产品倾斜角度估计算法
    # 返回图片的倾斜角度
    # 预处理图像
    blurred = cv2.GaussianBlur(src, (3, 3), 0)
    # 边缘检测
    edges = cv2.Canny(blurred, 80, 150)
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 闭运算
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 排序找到最大轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    # 画出最小外接矩形，得到角度
    rect_mask = cv2.minAreaRect(contours[0])
    # 如果矩形的长边和x轴的夹角大于45度，那么就将角度减去90度
    rotated_angle = rect_mask[2] if rect_mask[1][0] > rect_mask[1][1] else rect_mask[2] - 90
    # 画出矩形
    empty = np.zeros(src.shape, dtype=np.uint8)
    box = np.intp(cv2.boxPoints(rect_mask))
    # 旋转后的矩形
    col = cv2.cvtColor(empty, cv2.COLOR_GRAY2BGR)
    cv2.fillPoly(col, [box], (255, 255, 255))
    # 旋转到水平
    h, w = src.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, rotated_angle, 1.0)
    rotated_image = cv2.warpAffine(src, M, (w, h))
    rotated_mask = cv2.warpAffine(col, M, (w, h))
    mask_contour = cv2.cvtColor(rotated_mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    rect_mask = cv2.minAreaRect(contours[0])
    return [rotated_image, rotated_angle, rect_mask]


def calculate_outer_rect(src, padding=10) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    _, image_tube = cv2.threshold(src, 50, 255, cv2.THRESH_BINARY)
    con, _ = cv2.findContours(image_tube, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cts = sorted(con, key=cv2.contourArea, reverse=True)[:1]
    mask_rect = cv2.minAreaRect(cts[0])

    cw, ch = mask_rect[0]
    rh, rw = min(mask_rect[1]), max(mask_rect[1])
    dh, dw = int(ch - (rh // 2)), int(cw - (rw // 2))
    fh, fw = int(ch + (rh // 2)), int(cw + (rw // 2))
    cropped_image = src[dh - padding:fh + padding, dw - padding:fw + padding]
    return cropped_image, (dh, dw, fh, fw)


# 4. 产品对齐和IOU计算
def align_and_calculate_iou(image, template):
    # 特征提取
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    # 特征匹配
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(des1, des2)

    # 提取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffine2D(src_pts, dst_pts)

    # 对齐图像
    aligned_image = cv2.warpAffine(image, M, (template.shape[1], template.shape[0]))

    # 计算IOU
    intersection = np.logical_and(aligned_image, template)
    union = np.logical_or(aligned_image, template)
    iou_score = np.sum(intersection) / np.sum(union)

    # 返回对齐后的图像和IOU值
    return aligned_image, iou_score


def calculate_iou(rect_1: tuple[int, int, int, int], rect_2: tuple[int, int, int, int],
                  size: tuple[int, int] = (256, 256)) -> np.float32:
    table_1 = np.zeros(size, dtype=np.uint8)
    table_2 = np.zeros(size, dtype=np.uint8)
    table_1[rect_1[0]:rect_1[2], rect_1[1]:rect_1[3]] = 1
    table_2[rect_2[0]:rect_2[2], rect_2[1]:rect_2[3]] = 1
    # 计算IOU
    intersection = np.logical_and(table_1, table_2)
    union = np.logical_or(table_1, table_2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


#
template_tube_rotated = calculate_outer_rect(
    estimate_rotation_angle(template_tube[130 * i_shape[0] // 256:230 * i_shape[1] // 256])[0], padding=0)

estimated_list = []
for_iou_list = []
for tube_image in data_list:
    tube_image = tube_image[130 * i_shape[0] // 256:230 * i_shape[1] // 256]
    estimated = estimate_rotation_angle(tube_image)
    for_iou_list.append(calculate_outer_rect(tube_image, padding=0))
    estimated_list.append(estimated)

for i in range(3):
    p = np.where(label == i)[0]
    angle_list = [x[1] for x in estimated_list]
    angle_list = np.array(angle_list)
    print('第', i + 1, '类角度平均值为：', angle_list[p].mean(), '\t方差为：', angle_list[p].var())

print('--------------------------------------------------------------------')

iou_list = []
# 计算IOU
for i in range(3):
    p = np.where(label == i)[0]
    for j in p:
        iou_score = calculate_iou(for_iou_list[j][1], template_tube_rotated[1])
        iou_list.append(iou_score)
    iou_list = np.array(iou_list)
    print('第', i + 1, '类IOU平均值为：', iou_list.mean(), '\t方差为：', iou_list.var())
    iou_list = []


def preprocess(image):
    # 二值化
    _image = cv2.medianBlur(image, 3)
    # 残差连接，增强图像
    return image + _image


def absolute_compare(image: np.ndarray, template: np.ndarray) -> float:
    image = preprocess(image)
    template = preprocess(template)
    blk_1_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    tem_1_hist = cv2.calcHist([template], [0], None, [256], [0, 256])
    sim_1 = (cv2.compareHist(blk_1_hist, tem_1_hist, cv2.HISTCMP_INTERSECT))
    return sim_1.__float__()


def judge_by_block(image: np.ndarray, template: np.ndarray, depth: int = 1,
                   weight: tuple[int, int, int, int] = (1, 1, 1, 1),
                   cor_method: int = cv2.HISTCMP_CORREL, blur_size: int = 3) -> np.float32:
    if blur_size > 0:
        image = cv2.blur(image, [blur_size, blur_size])
        template = cv2.blur(template, [blur_size, blur_size])
    assert image.shape == template.shape and depth >= 0
    h, w = image.shape
    blk_1, tem_1 = image[:h // 2, :w // 2], template[:h // 2, :w // 2]
    blk_2, tem_2 = image[:h // 2, w // 2:], template[:h // 2, w // 2:]
    blk_3, tem_3 = image[h // 2:, :w // 2], template[h // 2:, :w // 2]
    blk_4, tem_4 = image[h // 2:, w // 2:], template[h // 2:, w // 2:]
    w1, w2, w3, w4 = weight
    blk_1_hist = cv2.calcHist([blk_1], [0], None, [256], [0, 256])
    tem_1_hist = cv2.calcHist([tem_1], [0], None, [256], [0, 256])
    sim_1 = cv2.compareHist(blk_1_hist, tem_1_hist, cor_method) * w1
    blk_2_hist = cv2.calcHist([blk_2], [0], None, [256], [0, 256])
    tem_2_hist = cv2.calcHist([tem_2], [0], None, [256], [0, 256])
    sim_2 = cv2.compareHist(blk_2_hist, tem_2_hist, cor_method) * w2
    blk_3_hist = cv2.calcHist([blk_3], [0], None, [256], [0, 256])
    tem_3_hist = cv2.calcHist([tem_3], [0], None, [256], [0, 256])
    sim_3 = cv2.compareHist(blk_3_hist, tem_3_hist, cor_method) * w3
    blk_4_hist = cv2.calcHist([blk_4], [0], None, [256], [0, 256])
    tem_4_hist = cv2.calcHist([tem_4], [0], None, [256], [0, 256])
    sim_4 = cv2.compareHist(blk_4_hist, tem_4_hist, cor_method) * w4
    sim_all = np.array([sim_1, sim_2, sim_3, sim_4]).mean()

    if depth != 0:
        _sim_1 = judge_by_block(blk_1, tem_1, depth=depth - 1)
        _sim_2 = judge_by_block(blk_2, tem_2, depth=depth - 1)
        _sim_3 = judge_by_block(blk_3, tem_3, depth=depth - 1)
        _sim_4 = judge_by_block(blk_4, tem_4, depth=depth - 1)
        _sim_all = np.array([_sim_1, _sim_2, _sim_3, _sim_4]).mean()
        sim_all = np.array([sim_all, _sim_all]).mean()

    return sim_all


def judge_by_line(a, b):
    plt.subplot(131)
    plt.imshow(a, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132)
    plt.imshow(b, cmap='gray')
    plt.xticks([]), plt.yticks([])

    a = cv2.blur(a, (7, 7))
    b = cv2.blur(b, (7, 7))
    # 开闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
    # 二值化
    _, a = cv2.threshold(a, 120, 255, cv2.THRESH_OTSU)
    _, b = cv2.threshold(b, 120, 255, cv2.THRESH_OTSU)
    a = cv2.Canny(a, 80, 150, apertureSize=3)
    b = cv2.Canny(b, 80, 150, apertureSize=3)
    # 对B膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    b_ = cv2.dilate(b, kernel)
    a_ = cv2.dilate(a, kernel)
    rt = (cv2.bitwise_or(a_, b) - a_) + (cv2.bitwise_or(a, b_) - b_)
    plt.subplot(133)
    plt.imshow(rt, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    return (rt.sum() / rt.shape[0] / rt.shape[1])


lst = []
res = []
for ind, test_tube_unit in enumerate(estimated_list):
    # 二值化
    outer_rect_o = calculate_outer_rect(test_tube_unit[0], padding=0)
    test = outer_rect_o[0]
    test = cv2.resize(test, template_tube_rotated[0].shape[::-1])
    temp = template_tube_rotated[0]
    t1 = test
    t2 = temp
    block_score = absolute_compare(t1, t2) / 1000
    block_score = judge_by_block(t1, t2, depth=1, weight=(1, 1, 1, 1),
                                 cor_method=cv2.HISTCMP_CORREL) + block_score
    if block_score > 13.5:
        res.append(1)
        continue

    line_score = judge_by_line(t1, t2)

    result = block_score * 5.5 + line_score
    if result > 73.55:
        res.append(3)
    else:
        res.append(2)

# 混淆矩阵，sklearn

con_mat = confusion_matrix(res, label + 1)
ConfusionMatrixDisplay(con_mat).plot()
plt.show()
plt.plot(res)
plt.show()
