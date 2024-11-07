import cv2
import numpy as np

from constants import *


def calculate_skew(image):
    # 将图像转换为灰度图
    gray = image

    # 对图像进行边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = cv2.dilate(src=edges, kernel=np.ones((20, 20), dtype=np.uint8))
    # 执行霍夫直线变换，检测图像中的直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    # 得到最上面的一条直线，这条直线的角度就是图像的旋转角度
    line = sorted(lines.squeeze(1), key=lambda x: x[0], reverse=True)[0]
    plt.imshow(edges, cmap='gray')
    plt.show()
    # 计算直线的角度
    theta = line[1]
    angle = theta * 180 / np.pi

    # 计算角度的平均值
    # avg_angle = np.mean(angles)

    return angle


def rotate_image(image, angle):
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算旋转中心
    center = (width // 2, height // 2)

    # 定义旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 执行旋转操作
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

    return rotated_image


def rotate_to_horizon(src):
    skew_angle = calculate_skew(src)
    print(skew_angle)
    corrected_image = rotate_image(src, -skew_angle)
    return corrected_image
