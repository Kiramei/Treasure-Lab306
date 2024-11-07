import cv2
import numpy as np
import matplotlib.pyplot as plt


def ex5():
    # 读取灰度图像
    img = cv2.imread('./shenzhen_gray.bmp', cv2.IMREAD_GRAYSCALE)

    # 获取图像的宽度和高度
    h, w = img.shape

    # 将图像矩阵降维成一维数组
    img = img.reshape(h * w)

    # 绘制直方图
    plt.hist(img, bins=100, density=True, color="#29a3ff")
    plt.show()


if __name__ == '__main__':
    # 调用ex5函数
    ex5()