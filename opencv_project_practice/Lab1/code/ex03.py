import cv2
from numpy import ndarray
import numpy as np

def func_1(src: ndarray) -> ndarray:
    # 创建与源图像相同大小的空数组
    ret = np.ndarray((len(src[:, -1]), len(src[-1, :])))
    # 遍历源图像的每个像素
    for i in range(len(src[:, -1])):
        for j in range(len(src[-1, :])):
            # 获取像素的B、G、R通道值
            b, g, r = src[i, j]
            # 根据给定的权重计算新的像素值
            f = int(b * 0.3 + g * 0.35 + r * 0.35)
            # 将新的像素值存储在目标数组中
            ret[i, j] = f
    # 将目标数组的数据类型转换为uint8
    return ret.astype(dtype=np.uint8)

def func_2(src: ndarray) -> ndarray:
    # 使用OpenCV的cvtColor函数将图像从BGR颜色空间转换为灰度图像
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

def ex2() -> None:
    # 读取彩色图像
    img = cv2.imread('./abc.jpg', cv2.IMREAD_COLOR)
    # 调用函数func_1和func_2对图像进行处理
    ret_1 = func_1(img)
    ret_2 = func_2(img)
    # 显示处理后的图像
    cv2.imshow('Experiment 2-a', ret_1)
    cv2.imshow('Experiment 2-b', ret_2)
    cv2.waitKey(0)

if __name__ == '__main__':
    # 调用ex2函数
    ex2()