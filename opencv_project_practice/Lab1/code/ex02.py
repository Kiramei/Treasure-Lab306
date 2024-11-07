import cv2
from numpy import ndarray
import numpy as np
import os
from functools import reduce


def ex3() -> None:
    # 创建一个空数组来存储图像
    im_arr = []
    # 获取指定目录下的文件列表
    fs = os.listdir('./noiseimages')
    # 遍历文件列表
    for f in fs:
        # 读取图像文件并将其转换为灰度图像（BGR到灰度）
        im = cv2.imread('./noiseimages/' + f, cv2.COLOR_BGR2GRAY)
        im_arr.append(im.astype(np.float64))
    # 使用reduce函数对数组中的每个元素矩阵进行累加，然后除以文件数得到平均图像
    ps = reduce(lambda x, y: x + y, im_arr)
    ps = (ps / len(fs)).astype(np.uint8)
    # 显示实验结果图像
    cv2.imshow('Experiment 3', ps)
    cv2.waitKey(0)


if __name__ == '__main__':
    # 调用ex3函数
    ex3()
