# 自编函数my_median_filter()，实现中值滤波, 并同cv2.medianBlur的效果做比较，测试不同滤波器大小下的滤波效果。
import cv2
import numpy as np


def read_image(path: str):
    '''
    Read image from path
    :param path: The path of image
    :return: The data of image
    '''
    return cv2.imread(path)


def original_median_filter(src: np.ndarray):
    '''
    Median filter
    :param src: The data of image
    :return: The data of image after median filter
    '''
    return cv2.medianBlur(src, 3)


def my_median_filter(src: np.ndarray):
    '''
    Median filter
    :param src: The data of image
    :return: The data of image after median filter
    '''
    img = src.copy()
    height, width = img.shape[:2]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            img[i, j] = np.median(src[i - 1:i + 2, j - 1:j + 2])
    return img


if __name__ == '__main__':
    img = read_image('data/shenzhen_noise.bmp')
    cv2.imshow('img', img)
    img_median_filter = original_median_filter(img)
    img_my_median_filter = my_median_filter(img)
    cv2.imshow(f'img_median_filter', img_median_filter)
    cv2.imshow(f'img_my_median_filter', img_my_median_filter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
