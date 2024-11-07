# 均值滤波
import cv2
import numpy as np


def read_image(path: str):
    '''
    Read image from path
    :param path: The path of image
    :return: The data of image
    '''
    return cv2.imread(path)


def mean_filter(src: np.ndarray, scale: int):
    '''
    Mean filter
    :param src: The data of image
    :return: The data of image after mean filter
    '''
    return cv2.blur(src, (scale, scale))


if __name__ == '__main__':
    img = read_image('data/shenzhen_noise.bmp')
    cv2.imshow('img_with_noise', img)
    for i in range(2, 10, 2):
        img_mean_filter = mean_filter(img, i)
        cv2.imshow(f'img_mean_filter-Scale:{i}', img_mean_filter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
