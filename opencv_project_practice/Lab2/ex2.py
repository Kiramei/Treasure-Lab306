import cv2
import numpy as np


# Read image
def read_image(path: str):
    '''
    Read image from path
    :param path: The path of image
    :return: The data of image
    '''
    return cv2.imread(path)


# show the gray ratio histogram
def equalize_image(src: np.ndarray):
    gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    return gray_img


def calc_the_hist_vector(img: np.ndarray):
    '''
    Calculate the histogram vector of image
    :param img: The data of image
    :return: The histogram vector of image
    '''
    return cv2.calcHist([img], [0], None, [256], [0, 256])


def calc_the_euclidean_distance(vector1: np.ndarray, vector2: np.ndarray):
    '''
    Calculate the Euclidean distance of two vectors
    :param vector1: The first vector
    :param vector2: The second vector
    :return: The Euclidean distance of two vectors
    '''
    return np.sqrt(np.sum(np.square(vector1 - vector2)))


if __name__ == '__main__':
    img = read_image('data/shenzhen_gray.bmp')
    img_equalized = equalize_image(img)
    hist_vector = calc_the_hist_vector(img)
    hist_vector_equalized = calc_the_hist_vector(img_equalized)
    print(calc_the_euclidean_distance(hist_vector, hist_vector_equalized))
