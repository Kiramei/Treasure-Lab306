import cv2
import matplotlib.pyplot as plt
import numpy


# Read image
def read_image(path: str):
    '''
    Read image from path
    :param path: The path of image
    :return: The data of image
    '''
    return cv2.imread(path)


# show the gray ratio histogram
def equalize_image(src: numpy.ndarray):
    gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    return gray_img


def show_histogram(before: numpy.ndarray, after: numpy.ndarray):
    '''
    Show the two histograms in one figure for comparison
    :param before: The data of image before equalization
    :param after: The data of image after equalization
    :return: None
    '''
    plt.subplot(2, 1, 1)
    plt.tight_layout(pad=3)
    plt.title('Before equalization')
    plt.hist(before.ravel(), 256, [0, 256])
    plt.subplot(2, 1, 2)
    plt.title('After equalization')
    plt.hist(after.ravel(), 256, [0, 256])
    plt.show()


if __name__ == '__main__':
    img = read_image('data/shenzhen_gray.bmp')
    img_equalized = equalize_image(img)
    show_histogram(img, img_equalized)
    cv2.imshow('img', img)
    cv2.imshow('img_equalized', img_equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
