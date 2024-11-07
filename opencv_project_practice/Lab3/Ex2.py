# Corresponding experiment: 3

import cv2
from matplotlib import pyplot as plt


def read_coke_image():
    return cv2.imread("./data/chemical_tube.png", cv2.IMREAD_GRAYSCALE)


#
def bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


def gaussian_filter(image):
    return cv2.GaussianBlur(image, (9, 9), 0)


if __name__ == '__main__':
    image = read_coke_image()
    plt.subplot(311), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(312), plt.imshow(bilateral_filter(image), cmap='gray'), plt.title('Bilateral')
    plt.xticks([]), plt.yticks([])
    plt.subplot(313), plt.imshow(gaussian_filter(image), cmap='gray'), plt.title('Gaussian')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
