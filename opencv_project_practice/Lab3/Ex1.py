# Corresponding experiment: 2

import cv2


def read_coke_image():
    return cv2.imread("./data/coke.png", cv2.IMREAD_GRAYSCALE)


def canny_edge_detection(image):
    return cv2.Canny(image, 38, 0)


def laplacian_edge_detection(image):
    return cv2.Laplacian(image, cv2.CV_64F)


def sobel_edge_detection(image):
    return cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)


if __name__ == '__main__':
    image = read_coke_image()
    cv2.imshow("original", image)
    cv2.imshow("sobel", sobel_edge_detection(image))
    cv2.imshow("laplacian", laplacian_edge_detection(image))
    cv2.imshow("canny", canny_edge_detection(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
