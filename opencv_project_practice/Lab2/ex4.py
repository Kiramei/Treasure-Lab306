import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path: str):
    return cv2.imread(path)


def equalize_image(src: cv2.UMat):
    gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    return gray_img


def histogram_norm(image, ref):
    out = np.zeros_like(image)
    hist_img, _ = np.histogram(image.flatten(), 256, [0, 256])
    hist_ref, _ = np.histogram(ref.flatten(), 256, [0, 256])
    cdf_img = np.cumsum(hist_img)
    cdf_img = cdf_img / cdf_img[-1]  # 归一化
    cdf_ref = np.cumsum(hist_ref)
    cdf_ref = cdf_ref / cdf_ref[-1]  # 归一化

    lut = np.interp(cdf_img, cdf_ref, np.arange(256))
    out = lut[image]

    return out.astype(np.uint8)


def equalize_image_with_style_transfer(src: np.ndarray, ref: np.ndarray):
    gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_img = histogram_norm(gray_img, ref)
    return gray_img


if __name__ == '__main__':
    img = read_image('data/shenzhen_gray.bmp')
    img_equalized = equalize_image(img)
    img_equalized_with_style_transfer = equalize_image_with_style_transfer(img, img_equalized)

    cv2.imshow('img', img)
    cv2.imshow('img_equalized', img_equalized)
    cv2.imshow('img_equalized_with_style_transfer', img_equalized_with_style_transfer)
    # show three images
    plt.subplot(3, 1, 1)
    plt.tight_layout(pad=3)
    plt.title('Before equalization')
    plt.hist(img.ravel(), 256, [0, 256])
    plt.subplot(3, 1, 2)
    plt.title('After equalization')
    plt.hist(img_equalized.ravel(), 256, [0, 256])
    plt.subplot(3, 1, 3)
    plt.title('After equalization with style transfer')
    plt.hist(img_equalized_with_style_transfer.ravel(), 256, [0, 256])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()