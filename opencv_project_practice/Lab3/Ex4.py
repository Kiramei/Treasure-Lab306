import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussian_highpass_filter(image, sigma):
    rows, cols = image.shape
    # 计算中心
    center_x = int(cols / 2)
    center_y = int(rows / 2)
    # 创建滤波器
    filter = np.zeros((rows, cols), dtype=np.float32)
    # 计算滤波器权值，生成滤波器矩阵
    # 公式：1-e^{-\frac{d^2}{2\sigma^2}}
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            filter[i, j] = 1 - np.exp(-d ** 2 / (2 * sigma ** 2))
    # 进行滤波
    result = np.fft.ifftshift(np.fft.fft2(image) * filter)
    result = np.abs(np.fft.ifft2(np.fft.fftshift(result)))
    return result


if __name__ == '__main__':
    # 读取原始图像
    image = cv2.imread('data/girl.png', cv2.IMREAD_GRAYSCALE)
    # 设计和应用高斯高通滤波器
    sigma1 = 50  # 第一种滤波器的标准差
    filtered_image1 = gaussian_highpass_filter(image, sigma1)
    sigma2 = 1000  # 第二种滤波器的标准差
    filtered_image2 = gaussian_highpass_filter(image, sigma2)
    # 显示滤波结果
    plt.subplot(121), plt.imshow(filtered_image1, cmap='gray')
    plt.title('Filtered Image (Sigma = 50)'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(filtered_image2, cmap='gray')
    plt.title('Filtered Image (Sigma = 1000)'), plt.xticks([]), plt.yticks([])
    # 显示直方图
    plt.figure()
    plt.subplot(121), plt.hist(filtered_image1.ravel(), 256, [0, 256])
    plt.title('Histogram (Sigma = 50)'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.hist(filtered_image2.ravel(), 256, [0, 256])
    plt.title('Histogram (Sigma = 1000)'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    # 显示频域图像
    plt.figure()
    plt.subplot(121), plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image)))), cmap='gray')
    plt.title('Frequency Domain Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(filtered_image2)))), cmap='gray')
    plt.title('Frequency Domain Image 2'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
