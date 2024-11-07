import cv2
import numpy as np
from matplotlib import pyplot as plt


def solve():
    # 读取图像
    image = cv2.imread('data/girl.png', cv2.IMREAD_GRAYSCALE)

    # 傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 计算频域的幅度谱和相位谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)

    # 绘制频域的幅度谱
    plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    # 绘制频域的相位谱
    plt.subplot(122), plt.imshow(phase_spectrum, cmap='gray')
    plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])

    # 傅里叶反变换，重建图像
    fshift = np.fft.ifftshift(fshift)
    reconstructed_image = np.fft.ifft2(fshift)
    reconstructed_image = np.abs(reconstructed_image)

    # 计算重建图像和原图的PSNR差异
    mse = np.mean((image - reconstructed_image) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)

    # 输出重建图像，计算PSNR值
    plt.figure()
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed'), plt.xticks([]), plt.yticks([])
    plt.show()
    print('重建后与原图的PSNR差异:', psnr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    solve()
