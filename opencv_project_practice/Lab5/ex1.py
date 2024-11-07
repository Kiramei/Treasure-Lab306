"""

编写函数P4_rgb2gray() 实现彩色图到灰度图的转换，转换公式如下
Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.144 * B(i,j);
其中R,G,B 分别为彩色图像的RGB通道


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def P4_rgb2gray(img):
    # 先将图像转换为float32类型
    img = img.astype(np.float32)
    # 灰度图像的计算公式
    img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.144 * img[:, :, 2]
    return img_gray

if __name__ == '__main__':
    image = mpimg.imread('./images/shenzhen.png')
    # 对比演示
    gray = P4_rgb2gray(image)
    gray_cv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.figure(figsize=(8, 8))
    plt.subplot(131)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image)
    plt.title('RGB')
    plt.subplot(132)
    plt.xticks([]), plt.yticks([])
    plt.imshow(gray, cmap='gray')
    plt.title('gray_mine')
    plt.subplot(133)
    plt.xticks([]), plt.yticks([])
    plt.imshow(gray_cv, cmap='gray')
    plt.title('gray_cv')
    plt.show()




