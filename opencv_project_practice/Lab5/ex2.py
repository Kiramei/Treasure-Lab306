"""
2.2 利用颜色进行目标识别
目的：检测图像blue_sign.png 中的蓝色标志牌。
	方法1. 如果使用灰度图中，是否可以检测该蓝色标志牌？为什么？
	方法2. 如果使用RGB彩色图，是否可以检测该蓝色标志牌？怎么检测？一个可能的办法是，可以设定(R,G,B)的范围，这个范围内的像素呈现蓝色，然后把不在这个范围内的像素置成黑色。如果不能检测，为什么？
	方法3. 如果使用HSV彩色空间，是否可以检测该蓝色标志牌？和方法2同样的思路，找到一个区间，去过滤像素。下图可以用来确定不同颜色对应H，S，V分量的区间。在HSV颜色空间里，是否可以检测？
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    image = cv2.imread('./images/blue_sign.png')
    # 对比演示
    gray = cv2. cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 显示灰度图
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image[:, :, ::-1])
    plt.title('RGB')
    plt.subplot(122)
    plt.xticks([]), plt.yticks([])
    plt.imshow(gray, cmap='gray')
    plt.title('gray')
    plt.show()

    cld = image.copy()
    cld = cv2.inRange(cld, np.array([155, 0, 0]), np.array([255, 150, 150]))
    cld = cv2.cvtColor(cld, cv2.COLOR_GRAY2BGR)
    plt.xticks([]), plt.yticks([])
    plt.imshow(cld)
    plt.title('blue')
    plt.show()

    # 选取蓝色区域
    hsv = hsv.astype(np.uint8)
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # 显示蓝色区域
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image[:, :, ::-1])
    plt.title('RGB')
    plt.subplot(122)
    plt.xticks([]), plt.yticks([])
    plt.imshow(mask, cmap='gray')
    plt.title('mask')
    plt.show()