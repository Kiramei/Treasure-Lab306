import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取PCB.png
img = cv2.imread('images/PCB.png', cv2.IMREAD_COLOR)
src = img[..., 2]
# 二值化
ret, binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 去噪
kernel = np.ones((7, 7), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# 闭运算
kernel = np.ones((7, 7), np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# 轮廓检测
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 轮廓所有中心点计算

# 输出焊点个数
print('焊点个数:', len(contours))

cx, cy = [], []
for i in range(len(contours)):
    M = cv2.moments(contours[i])
    cx.append(int(M['m10'] / M['m00']))
    cy.append(int(M['m01'] / M['m00']))

# 绘制中心点
for i in range(len(cx)):
    cv2.circle(img, (cx[i], cy[i]), 5, (235, 123, 205), -1)
    print(f'第{i + 1}个点的位置:\t', cx[i], ',\t', cy[i], '')
plt.subplot(1, 1, 1), plt.imshow(img), plt.title('Original')
plt.show()
