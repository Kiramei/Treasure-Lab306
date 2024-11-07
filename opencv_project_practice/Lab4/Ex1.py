import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('images/house.jpg', cv2.IMREAD_GRAYSCALE)

# 双峰法二值化
th_1, ret_1 = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY)
th_2, ret_2 = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY_INV)
th_3, ret_3 = cv2.threshold(img, 105, 255, cv2.THRESH_TRUNC)
th_4, ret_4 = cv2.threshold(img, 105, 255, cv2.THRESH_TOZERO)

# plt展示
plt.subplot(2, 2, 1), plt.imshow(ret_1, cmap='gray'), plt.title('THRESH_BINARY')
plt.subplot(2, 2, 2), plt.imshow(ret_2, cmap='gray'), plt.title('THRESH_BINARY_INV')
plt.subplot(2, 2, 3), plt.imshow(ret_3, cmap='gray'), plt.title('THRESH_TRUNC')
plt.subplot(2, 2, 4), plt.imshow(ret_4, cmap='gray'), plt.title('THRESH_TOZERO')
plt.tight_layout(), plt.show()

# 灰度直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# 灰度累计直方图
cdf = hist.cumsum()
# 归一化
cdf_normalized = cdf * hist.max() / cdf.max()
# 画图
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()
