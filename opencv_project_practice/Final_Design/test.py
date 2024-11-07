import cv2
import numpy as np
import matplotlib.pyplot as plt
i1 = cv2.imread('./data_test_tube/OK/OK_0001.bmp', cv2.IMREAD_GRAYSCALE)
i2 = cv2.imread('./data_test_tube/OK/OK_0002.bmp', cv2.IMREAD_GRAYSCALE)
i3 = cv2.imread('./data_test_tube/NG/daqipao/da_qipao_0002.bmp', cv2.IMREAD_GRAYSCALE)
# print(np.sqrt(((np.abs(i1 - i3) / 255) ** 2).sum()))
# print(np.sqrt(((np.abs(i1 - i2) / 255) ** 2).sum()))

# i1 = cv2.pyrDown(i1)
# i1 = cv2.pyrDown(i1)
# i1 = cv2.pyrDown(i1)
# i2 = cv2.pyrDown(i2)
# i2 = cv2.pyrDown(i2)
# i2 = cv2.pyrDown(i2)
# id = np.abs(i1 - i2)
# cv2.imshow('i1', i1)
# 画出灰度直方图
# plt.hist(i1.ravel(), 256, [0, 256])
# plt.hist(i2.ravel(), 256, [0, 256])
# plt.hist(i3.ravel(), 256, [0, 256])

# canny
# i1 = cv2.Canny(i1, 100, 200)

plt.imshow(i1, cmap='gray')

plt.show()
