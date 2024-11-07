import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('images/wall.png', cv2.IMREAD_GRAYSCALE)

# 顶帽
kernel = np.ones((20, 20), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# 黑帽
kernel = np.ones((20, 20), np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# plt展示
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(tophat, cmap='gray'), plt.title('TopHat')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(blackhat, cmap='gray'), plt.title('BlackHat')
plt.xticks([]), plt.yticks([])
plt.tight_layout(), plt.show()

