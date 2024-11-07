import cv2
import numpy as np

# 加载图像
# image = cv2.imread('images/house_right.png', cv2.IMREAD_GRAYSCALE)
image = np.array([
    [3, 1, 3, 4, 1],
    [4, 2, 65, 4, 5],
    [3, 50, 2, 45, 1],
    [1, 5, 3, 2, 1]
],dtype=np.uint8)
# 均值滤波
mean_filtered = cv2.blur(image, (3, 3))

# 中值滤波
median_filtered = cv2.medianBlur(image, 3)

print(image)
print(mean_filtered)
print(median_filtered)
