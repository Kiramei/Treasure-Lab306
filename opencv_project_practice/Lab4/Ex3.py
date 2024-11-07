import cv2
import matplotlib.pyplot as plt

# 基于形态学的边缘提取
image = cv2.imread("./images/morph_test.png", cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
dilate_img = cv2.dilate(image, kernel)
dilate_img = cv2.erode(dilate_img, kernel)
result = cv2.subtract(dilate_img, image)

# show
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(result, cmap='gray'), plt.title('Result')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
