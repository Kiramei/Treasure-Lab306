import cv2
import numpy as np


def P3_2_CE_Binarize(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算灰度直方图
    hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    # 计算灰度概率分布
    whole_prob = hist / np.sum(hist)
    # 计算灰度累积分布
    cum_prob = np.cumsum(whole_prob)
    # 初始化最小交叉熵和最佳阈值
    min_entropy = float('inf')
    best_threshold = 0
    # 遍历所有可能的阈值
    for threshold in range(256):
        # 计算前景和背景灰度概率分布
        fore_prob = whole_prob[:threshold]
        back_prob = whole_prob[threshold:]
        # 计算前景和背景交叉熵
        fore_ent = -np.sum(fore_prob * np.log2(fore_prob)) if np.sum(
            fore_prob) > 0 else 0
        back_ent = -np.sum(back_prob * np.log2(back_prob)) if np.sum(
            back_prob) > 0 else 0
        # 计算交叉熵
        entropy = fore_ent + back_ent

        # 更新最小交叉熵和最佳阈值
        if entropy < min_entropy:
            min_entropy = entropy
            best_threshold = threshold
    print('best_threshold:', best_threshold)
    # 使用最佳阈值进行二值化
    _, img2 = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)

    return img2


# 读取图像
img = cv2.imread('images/house.jpg')

# 使用自定义函数进行图像二值化
binarized_img = P3_2_CE_Binarize(img)
# cv2.imshow('Custom Function: Binarized Image', binarized_img)
# cv2.waitKey(0)

# plt 对比
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(binarized_img, cmap='gray'), plt.title('Binarized')
plt.xticks([]), plt.yticks([])
plt.tight_layout(), plt.show()

# 关闭窗口
# cv2.destroyAllWindows()
