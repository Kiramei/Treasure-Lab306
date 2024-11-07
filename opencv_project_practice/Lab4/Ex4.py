import cv2
import numpy as np
import matplotlib.pyplot as plt


def P3_1(gray):
    # 计算图像直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # 计算灰度级总数和像素总数
    total_pixels = gray.shape[0] * gray.shape[1]
    gray_levels = np.arange(256)

    # 计算每个灰度级的概率
    probabilities = hist / total_pixels

    # 计算累积概率和灰度级均值
    cumulative_probabilities = np.cumsum(probabilities)
    responsive = gray_levels.reshape(256, 1) * probabilities
    gray_means = np.cumsum(responsive)
    global_mean = np.dot(gray_levels, probabilities)

    # 初始化最大类间方差和最佳阈值
    max_variance = 0
    best_threshold = 0

    # 遍历所有可能的阈值
    for threshold in gray_levels:
        # 计算类间方差
        omega0 = cumulative_probabilities[threshold]
        omega1 = 1 - omega0
        u0 = gray_means[threshold] / omega0 if omega0 > 0 else 0
        u1 = (global_mean - gray_means[threshold]) / omega1 if omega1 > 0 else 0
        variance = omega0 * omega1 * ((u0 - u1) ** 2)

        # 更新最大类间方差和最佳阈值
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold

    # 使用最佳阈值进行二值化
    _, img2 = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)

    # 绘制类间方差和分割阈值的变化曲线
    variances = []
    thresholds = []
    for threshold in gray_levels:
        omega0 = cumulative_probabilities[threshold]
        omega1 = 1 - omega0
        u0 = gray_means[threshold] / omega0 if omega0 > 0 else 0
        u1 = (global_mean - gray_means[threshold]) / omega1 if omega1 > 0 else 0
        variance = omega0 * omega1 * ((u0 - u1) ** 2)
        variances.append(variance)
        thresholds.append(threshold)

    # 显示类间方差和分割阈值的变化曲线
    plt.plot(thresholds, variances)
    plt.xlabel('Threshold')
    plt.ylabel('Inter-class Variance')
    plt.title('Otsu Method: Inter-class Variance vs Threshold')
    plt.show()

    return img2, best_threshold


# 读取图像
img = cv2.imread('./images/house.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('Original', img)
# cv2.waitKey(0)
img2, thd = P3_1(img)

plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])

# 使用自定义函数进行图像分割
plt.subplot(132), plt.imshow(img2, cmap='gray'), plt.title('Custom Result')
plt.xticks([]), plt.yticks([])

# 使用OpenCV函数进行图像分割
ret, img2_opencv = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.subplot(133), plt.imshow(img2_opencv, cmap='gray'), plt.title('OpenCV Result')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
# 比较自定义函数和OpenCV函数的结果
print('Custom Function Threshold:', thd)
print('OpenCV Function Threshold:', ret)

# 关闭窗口
cv2.destroyAllWindows()
