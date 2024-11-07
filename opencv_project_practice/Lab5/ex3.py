# 计算star.png所示目标的Hu矩，并验证Hu矩具有平移，旋转，尺度不变性。
import numpy as np
import cv2
import matplotlib.pyplot as plt


def P5_hu_moment(img):
    # 计算Hu矩
    M = cv2.moments(img)
    hu_moments = cv2.HuMoments(M)
    # 对Hu矩取对数
    for i in range(7):
        hu_moments[i] = (-1 * np.sign(hu_moments[i])
                         * np.log10(np.abs(hu_moments[i])))
    return hu_moments


if __name__ == '__main__':
    image = cv2.imread('./images/star.png')
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算原图Hu矩，并且输出
    hu_moments_1 = P5_hu_moment(gray)
    print(hu_moments_1)
    # 旋转图像
    rows, cols = gray.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 50, 1)
    dst_1 = cv2.warpAffine(gray, M, (cols, rows))
    hu_moments_2 = P5_hu_moment(dst_1)
    print('平旋转Hu矩误差值：', np.sum(np.abs(hu_moments_1 - hu_moments_2)/hu_moments_1))
    # 缩放图像
    dst_2 = cv2.resize(gray, (int(cols * 0.5), int(rows * 0.5)))
    hu_moments_3 = P5_hu_moment(dst_2)
    print('缩放Hu矩误差值：', np.sum(np.abs(hu_moments_1 - hu_moments_3)/hu_moments_1))
    # 平移图像
    M = np.float32([[1, 0, 50], [0, 1, 20]])
    dst_3 = cv2.warpAffine(gray, M, (cols, rows))
    hu_moments_4 = P5_hu_moment(dst_3)
    # 输出差值
    print('平移Hu矩误差值：', np.sum(np.abs(hu_moments_1 - hu_moments_4)/hu_moments_1))

    # 显示旋转后的图像
    plt.figure(figsize=(4, 4))
    plt.subplot(221)
    plt.xticks([]), plt.yticks([])
    plt.imshow(gray, cmap='gray')
    plt.title('original')
    plt.subplot(222)
    plt.xticks([]), plt.yticks([])
    plt.imshow(dst_1, cmap='gray')
    plt.title('rotated')
    plt.subplot(223)
    plt.xticks([]), plt.yticks([])
    plt.imshow(dst_2, cmap='gray')
    plt.title('scaled')
    plt.subplot(224)
    plt.xticks([]), plt.yticks([])
    plt.imshow(dst_3, cmap='gray')
    plt.title('translated')

    plt.show()
