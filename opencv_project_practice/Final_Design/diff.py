import os.path

import matplotlib.pyplot as plt
from util import *


class DiffPatcher:
    def __init__(self):
        self.parameters = [0, 0]

    @staticmethod
    def detect_changes(_images):
        # 将图像转换为灰度图像
        gray_images = _images
        # 初始化前一帧图像
        prev_image = gray_images[0]
        # 初始化结果图像
        result = np.zeros_like(prev_image)
        for image in gray_images[1:]:
            # 计算当前帧与前一帧的差异图像
            diff = cv2.absdiff(prev_image, image)
            # 对差异图像进行阈值处理
            threshold = 50
            _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            # 使用逻辑或运算累积变化区域
            result = cv2.bitwise_or(result, thresholded)
            # 更新前一帧图像
            prev_image = image
        # 对累积的变化区域进行腐蚀和膨胀操作，以去除噪声和填充空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        return result

    def patch(self) -> None:
        if os.path.exists('./param.npy'):
            self.parameters = np.load('./param.npy')
            return

        # 读取多张图像
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in DIFF_ARRAY]
        # 检测变化区域
        changes = self.detect_changes(images)
        contours, _ = cv2.findContours(changes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制边界框
        changes = cv2.cvtColor(changes, cv2.COLOR_GRAY2BGR)
        # 只取最大的轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        x, y, w, h = cv2.boundingRect(contours[0])
        # 上下界轮廓外延100个像素
        y -= 200
        h += 400
        self.parameters[0] = y
        self.parameters[1] = h

    def test_patch(self):
        selected = [get_a_random_img() for _ in range(4)]
        for i in range(len(selected)):
            selected[i] = selected[i][self.parameters[0]:self.parameters[1] + self.parameters[0], :]
            selected[i] = rotate_to_horizon(selected[i])



        plt.subplot(2, 2, 1)
        plt.imshow(selected[0], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 2)
        plt.imshow(selected[1], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 3)
        plt.imshow(selected[2], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 4)
        plt.imshow(selected[3], cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def crop(self, origin):
        return origin[self.parameters[0]:self.parameters[1] + self.parameters[0]]


patcher = DiffPatcher()
patcher.patch()
# patcher.crop(get_a_random_img())
patcher.test_patch()
