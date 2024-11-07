import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def cal(img):
    # 1. 读取图像文件，显示图像的尺寸和通道数
    _img = cv2.imread(img)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    # 图像锐化
    kernel = np.array([[0, -1, 0], [-1, 10, -1], [0, -1, 0]])
    img = cv2.filter2D(_img, -1, kernel)
    # 腐蚀
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    ret, img = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    # 二值化
    ret, img = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY)
    # 获取所有白点，计算中心点
    points = np.where(img == 255)
    center_x = int(np.mean(points[1]))
    center_y = int(np.mean(points[0]))

    pat = cv2.bitwise_and(_img, img)
    # 高斯模糊
    # 提取图像中的白色药片位置
    pat = _img
    # 二值化，二分法
    _, pat = cv2.threshold(pat, 150, 255, cv2.THRESH_BINARY)
    # 腐蚀
    kernel = np.ones((9, 9), np.uint8)
    pat = cv2.erode(pat, kernel, iterations=1)
    # 膨胀
    kernel = np.ones((7, 7), np.uint8)
    pat = cv2.dilate(pat, kernel, iterations=1)
    # 锐化
    kernel = np.ones((3, 3), np.uint8)
    pat = cv2.erode(pat, kernel, iterations=1)
    # 降采样
    pat = cv2.pyrDown(pat)
    kernel = np.ones((3, 3), np.uint8)
    pat = cv2.erode(pat, kernel, iterations=1)
    pat = cv2.pyrDown(pat)
    pat = cv2.dilate(pat, kernel, iterations=1)

    pat = cv2.pyrDown(pat)
    pat = cv2.erode(pat, kernel, iterations=1)
    pat = cv2.pyrDown(pat)
    pat = cv2.dilate(pat, kernel, iterations=1)
    pat = cv2.pyrDown(pat)

    # 二值化
    ret, pat = cv2.threshold(pat, 0, 255, cv2.THRESH_OTSU)
    # 升采样
    pat = cv2.pyrUp(pat)
    pat = cv2.pyrUp(pat)
    pat = cv2.pyrUp(pat)
    pat = cv2.pyrUp(pat)
    pat = cv2.pyrUp(pat)
    ret, pat = cv2.threshold(pat, 0, 255, cv2.THRESH_OTSU)

    # 计算中心点
    contours, hierarchy = cv2.findContours(pat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = [], []
    for _i in range(len(contours)):
        M = cv2.moments(contours[_i])
        cx.append(int(M['m10'] / M['m00']))
        cy.append(int(M['m01'] / M['m00']))
    # 绘制中心点
    for _i in range(len(cx)):
        cv2.circle(pat, (cx[_i], cy[_i]), 5, (235, 123, 205), -1)

    # 中心点的重心
    M = cv2.moments(pat)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx / center_x > 1


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['kaiti']
    pr = os.listdir('./images/medicine')
    # 大标题, 防止乱码
    plt.suptitle('薬剤ディスクの配置方向検出方法使用例', fontsize=20, fontweight='bold')
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()

    for ind, i in enumerate(pr):
        jud = cal('./images/medicine/' + i)
        # print(i, '正常' if jud else '异常')
        plt.subplot(4, 4, ind+1)
        plt.axis('off')
        plt.xticks([]), plt.yticks([])
        plt.title('●', color='green' if jud else 'red')
        plt.imshow(cv2.imread('./images/medicine/' + i))

    plt.tight_layout()
    plt.show()
