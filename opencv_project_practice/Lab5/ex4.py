"""
目的：在图像planes.png中定位出one_plane.png 所示目标的位置，并画出其bounding box。大致步骤如下
1)	查找planes.png中所有目标的轮廓
2)	对每一个轮廓求得其最小的外接矩形 (bounding box)，得到各个飞机的小图
3)	对one_plane.png 和步骤2中的每个小图计算Hu矩，并计算Hu矩的距离。Hu矩最相近的小图应该是目标所在。
4)	在planes.png 中画出矩形定位框。使用函数cv2.rectangle(img, pt1, pt2, 255, 2), 其中pt1为bounding box 的左上角坐标，pt2 为右下角坐标。

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def solve_hu(img):
    # 计算Hu矩
    M = cv2.moments(img)
    hu_moments = cv2.HuMoments(M)
    return hu_moments


def solution() -> None:
    # 读取图像
    image = cv2.imread('./images/planes.png')
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 读取one_plane.png
    one_plane = cv2.imread('./images/one_plane.png')
    # 转换为灰度图
    one_plane_gray = cv2.cvtColor(one_plane, cv2.COLOR_BGR2GRAY)
    # 显示原图
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image[:, :, ::-1])
    plt.title('original')
    plt.subplot(122)
    plt.xticks([]), plt.yticks([])
    plt.imshow(one_plane[:, :, ::-1])
    plt.title('one_plane')
    plt.show()

    # 二值化
    thresh, ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 画出轮廓
    image_contour = image.copy()
    cv2.drawContours(image_contour, contours, -1, (0, 0, 255), 2)
    # 得到小飞机
    image_plane = []
    for contour in contours:
        # 画出bounding box
        x, y, w, h = cv2.boundingRect(contour)
        image_plane.append(image[y:y + h, x:x + w])
    # 显示小飞机
    plt.figure(figsize=(4, 4))
    plt.subplot(221)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image_plane[0][:, :, ::-1])
    plt.title('plane_1')
    plt.subplot(222)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image_plane[1][:, :, ::-1])
    plt.title('plane_2')
    plt.subplot(223)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image_plane[2][:, :, ::-1])
    plt.title('plane_3')
    plt.subplot(224)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image_plane[3][:, :, ::-1])
    plt.title('plane_4')
    plt.show()

    # 对每一个小飞机计算Hu矩
    hu_moments = []
    for plane in image_plane:
        plane = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY)
        hu_moments.append(solve_hu(plane))
    # 计算one_plane.png的Hu矩
    one_plane_hu_moments = solve_hu(one_plane_gray)
    # 计算Hu矩距离
    hu_moments_distance = []
    for hu_moment in hu_moments:
        hu_moments_distance.append(np.abs(np.sum(np.abs(hu_moment - one_plane_hu_moments) / hu_moment)))
    # 找到距离最小的Hu矩
    min_index = np.argmin(hu_moments_distance)
    # 画出bounding box
    image_bbox = image.copy()
    x, y, w, h = cv2.boundingRect(contours[min_index])
    cv2.rectangle(image_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 显示bounding box
    plt.figure(figsize=(8, 8))
    plt.xticks([]), plt.yticks([])
    plt.imshow(image_bbox[:, :, ::-1])
    plt.title('bounding box')
    plt.show()


if __name__ == '__main__':
    solution()
