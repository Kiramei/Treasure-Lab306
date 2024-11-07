"""
利用cv.findContours 产生star.png 的边界，编写函数P4_contour_code(pointList)
生成8领域归一化一阶差分链码, 函数的输入是边界点序列，给出图中所示目标的边界编码。

首先我们明确链码算法是一种起点开始，沿边界编码，至起点被重新碰到，结束一个对象的编码的算法。而归一化链码是将链码看作n位自然数，将该码按一个方向循环，使其构成的n位自然数最小，此时形成的唯一的链码，就是归一化链码。而一阶差分链码是对原链码进行差分操作得到的链码。那么归一化一阶差分链码则是一种先利用链码算法得到原链码，然后得到做差分，最后归一化得到的链码。
这样做的好处是：差分得到的链码具有旋转不变性，归一化的链码具有唯一性，这样形成的链码方便我们在图上快速找到图像的边界。

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def P4_contour_code(_pointList: np.ndarray) -> list:
    # 生成8领域归一化一阶差分链码
    # direction，需要注意的是，这里的方向是按照逆时针方向排列的
    # 而且长宽维度相反
    direction = np.array([
        [1, 0], [1, 1], [0, 1], [-1, 1],
        [-1, 0], [-1, -1], [0, -1], [1, -1]
    ])[:, ::-1]
    # code
    _codeList = []
    _pointList = np.concatenate((_pointList, _pointList[0:1]), axis=0)
    diff = np.diff(_pointList, axis=0)
    # 生成链码
    for i in range(len(diff)):
        # 计算方向编码
        code = np.where(np.all(diff[i] == direction, axis=1))[0][0]
        _codeList.append(code)
    # 生成差分链码
    _codeList = np.array(_codeList)
    # 生成归一化链码
    # 将序列转换为数字
    num_list = _codeList
    num_str = ''.join(map(str, num_list))
    num = int(num_str)
    ptr = num_str
    # 得到最小数的序列
    for i in range(len(num_str)):
        num_str = num_str[1:] + num_str[0]
        if int(num_str) < num:
            ptr = num_str
    return ptr


if __name__ == '__main__':
    # 加载图像
    image = cv2.imread('./images/star.png', cv2.IMREAD_GRAYSCALE)
    # 二值化处理
    _, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # 查找边界
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 获取第一个边界的点序列
    pointList = contours[0].reshape(-1, 2)

    # 画出边界
    image_contour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_contour, contours, -1, (255, 0, 255), 3)
    plt.xticks([]), plt.yticks([])
    plt.imshow(image_contour)
    plt.title('contour')
    plt.show()

    # # 生成边界编码
    codeList = P4_contour_code(pointList)

    print('最小一阶差分链码是：', codeList)
