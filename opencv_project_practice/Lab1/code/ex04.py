import cv2


def ex4():
    # 以灰度模式读取矩形图像和圆形图像
    rectangle = cv2.imread('./rectangle.bmp', cv2.IMREAD_GRAYSCALE)
    circle = cv2.imread('./circle.bmp', cv2.IMREAD_GRAYSCALE)

    # 对两个图像进行按位与和按位异或操作
    ret_1 = cv2.bitwise_and(rectangle, circle)
    ret_2 = cv2.bitwise_xor(rectangle, circle)

    # 显示处理后的图像
    cv2.imshow('Experiment 4-a', ret_1)
    cv2.imshow('Experiment 4-b', ret_2)
    cv2.waitKey(0)


if __name__ == '__main__':
    # 调用ex4函数
    ex4()