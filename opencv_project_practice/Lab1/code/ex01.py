import cv2

def ex1():
    # 以灰度模式读取图像文件
    img = cv2.imread('./shenzhen_gray.bmp', cv2.IMREAD_GRAYSCALE)
    # 遍历图像的行
    for i in range(len(img[:, -1])):
        # 遍历图像的列
        for j in range(len(img[-1, :])):
            # 获取像素强度值
            p = img[i, j]
            # 根据像素强度应用不同的转换
            if p < 90:
                o = 0.2 * p
            elif p < 160:
                o = 3 * p
            else:
                o = 0.8 * p
            # 更新像素值为转换后的值
            img[i, j] = o
    # 显示修改后的图像
    cv2.imshow('实验 1', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    # 调用 ex1 函数
    ex1()