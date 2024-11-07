# -*- coding: utf-8 -*-

"""
Created on Wed Jun  5 15:23:01 2019

@author: zhangjuefei
"""
import math

import numpy as np

from matrixslow.core import Node


def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
           filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    '''
    定义操作符抽象类
    '''
    pass


class Add(Operator):
    """
    （多个）矩阵加法
    """

    def compute(self):
        # assert len(self.parents) == 2 and self.parents[0].shape() == self.parents[1].shape()
        self.value = np.mat(np.zeros(self.parents[0].shape()))

        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        return np.mat(np.eye(self.dimension()))  # 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵


class tanh(Operator):

    def compute(self):
        self.value = np.mat(2 / (1 + np.power(np.e, -2 * self.parents[0].value)) - 1)

    def get_jacobi(self, parent):
        # print((1 - np.power(self.value,2)).A1)
        return np.mat(np.diag((1 - np.power(self.value, 2)).A1))


class MatMul(Operator):
    """
    矩阵乘法
    """

    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[
            1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        """
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅克比矩阵。
        """

        # 很神秘，靠注释说不明白了
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Logistic(Operator):
    """
    对向量的分量施加Logistic函数
    """

    def compute(self):
        x = self.parents[0].value
        # 对父节点的每个分量施加Logistic
        self.value = np.mat(
            1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))

    def get_jacobi(self, parent):
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class ReLU(Operator):
    """
    对矩阵的元素施加ReLU函数
    """

    nslope = 0.1  # 负半轴的斜率

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value > 0.0,
            self.parents[0].value,
            self.nslope * self.parents[0].value)
        )

    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0, 1.0, self.nslope))


class SoftMax(Operator):
    """
    SoftMax函数
    """

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # 防止指数过大
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        我们不实现SoftMax节点的get_jacobi函数，
        训练时使用CrossEntropyWithSoftMax节点
        """
        raise NotImplementedError("Don't use SoftMax's get_jacobi")


class Reshape(Operator):
    """
    改变父节点的值（矩阵）的形状
    """

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.to_shape = kargs.get('shape')
        assert isinstance(self.to_shape, tuple) and len(self.to_shape) == 2

    def compute(self):
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))


class Multiply(Operator):
    """
    两个父节点的值是相同形状的矩阵，将它们对应位置的值相乘
    """

    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)


class Convolve(Operator):
    """
    以第二个父节点的值为滤波器，对第一个父节点的值做二维离散卷积
    """

    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)
        self.padded = None

    def compute(self):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw + w, hkh:hkh + h] = data

        self.value = np.mat(np.zeros((w, h)))

        # 二维离散卷积
        for i in np.arange(hkw, hkw + w):
            for j in np.arange(hkh, hkh + h):
                self.value[i - hkw, j - hkh] = np.sum(
                    np.multiply(self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh], kernel))

    def get_jacobi(self, parent):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        else:
            raise Exception("You're not my father")

        return np.mat(jacobi)


class fft_Convolve(Operator):
    """
    以第二个父节点的值为滤波器，对第一个父节点的值做二维离散卷积
    """

    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)

        self.padded = None

    def compute(self):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw + w, hkh:hkh + h] = data
        padded_fft = np.fft.fft2(data)
        kernel_fft = np.fft.fft2(kernel, s=data.shape)
        self.value = np.mat(np.abs(np.fft.ifft2(padded_fft * kernel_fft)))

    def get_jacobi(self, parent):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        else:
            raise Exception("You're not my father")

        return np.mat(jacobi)


class MaxPooling(Operator):
    """
    最大值池化
    """

    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.stride = kargs.get('stride')
        assert self.stride is not None
        self.stride = tuple(self.stride)
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert self.size is not None
        self.size = tuple(self.size)
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None

    def compute(self):
        data = self.parents[0].value  # 输入特征图
        w, h = data.shape  # 输入特征图的宽和高
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size  # 池化核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 池化核长宽的一半

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的最大值
                top, bottom = max(0, i), min(w, i + hkw + 1)
                left, right = max(0, j), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                if (window.shape != self.size):
                    continue
                row.append(
                    np.max(window)
                )

                # 记录最大值在原特征图中的位置
                pos = np.argmax(window)
                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)
            if row:
                result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):
        # assert parent is self.parents[0] and self.jacobi is not None
        return self.flag


class Concat(Operator):
    """
    将多个父节点的值连接成向量
    """

    def compute(self):
        assert len(self.parents) > 0

        # 将所有父节点矩阵展平并连接成一个向量
        self.value = np.concatenate(
            [p.value.flatten() for p in self.parents],
            axis=1
        ).T

    def get_jacobi(self, parent):
        assert parent in self.parents

        dimensions = [p.dimension() for p in self.parents]  # 各个父节点的元素数量
        pos = self.parents.index(parent)  # 当前是第几个父节点
        dimension = parent.dimension()  # 当前父节点的元素数量

        assert dimension == dimensions[pos]

        jacobi = np.mat(np.zeros((self.dimension(), dimension)))
        start_row = int(np.sum(dimensions[:pos]))
        jacobi[start_row:start_row + dimension,
        0:dimension] = np.eye(dimension)

        return jacobi


class ScalarMultiply(Operator):
    """
    用标量（1x1矩阵）数乘一个矩阵
    """

    def compute(self):
        assert self.parents[0].shape() == (1, 1)  # 第一个父节点是标量
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        assert parent in self.parents

        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]


class Step(Operator):

    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))

    def get_jacobi(self, parent):
        np.mat(np.eye(self.dimension()))
        return np.zeros(np.where(self.parents[0].value.A1 >= 0.0, 0.0, -1.0))


class Welding(Operator):

    def compute(self):
        assert len(self.parents) == 1 and self.parents[0] is not None
        self.value = self.parents[0].value

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))

    def weld(self, node):
        """
        将本节点焊接到输入节点上
        """

        # 首先与之前的父节点断开

        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)

        self.parents.clear()

        # 与输入节点焊接
        self.parents.append(node)
        node.children.append(self)


class haar_L1(Operator):
    def __init__(self, *parents, **kargs):
        Operator.__init__(self, *parents, **kargs)
        self.opera1 = np.mat([[1, 1], [1, 1]])
        self.opera2 = np.mat([[1, -1], [1, -1]])
        self.opera3 = np.mat([[1, 1], [-1, -1]])
        self.opera4 = np.mat([[1, -1], [-1, 1]])

    def compute(self):
        H, W = self.parents[0].shape()
        assert H % 2 == 0 and W % 2 == 0
        self.value = np.zeros(self.parents[0].shape())
        data = self.parents[0].value
        H_, W_ = H // 2, W // 2
        for i in range(H // 2):
            for j in range(W // 2):
                data_ = data[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
                self.value[i, j] = np.sum(np.multiply(data_, self.opera1)) / 4
                self.value[i, W_ + j] = np.sum(np.multiply(data_, self.opera2)) / 4
                self.value[i + H_, j] = np.sum(np.multiply(data_, self.opera3)) / 4
                self.value[H_ + i, W_ + j] = np.sum(np.multiply(data_, self.opera4)) / 4

    def get_jacobi(self, parent):
        jacobi = np.mat(np.zeros((self.dimension(), parent.dimension())))
        H, W = self.parents[0].shape()
        H_, W_ = H // 2, W // 2
        for i in np.arange(self.dimension()):
            tem = np.zeros(parent.shape())
            h, w = i % H, i // H
            if h < H_:
                if w < W_:
                    tem[h * 2:h * 2 + 2, w * 2:w * 2 + 2] = self.opera1 / 4
                else:
                    tem[h * 2:h * 2 + 2, (w - W_) * 2:(w - W_) * 2 + 2] = self.opera2 / 4
            else:
                if w < W_:
                    tem[(h - H_) * 2:(h - H_) * 2 + 2, w * 2:w * 2 + 2] = self.opera3 / 4
                else:
                    tem[(h - H_) * 2:(h - H_) * 2 + 2, (w - W_) * 2:(w - W_) * 2 + 2] = self.opera4 / 4
            jacobi[i, :] = tem.flatten()
        return jacobi


class AveragePooling(Operator):
    def __init__(self, *parent, **kargs):
        Operator.__init__(self, *parent, **kargs)

        self.stride = kargs.get('stride')
        assert self.stride is not None
        self.stride = tuple(self.stride)
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert self.size is not None
        self.size = tuple(self.size)
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None

    def compute(self):
        data = self.parents[0].value  # 输入特征图
        w, h = data.shape  # 输入特征图的宽和高
        sw, sh = self.stride
        kw, kh = self.size  # 池化核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 池化核长宽的一半

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的最大值
                top, bottom = max(0, i), min(w, i + hkw + 1)
                left, right = max(0, j), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                if (window.shape != self.size):
                    continue
                row.append(
                    np.average(window)
                )

                tmp = np.zeros((w, h))
                tmp[top:bottom, left:right] = 1 / ((bottom - top) * (right - left))
                flag.append(tmp.flatten())
            if row:
                result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):
        # assert parent is self.parents[0] and self.jacobi is not None
        return self.flag


class Pooling_Convolve(Operator):
    """
    以第二个父节点的值为滤波器，对第一个父节点的值做二维离散卷积
    """

    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)
        self.stride = kargs.get('stride')
        self.padded = None

    def compute(self):
        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器
        sw, sh = self.stride
        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        self.value = np.zeros((int((w - kw) / sw + 1), int((h - kh) / sh + 1)))
        self.flag1 = []
        self.flag2 = []
        # 二维离散卷积
        for i_, i in enumerate(np.arange(0, w, sw)):
            for j_, j in enumerate(np.arange(0, h, sh)):
                top, bottom = max(0, i), min(w, i + hkw + 1)
                left, right = max(0, j), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                if (window.shape != kernel.shape):
                    continue
                self.value[i_, j_] = np.sum(np.multiply(window, kernel))
                tmp = np.zeros((w, h))
                tmp[top:bottom, left:right] = 1 / ((bottom - top) * (right - left))
                self.flag1.append(tmp.flatten())
                self.flag2.append(window.flatten)
        self.value = np.matrix(self.value)
        self.flag1 = np.matrix(self.flag1)
        self.flag2 = np.matrix(self.flag2)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return self.flag1
        elif parent is self.parents[1]:
            return self.flag2

        else:
            raise Exception("You're not my father")


class TransposeConvolution(Operator):
    def __init__(self, *parents, **kwargs):
        self.jacobi_kernel, self.jacobi_input = [], []
        assert len(parents) == 2, "Two parent nodes are required"
        super().__init__(*parents, **kwargs)
        self.stride = kwargs.get('stride')
        assert self.stride is not None, "Stride must be provided"

    def compute(self):
        input_data = np.array(self.parents[0].value)  # Input image
        kernel = np.array(self.parents[1].value)  # Filter
        stride_w, stride_h = self.stride  # Stride dimensions

        input_w, input_h = input_data.shape  # Input image dimensions
        kernel_w, kernel_h = kernel.shape  # Kernel dimensions

        output_w = stride_w * (input_w - 1) + kernel_w  # Output width
        output_h = stride_h * (input_h - 1) + kernel_h  # Output height

        padding_w, padding_h = kernel_w - 1, kernel_h - 1  # Padding size
        dilation_w, dilation_h = stride_w - 1, stride_h - 1  # Dilation size

        expanded_w = input_w + 2 * padding_w + (input_w - 1) * dilation_w  # Expanded width
        expanded_h = input_h + 2 * padding_h + (input_h - 1) * dilation_h  # Expanded height

        self.value = np.zeros((output_w, output_h))
        expanded_data = np.zeros((expanded_w, expanded_h))
        for i in range(input_w):
            for j in range(input_h):
                expanded_data[padding_w + i * (dilation_w + 1), padding_h + j * (dilation_h + 1)] = input_data[i, j]

        for output_i, input_i in enumerate(range(0, expanded_w)):
            for output_j, input_j in enumerate(range(0, expanded_h)):
                top, bottom = max(0, input_i), min(expanded_w, input_i + kernel_w)
                left, right = max(0, input_j), min(expanded_h, input_j + kernel_h)
                window = expanded_data[top:bottom, left:right]

                if window.shape != kernel.shape: continue

                self.value[output_i, output_j] = np.sum(np.multiply(window, kernel))
                self.jacobi_kernel.append(window.flatten())

                jacobi_input_matrix = np.zeros((input_w, input_h))
                for win_i in range(input_w):
                    for win_j in range(input_h):
                        if top <= padding_w + win_i * (dilation_w + 1) < bottom and left <= padding_h + win_j * (
                                dilation_h + 1) < right:
                            jacobi_input_matrix[win_i, win_j] = kernel[
                                padding_w + win_i * (dilation_w + 1) - top, padding_h + win_j * (dilation_h + 1) - left]

                self.jacobi_input.append(jacobi_input_matrix.flatten())

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.matrix(self.jacobi_input)
        elif parent is self.parents[1]:
            return np.matrix(self.jacobi_kernel)
        else:
            raise ValueError("The specified parent is not recognized")


# class Transpose_Convolve(Operator):
#     def __init__(self, *parents, **kargs):
#         super().__init__()
#         assert len(parents) == 2
#         Operator.__init__(self, *parents, **kargs)
#         self.stride = kargs.get('stride')
#
#     def forward(self):
#         data = np.array(self.parents[0].value)  # 图像
#         kernel = np.array(self.parents[1].value)  # 滤波器
#         sw, sh = self.stride  # 步长
#         w, h = data.shape  # 图像的宽和高
#         kw, kh = kernel.shape  # 滤波器尺寸
#         w_, h_ = sw * (w - 1) + kw, sh * (h - 1) + kh  # 卷积后输出的大小
#         pw, ph = kw - 1, kh - 1  # 输入图像需要添加的padding
#         sw_, sh_ = sw - 1, sh - 1  # 输入图像相邻像素插入的空洞
#         _w, _h = w + 2 * pw + (w - 1) * sw_, h + 2 * ph + (h - 1) * sh_  # 扩张后图片大小
#         self.value = np.zeros((w_, h_))
#         data_ = np.zeros((_w, _h))
#         self.flag1 = []
#         self.flag2 = []
#         for i in range(w):
#             for j in range(h):
#                 data_[pw + i * (sw_ + 1)][ph + j * (sh_ + 1)] = data[i][j]
#
#         for i_, i in enumerate(np.arange(0, _w)):
#             for j_, j in enumerate(np.arange(0, _h)):
#                 top, bottom = max(0, i), min(_w, i + kw)
#                 left, right = max(0, j), min(_h, j + kh)
#                 window = data_[top:bottom, left:right]
#                 if (window.shape != kernel.shape):
#                     continue
#                 self.value[i_, j_] = np.sum(np.multiply(window, kernel))
#                 self.flag2.append(window.flatten())
#                 tem = np.zeros((w, h))
#                 for _i in range(w):
#                     for _j in range(h):
#                         if top <= pw + _i * (sw_ + 1) < bottom and left <= ph + _j * (sh_ + 1) < right:
#                             tem[_i, _j] = kernel[pw + _i * (sw_ + 1) - top][ph + _j * (sh_ + 1) - left]
#                 self.flag1.append(tem.flatten())
#
#     def get_jacobi(self, parent):
#         if parent is self.parents[0]:
#             return np.matrix(self.flag1)
#         elif parent is self.parents[1]:
#             return np.matrix(self.flag2)
#         else:
#             raise Exception("You're not my father")


# class Circle_Convolve(Operator):
#     def __init__(self, *parents, **kargs):
#         assert len(parents) == 2
#         Operator.__init__(self, *parents, **kargs)
#         self.stride = kargs.get('stride')
#         self.flag1 = []
#         self.flag2 = []
#
#     def forward(self):
#         data = np.array(self.parents[0].value)  # 图像
#         kernel = np.array(self.parents[1].value)  # 滤波器
#         w, h = data.shape  # 图像的宽和高
#         kw, kh = kernel.shape  # 滤波器尺寸
#         self.r = kw // 2
#         sw, sh = self.stride  # 步长
#         self.value = np.zeros((int((w - kw) / sw + 1), int((h - kh) / sh + 1)))
#         hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半
#         idx = self.get_idx(kernel.shape)
#         for i_, i in enumerate(np.arange(hkw, w - hkw, sw)):
#             for j_, j in enumerate(np.arange(hkh, h - hkh, sh)):
#                 window = np.zeros(kernel.shape)
#                 des_x, des_y = i - hkw, j - hkw
#                 tem_idx = idx.copy()
#                 tem_idx[:, :, 0] = idx[:, :, 0] + des_x
#                 tem_idx[:, :, 1] = idx[:, :, 1] + des_y
#                 tem_1 = np.zeros(data.shape)
#                 for tem_i in range(kw):
#                     for tem_j in range(kh):
#                         up_w, up_h = math.ceil(tem_idx[tem_i, tem_j][0]), math.ceil(tem_idx[tem_i, tem_j][1])
#                         down_w, down_h = math.floor(tem_idx[tem_i, tem_j][0]), math.floor(tem_idx[tem_i, tem_j][1])
#                         window[tem_i, tem_j] = (tem_idx[tem_i, tem_j][0] - down_w + tem_idx[tem_i, tem_j][
#                             1] - down_h) / 2 * data[down_w, down_h]
#                         window[tem_i, tem_j] += (-tem_idx[tem_i, tem_j][0] + up_w - tem_idx[tem_i, tem_j][
#                             1] + up_h) / 2 * data[up_w, up_h]
#                         tem_1[up_w, up_h] += (-tem_idx[tem_i, tem_j][0] + up_w - tem_idx[tem_i, tem_j][
#                             1] + up_h) / 2 * kernel[tem_i][tem_j]
#                         tem_1[down_w, down_h] += (tem_idx[tem_i, tem_j][0] - down_w + tem_idx[tem_i, tem_j][
#                             1] - down_h) / 2 * kernel[tem_i][tem_j]
#                 self.value[i_, j_] = np.sum(np.multiply(window, kernel))
#                 self.flag2.append(window.flatten())
#                 self.flag1.append(tem_1.flatten())
#
#     def get_jacobi(self, parent):
#         if parent is self.parents[0]:
#             return np.matrix(self.flag1)
#         elif parent is self.parents[1]:
#             return np.matrix(self.flag2)
#         else:
#             raise Exception("You're not my father")
#
#     def get_idx(self, kernel_shape):
#         assert kernel_shape[0] == kernel_shape[1]
#         center_x, center_y = kernel_shape[0] // 2, kernel_shape[1] // 2
#         w = kernel_shape[0] * kernel_shape[1] - 1
#         m = (2 * np.pi) / w
#         idx_ = np.zeros((kernel_shape[0], kernel_shape[1], 2))
#         idx = []
#         for i in range(0, w):
#             x = center_x + self.r * np.math.sin(m * i)
#             y = center_y + self.r * np.math.cos(m * i)
#             idx.append(np.array([x, y]))
#         idx.append(np.array([center_x, center_y]))
#         idx = np.array(idx)
#         x_sort = np.argsort(idx[:, 0])
#         for i in range(kernel_shape[0]):
#             x_tem = idx[x_sort[i * kernel_shape[0]:(i + 1) * kernel_shape[0]]]
#             y_sort = np.argsort(x_tem[:, 1])
#             idx_[:, i] = x_tem[y_sort]
#         return idx_


class CircleConvolution(Operator):
    def __init__(self, *parents, **kwargs):
        assert len(parents) == 2, "Two parent nodes are required"
        super().__init__(*parents, **kwargs)
        self.stride = kwargs.get('stride')
        assert self.stride is not None, "Stride must be provided"
        self.jacobi_input = []
        self.jacobi_kernel = []

    def compute(self):
        input_data = np.array(self.parents[0].value)
        kernel = np.array(self.parents[1].value)
        input_height, input_width = input_data.shape
        kernel_height, kernel_width = kernel.shape
        self.radius = kernel_width // 2
        stride_width, stride_height = self.stride

        output_height = (input_height - kernel_height) // stride_height + 1
        output_width = (input_width - kernel_width) // stride_width + 1
        self.value = np.zeros((output_height, output_width))

        half_kernel_width, half_kernel_height = kernel_width // 2, kernel_height // 2
        index_matrix = self._generate_index_matrix(kernel.shape)

        for output_row, row in enumerate(range(half_kernel_height, input_height - half_kernel_height, stride_height)):
            for output_col, col in enumerate(range(half_kernel_width, input_width - half_kernel_width, stride_width)):
                window = np.zeros(kernel.shape)
                adjusted_index_matrix = index_matrix.copy()
                adjusted_index_matrix[:, :, 0] += row - half_kernel_height
                adjusted_index_matrix[:, :, 1] += col - half_kernel_width

                input_gradient = np.zeros(input_data.shape)

                for kernel_row in range(kernel_height):
                    for kernel_col in range(kernel_width):
                        up_row, up_col = np.ceil(adjusted_index_matrix[kernel_row, kernel_col]).astype(np.int64)
                        down_row, down_col = np.floor(adjusted_index_matrix[kernel_row, kernel_col]).astype(np.int64)
                        interpolation = ((adjusted_index_matrix[kernel_row, kernel_col, 0] - down_row +
                                          adjusted_index_matrix[kernel_row, kernel_col, 1] - down_col) / 2)
                        window[kernel_row, kernel_col] = interpolation * input_data[down_row, down_col]
                        window[kernel_row, kernel_col] += ((-adjusted_index_matrix[kernel_row, kernel_col, 0] + up_row -
                                                            adjusted_index_matrix[
                                                                kernel_row, kernel_col, 1] + up_col) / 2) * input_data[
                                                              up_row, up_col]

                        input_gradient[up_row, up_col] += ((-adjusted_index_matrix[kernel_row, kernel_col, 0] + up_row -
                                                            adjusted_index_matrix[
                                                                kernel_row, kernel_col, 1] + up_col) / 2) * kernel[
                                                              kernel_row, kernel_col]
                        input_gradient[down_row, down_col] += (interpolation * kernel[kernel_row, kernel_col])

                self.value[output_row, output_col] = np.sum(np.multiply(window, kernel))
                self.jacobi_kernel.append(window.flatten())
                self.jacobi_input.append(input_gradient.flatten())

    def _generate_index_matrix(self, kernel_shape):
        assert kernel_shape[0] == kernel_shape[1], "Kernel shape must be square"
        center_x, center_y = kernel_shape[0] // 2, kernel_shape[1] // 2
        num_points = kernel_shape[0] * kernel_shape[1] - 1
        angle_increment = (2 * np.pi) / num_points

        index_matrix = np.zeros((kernel_shape[0], kernel_shape[1], 2))
        indices = []

        for i in range(num_points):
            x = center_x + self.radius * np.sin(angle_increment * i)
            y = center_y + self.radius * np.cos(angle_increment * i)
            indices.append([x, y])

        indices.append([center_x, center_y])
        indices = np.array(indices)

        sorted_indices = np.argsort(indices[:, 0])
        for i in range(kernel_shape[0]):
            x_sorted = indices[sorted_indices[i * kernel_shape[0]:(i + 1) * kernel_shape[0]]]
            y_sorted = np.argsort(x_sorted[:, 1])
            index_matrix[:, i] = x_sorted[y_sorted]

        return index_matrix

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.matrix(self.jacobi_input)
        elif parent is self.parents[1]:
            return np.matrix(self.jacobi_kernel)
        else:
            raise ValueError("The specified parent is not recognized")


if __name__ == '__main__':

    center_x, center_y = 1, 1
    w = 8
    r = 1
    m = (2 * np.pi) / w
    idx = []
    for i in range(0, w + 1):
        x = center_x + r * np.math.sin(m * i)
        y = center_y + r * np.math.cos(m * i)
        idx.append(np.array([x, y]))
    idx.append(np.array([center_x, center_y]))
    idx = np.array(idx)
    x_sort = np.argsort(idx[:, 0])
    print(idx[x_sort[0:3]])

    # def circle_kernel(shape=(3, 3)):
    #     r, c = shape
    #
    #
    #     if r % 2 == 0:
    #         y_axis = np.concatenate([-(np.arange(r / 2) + 1)[::-1], np.arange(r / 2) + 1])
    #     else:
    #         y_axis = (np.arange(r) - r // 2)
    #     if c % 2 == 0:
    #         x_axis = np.concatenate([-(np.arange(c / 2) + 1)[::-1], np.arange(c / 2) + 1])
    #     else:
    #         x_axis = np.arange(c) - c // 2
    #     axis = np.meshgrid(np.abs(x_axis), np.abs(y_axis))  # 生成x,y坐标系
    #     radius_y = r / 2 - 0.5  # 生成椭圆在两个方向上的半径
    #     radius_x = c / 2 - 0.5
    #     return np.mat(
    #         ((axis[0] ** 2 / radius_x ** 2 + axis[1] ** 2 / radius_y ** 2) <= 1).astype(float))  # 根据椭圆的方程式,在椭圆内的为1,在椭圆外的为0
