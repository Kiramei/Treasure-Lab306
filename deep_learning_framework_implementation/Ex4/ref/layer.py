# -*- coding: utf-8 -*-

from matrixslow.core import *
from matrixslow.ops import *
from ref.ops import Pooling_Convolve


def conv(feature_maps, input_shape, kernels, kernel_shape, activation):
    """
    :param feature_maps: 数组，包含多个输入特征图，它们应该是值为同形状的矩阵的节点
    :param input_shape: tuple ，包含输入特征图的形状（宽和高）
    :param kernels: 整数，卷积层的卷积核数量
    :param kernel_shape: tuple ，卷积核的形状（宽和高）
    :param activation: 激活函数类型
    :return: 数组，包含多个输出特征图，它们是值为同形状的矩阵的节点
    """
    # 与输入同形状的全 1 矩阵
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))

    outputs = []
    for i in range(kernels):

        channels = []
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            conv = Convolve(fm, kernel)
            channels.append(conv)

        channles = Add(*channels)
        bias = ScalarMultiply(Variable((1, 1), init=True, trainable=True), ones)
        affine = Add(channles, bias)

        if activation == "ReLU":
            outputs.append(ReLU(affine))
        elif activation == "Logistic":
            outputs.append(Logistic(affine))
        else:
            outputs.append(affine)

    assert len(outputs) == kernels
    return outputs


def Pooling_conv(feature_maps, kernels, kernel_shape, stride, model):
    outputs = []
    for i in range(kernels):

        channels = []
        for fm in feature_maps:
            if model == 'AveragePooling':
                kernel = Variable(kernel_shape, init=False, trainable=False)
                kernel.set_value(np.mat(np.ones(kernel_shape) / (kernel_shape[0] * kernel_shape[1])))
            else:
                kernel = Variable(kernel_shape, init=True, trainable=True)
            conv = Pooling_Convolve(fm, kernel, stride=stride)

        if model == 'MaxPooling':
            outputs.append(ReLU(conv))
        else:
            outputs.append(conv)
    assert len(outputs) == kernels
    return outputs


def FFT_Conv(feature_maps, input_shape, kernels, kernel_shape, activation):
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))
    outputs = []
    for i in range(kernels):
        channels = []
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            fft_kernel = np.fft.fft2(kernel.value)
            fft_fm = np.fft.fft2(fm.value)
            fft_conv = fft_kernel * fft_fm
            channels.append(np.fft.ifft2(fft_conv))
        channles = Add(*channels)
        bias = ScalarMultiply(Variable((1, 1), init=True, trainable=True), ones)
        affine = Add(channles, bias)

        if activation == "ReLU":
            outputs.append(ReLU(affine))
        elif activation == "Logistic":
            outputs.append(Logistic(affine))
        else:
            outputs.append(affine)

    assert len(outputs) == kernels
    return outputs


def pooling(feature_maps, kernel_shape, stride, model):
    """
    :param feature_maps: 数组，包含多个输入特征图，它们应该是值为同形状的矩阵的节点
    :param kernel_shape: tuple ，池化核（窗口）的形状（宽和高）
    :param stride: tuple ，包含横向和纵向步幅
    :return: 数组，包含多个输出特征图，它们是值为同形状的矩阵的节点
    """
    outputs = []
    for fm in feature_maps:
        if model == 'MaxPooling':
            outputs.append(MaxPooling(fm, size=kernel_shape, stride=stride))
        else:
            outputs.append(AveragePooling(fm, size=kernel_shape, stride=stride))

    return outputs


def fc(input, input_size, size, activation):
    """
    :param input: 输入向量
    :param input_size: 输入向量的维度
    :param size: 神经元个数，即输出个数（输出向量的维度）
    :param activation: 激活函数类型
    :return: 输出向量
    """
    weights = Variable((size, input_size), init=True, trainable=True)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)
    print(weights)
    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine
