# -*- coding: utf-8 -*-
from ..core import *
from ..ops import *


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

        channles = Add(*channels) #传入元组channels
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

def fft_conv(feature_maps, input_shape, kernels, kernel_shape, activation):
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
            # 使用 numpy.fft 计算卷积
            fm_fft = np.fft.fft2(fm.value)
            kernel_fft = np.fft.fft2(kernel.value, s=fm.value.shape)
            conv = np.real(np.fft.ifft2(fm_fft * kernel_fft))
            channels.append(Variable(conv.shape, init=False, trainable=False))
            channels[-1].set_value(np.mat(conv))

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


def pooling(feature_maps, kernel_shape, stride):
    """
    :param feature_maps: 数组，包含多个输入特征图，它们应该是值为同形状的矩阵的节点
    :param kernel_shape: tuple ，池化核（窗口）的形状（宽和高）
    :param stride: tuple ，包含横向和纵向步幅
    :return: 数组，包含多个输出特征图，它们是值为同形状的矩阵的节点
    """
    outputs = []
    for fm in feature_maps:
        outputs.append(MaxPooling(fm, size=kernel_shape, stride=stride))

    return outputs


def fc(input, input_size, output_size, activation):
    """
    :param input: 输入向量
    :param input_size: 输入向量的维度
    :param size: 神经元个数，即输出个数（输出向量的维度）
    :param activation: 激活函数类型
    :return: 输出向量
    """
    weights = Variable((output_size, input_size), init=True, trainable=True)
    bias = Variable((output_size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine
