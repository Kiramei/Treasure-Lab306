import numpy as np

from matrixslow import Variable
from matrixslow.ops import Operator, ReLU


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

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的最大值
                top, bottom = max(0, i), min(w, i + sw)
                left, right = max(0, j), min(h, j + sh)
                window = data[top:bottom, left:right]
                row.append(np.max(window))

                # 记录最大值在原特征图中的位置
                pos = np.argmax(window)
                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(np.array(result))

    def get_jacobi(self, parent):
        # assert parent is self.parents[0] and self.jacobi is not None
        return self.flag


class AveragePooling(Operator):
    """
    平均值池化
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
        sw, sh = self.stride
        kw, kh = self.size  # 池化核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 池化核长宽的一半

        result = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的平均值
                top, bottom = max(0, i), min(w, i + sw)
                left, right = max(0, j), min(h, j + sh)
                window = data[top:bottom, left:right]
                row.append(np.mean(window))

            result.append(row)

        self.value = np.mat(np.array(result))

    def get_jacobi(self, parent):
        # assert parent is self.parents[0] and self.jacobi is not None
        w, h = self.parents[0].value.shape
        sw, sh = self.stride
        kw, kh = self.size
        hkw, hkh = int(kw / 2), int(kh / 2)  # 池化核长宽的一半
        jacobi = np.zeros((self.value.size, w * h))

        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                top, bottom = max(0, i * sw), min(w, i * sw + sw)
                left, right = max(0, j * sh), min(h, j * sh + sh)
                window_size = (bottom - top) * (right - left)
                for x in range(top, bottom):
                    for y in range(left, right):
                        jacobi[i * self.value.shape[1] + j, x * h + y] = 1 / window_size

        return jacobi


class Pooling_Convolve(Operator):
    def __init__(self, *parents, **kargs):
        self.sampleWindowSet = []
        self.windowSet = []
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)
        self.stride = kargs.get('stride')

    def compute(self):
        inputData = self.parents[0].value
        kernel = self.parents[1].value
        strideWidth, strideHeight = self.stride
        dataWidth, dataHeight = inputData.shape
        kernelWidth, kernelHeight = kernel.shape
        outputWidth = int((dataWidth - kernelWidth) / strideWidth + 1)
        outputHeight = int((dataHeight - kernelHeight) / strideHeight + 1)
        self.value = np.zeros((outputWidth, outputHeight))
        hkw, hkh = int(kernelWidth / 2), int(kernelHeight / 2)  # 滤波器长宽的一半
        for indexWidth, sampledWidth in enumerate(list(np.arange(0, dataWidth, strideWidth))):
            for indexHeight, sampledHeight in enumerate(list(np.arange(0, dataHeight, strideHeight))):
                recTop, recBottom = max(0, sampledWidth), min(dataWidth, sampledWidth + hkw + 1)
                recLeft, recRight = max(0, sampledHeight), min(dataHeight, sampledHeight + hkh + 1)
                recData = inputData[recTop:recBottom, recLeft:recRight]
                if recData.shape != kernel.shape: continue
                self.value[indexWidth, indexHeight] = np.multiply(recData, kernel).sum()
                sampleWindow = np.zeros((dataWidth, dataHeight))
                sampleWindow[recTop:recBottom, recLeft:recRight] = 1 / ((recBottom - recTop) * (recRight - recLeft))
                self.sampleWindowSet.append(sampleWindow.flatten())
                self.windowSet.append(recData.flatten)
        self.value = np.matrix(self.value)
        self.sampleWindowSet, self.windowSet = np.matrix(self.sampleWindowSet), np.matrix(self.windowSet)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return self.sampleWindowSet
        elif parent is self.parents[1]:
            return self.windowSet
        else:
            raise ValueError("The specified parent is not recognized")


def pool4conv(featMaps, kernels, kernel_shape, stride, mode):
    outputs = []
    for kernel in range(kernels):
        channels, conv = [], None
        for feat in featMaps:
            if mode == 'AveragePooling':
                kernel = Variable(kernel_shape, init=False, trainable=False)
                kernel.set_value(np.mat(np.ones(kernel_shape) / (kernel_shape[0] * kernel_shape[1])))
            else:
                kernel = Variable(kernel_shape, init=True, trainable=True)
            conv = Pooling_Convolve(feat, kernel, stride=stride)
        outputs.append(ReLU(conv) if mode == 'MaxPooling' else conv)
    assert len(outputs) == kernels
    return outputs


def pooling(featMaps, kernel_shape, stride, mode):
    return [eval(mode)(feat, size=kernel_shape, stride=stride) for feat in featMaps]


# 随机生成一个 4x4 的矩阵
data = np.random.randint(1, 10, 16)

mode = 'AveragePooling'

inputShape = (4, 4)
inputMat = np.mat(data.reshape(inputShape))
Input1 = [Variable(dim=inputShape, init=False, trainable=False)]
Input2 = [Variable(dim=inputShape, init=False, trainable=False)]
Input2[0].set_value(inputMat)
Input1[0].set_value(inputMat)
pooling1 = pooling(Input1, (2, 2), (2, 2), mode=mode)
pooling2 = pool4conv(Input2, 1, (2, 2), (2, 2), mode=mode)
pooling1[0].forward()
pooling2[0].forward()

# 画图
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 5))
plt.subplot(131)
plt.title('Input Data')
sns.heatmap(inputMat, annot=True, cmap='coolwarm', linecolor='white', linewidth=1)
plt.subplot(132)
plt.title('Original Pooling')
sns.heatmap(pooling1[0].value, annot=True, cmap='coolwarm', linecolor='white', linewidth=1)
plt.subplot(133)
plt.title('Transformed Pooling')
sns.heatmap(pooling2[0].value, annot=True, cmap='coolwarm', linecolor='white', linewidth=1)
plt.legend('两种不同的池化方式对比')
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.title('Original Pooling')
sns.heatmap(pooling1[0].get_jacobi(Input1[0]), annot=False, cmap='coolwarm', linecolor='white', linewidth=1)
plt.subplot(122)
plt.title('Transformed Pooling')
sns.heatmap(pooling2[0].get_jacobi(Input2[0]), annot=False, cmap='coolwarm', linecolor='white', linewidth=1)
plt.show()
