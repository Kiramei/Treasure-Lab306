import sys

# sys.path.append('../..')
import matrixslow as ms
from layer import *
from ref.ops import TransposeConvolution, CircleConvolution

if __name__ == '__main__':
    # arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1], [8, 7, 6, 5]])
    # input_ = np.matrix(arr)
    # Input1 = [ms.core.Variable(dim=(4, 4), init=False, trainable=False)]
    # Input2 = [ms.core.Variable(dim=(4, 4), init=False, trainable=False)]
    # Input2[0].set_value(input_)
    # Input1[0].set_value(input_)
    # pooling1 = pooling(Input1, (2, 2), (2, 2), model='MaxPooling')
    # pooling2 = Pooling_conv(Input2, 1, (2, 2), (2, 2), model="AveragePooling")
    # pooling1[0].forward()
    # pooling2[0].forward()
    # print("Pooling", pooling1[0].value)
    # print("ConvPooling", pooling2[0].value)
    # print("Pooling", pooling1[0].get_jacobi(Input1[0]))
    # print("ConvPooling", pooling2[0].get_jacobi(Input2[0]))

    # data = np.matrix(np.array([[1, 2], [3, 4]]))  # 图像
    # kernel = np.matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  # 滤波器
    # Input1 = ms.core.Variable(dim=(2, 2), init=False, trainable=False)
    # Input2 = ms.core.Variable(dim=(3, 3), init=False, trainable=False)
    # Input1.set_value(data)
    # Input2.set_value(kernel)
    # TransCov = TransposeConvolution(Input1, Input2, stride=(2,2))
    # TransCov.forward()
    # print("前向传播",TransCov.value)
    # print("对图像求Jacobi", TransCov.get_jacobi(Input1))
    # print("对卷积核求Jacobi", TransCov.get_jacobi(Input2))

    data = np.matrix(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1], [8, 7, 6, 5]]))
    kernel = np.matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  # 滤波器
    Input1 = ms.core.Variable(dim=(4, 4), init=False, trainable=False)
    Input2 = ms.core.Variable(dim=(3, 3), init=False, trainable=False)
    Input1.set_value(data)
    Input2.set_value(kernel)
    Circle = CircleConvolution(Input1, Input2, stride=(1,1))
    Circle.forward()
    print("前向传播",Circle.value)
    print("对data求jacobi", Circle.get_jacobi(Input1))
    print("对kernel求jacobi",Circle.get_jacobi(Input2))
