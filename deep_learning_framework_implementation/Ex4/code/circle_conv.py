import cv2
import numpy as np
from matrixslow import Variable
from matrixslow.ops import Operator


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



input_size = (15, 15)
kernel_size = (5, 5)
data = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)
data_resized = cv2.resize(data, input_size)
data_resized = np.matrix(data_resized) / 255  # 图像
kernel = np.matrix(np.ones(kernel_size).T)  # 滤波器
Input1 = Variable(dim=input_size, init=False, trainable=False)
Input2 = Variable(dim=kernel_size, init=False, trainable=False)
Input1.set_value(data_resized)
Input2.set_value(kernel)
trans_conv = CircleConvolution(Input1, Input2, stride=(1, 1))
trans_conv.forward()
print("前向传播", trans_conv.value)
Jacobi_1 = trans_conv.get_jacobi(Input1)
Jacobi_2 = trans_conv.get_jacobi(Input2)
# Show the image

import matplotlib.pyplot as plt

plt.figure()

plt.subplot(141)
img = data
plt.title(f'Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(142)
img = kernel
plt.title(f'Kernel')
plt.imshow(img, cmap='gray')

plt.subplot(143)
img = data_resized
plt.title(f'Resized Image')
plt.imshow(img, cmap='gray')

plt.subplot(144)
img = trans_conv.get_jacobi(Input2)
plt.title(f'Forward Once')
plt.imshow(trans_conv.value, cmap='gray')
plt.tight_layout()
plt.show()

# 可视化Jacobi矩阵，利用Seaborn库

import seaborn as sns

plt.figure(figsize=(10, 10))

for i in range(1, Jacobi_1.shape[1] + 1):
    dim = int(Jacobi_1.shape[0] ** (1 / 2))
    _dim = int(Jacobi_1.shape[1] ** (1 / 2))
    plt.subplot(_dim, _dim, i)
    img = np.array(Jacobi_1)[..., i - 1].reshape(dim, dim)
    sns.heatmap(img, annot=False, cbar_ax=None, xticklabels=False, yticklabels=False, cbar=False)
    plt.axis('off')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
plt.legend("Jacobi matrix of input image")
plt.show()

plt.figure(figsize=(10, 10))

for i in range(1, Jacobi_2.shape[1] + 1):
    dim = int(Jacobi_2.shape[0] ** (1 / 2))
    _dim = int(Jacobi_2.shape[1] ** (1 / 2))
    plt.subplot(_dim, _dim, i)
    img = np.array(Jacobi_2)[..., i - 1].reshape(dim, dim)
    sns.heatmap(img, annot=False, cbar_ax=None, xticklabels=False, yticklabels=False, cbar=False)
    plt.axis('off')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
plt.legend("Jacobi matrix of kernel")
plt.show()
