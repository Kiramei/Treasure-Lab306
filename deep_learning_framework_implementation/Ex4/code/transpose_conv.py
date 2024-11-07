import cv2
import numpy as np

import matplotlib.pyplot as plt
from matrixslow import Variable
from matrixslow.ops import Operator


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



input_size = (5, 5)
data = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)
data_resized = cv2.resize(data, input_size)
data_resized = np.matrix(data_resized) / 255  # 图像
kernel = np.matrix(np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]]).T)  # 滤波器
Input1 = Variable(dim=input_size, init=False, trainable=False)
Input2 = Variable(dim=(3, 3), init=False, trainable=False)
Input1.set_value(data_resized)
Input2.set_value(kernel)
trans_conv = TransposeConvolution(Input1, Input2, stride=(2, 2))
trans_conv.forward()
Jacobi_1 = trans_conv.get_jacobi(Input1)
Jacobi_2 = trans_conv.get_jacobi(Input2)
# Show the image


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
img = trans_conv.value
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
    sns.heatmap(img, annot=False, cmap='coolwarm', cbar_ax=None, xticklabels=False, yticklabels=False, cbar=False)
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
    sns.heatmap(img, annot=False, cmap='coolwarm', cbar_ax=None, xticklabels=False, yticklabels=False, cbar=False)
    plt.axis('off')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
plt.legend("Jacobi matrix of kernel")
plt.show()
