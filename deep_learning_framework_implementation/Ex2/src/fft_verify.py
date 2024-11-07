import numpy as np

import matrixslow as ms
import time


def compute_fft(self: ms.core.Variable):
    data = self.value  # 图像
    kernel = np.array([
        [0, 0, 0],
        [0, 1, 0 ],
        [0, 0, 0]
    ], dtype=np.float64)  # 滤波器

    w, h = data.shape  # 图像的宽和高
    kw, kh = kernel.shape  # 滤波器尺寸
    hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

    # 补齐数据边缘
    pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))
    self.padded = np.mat(np.zeros((pw, ph)))
    self.padded[0: w, 0: h] = data

    # 卷积结果的尺寸和输入相同。
    self.value = np.mat(np.zeros((w, h)))

    # 计算图像和滤波器的傅里叶变换
    data_fft = np.fft.fft2( self.padded)
    kernel_fft = np.fft.fft2(kernel,  self.padded.shape)

    # 频域卷积
    result_fft = data_fft * kernel_fft

    # 傅里叶反变换获得卷积结果
    self.value = np.fft.ifft2(result_fft).astype(np.float64)

    return self.value[hkw:hkw + w, hkh:hkh + h]

def compute_slow(self: ms.core.Variable):
    data = self.value  # 图像
    kernel = np.array([
        [0, 0, 0],
        [0, 1, 0 ],
        [0, 0, 0]
    ],dtype=np.float64)  # 滤波器

    w, h = data.shape  # 图像的宽和高
    kw, kh = kernel.shape  # 滤波器尺寸
    hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

    # 补齐数据边缘
    pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))
    self.padded = np.mat(np.zeros((pw, ph)))
    self.padded[hkw:hkw + w, hkh:hkh + h] = data

    # 卷积结果的尺寸和输入相同。
    self.value = np.mat(np.zeros((w, h)))

    # 二维离散卷积
    for i in np.arange(hkw, hkw + w):
        for j in np.arange(hkh, hkh + h):
            # element-wise multiplication, then sum up.
            tx = self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh]
            self.value[i - hkw, j - hkh] = np.sum(np.multiply(tx, kernel))

    return self.value
def main():
    # 随机生成输入特征图
    num_feature_maps = 3
    input_shape = (25, 25)
    x = ms.core.Variable(dim=(25, 25), init=False, trainable=False)
    x.set_value(np.mat((np.random.random((25, 25))*255).astype(np.uint8)))
    temp = x.value
    # 使用原始实现
    start_time = time.time()
    outputs_slow = compute_slow(x)
    end_time = time.time()
    time_cost_slow = end_time - start_time

    # 使用 FFT 实现
    start_time = time.time()
    outputs_fft = compute_fft(x)
    end_time = time.time()
    time_cost_fft = end_time - start_time

    # 比较两种实现的结果

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(temp, cmap='gray')
    plt.title('Input')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(outputs_slow, cmap='gray')
    plt.title('Slow')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(outputs_fft, cmap='gray')
    plt.title('FFT')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(outputs_slow - outputs_fft, cmap='gray')
    plt.title('Difference')
    plt.axis('off')

    plt.show()

    assert np.allclose(outputs_fft, outputs_slow)

    print(f"原始实现耗时: {time_cost_slow:.6f} 秒")
    print(f"FFT 实现耗时: {time_cost_fft:.6f} 秒")


if __name__ == '__main__':
    main()
