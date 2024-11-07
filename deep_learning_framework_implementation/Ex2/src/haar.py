import numpy as np

import matrixslow
from matrixslow.ops import Operator


class WaveletHaar(Operator):
    def __init__(self, *parents, **kargs):
        Operator.__init__(self, *parents, **kargs)
        # 四个子矩阵的操作
        self.ops = [np.mat(np.array([[1, 1], [1, 1]])),
                    np.mat(np.array([[1, -1], [1, -1]])),
                    np.mat(np.array([[1, 1], [-1, -1]])),
                    np.mat(np.array([[1, -1], [-1, 1]]))]

    def compute(self):
        H, W = self.parents[0].shape()
        assert H % 2 == 0 and W % 2 == 0
        self.value = np.zeros(self.parents[0].shape())
        data_ = self.parents[0].value
        H, W = H // 2, W // 2
        for i in range(H):
            for j in range(W):
                i2, j2 = i * 2, j * 2
                _data = data_[i2:i2 + 2, j2:j2 + 2]
                self.value[i, j] = np.sum(np.multiply(_data, self.ops[0])) / 4
                self.value[i, W + j] = np.sum(np.multiply(_data, self.ops[1])) / 4
                self.value[i + H, j] = np.sum(np.multiply(_data, self.ops[2])) / 4
                self.value[H + i, W + j] = np.sum(np.multiply(_data, self.ops[3])) / 4

    def get_jacobi(self, parent):
        jacobi = np.zeros((self.dimension(), parent.dimension()))
        H, W = self.parents[0].shape()
        H, W = H // 2, W // 2
        # Pre-compute the necessary offsets
        h_offsets = np.tile(np.arange(H) * 2, H)
        w_offsets = np.repeat(np.arange(W) * 2, H)
        h_offsets_higher = h_offsets + H * 2
        w_offsets_higher = w_offsets + W * 2
        ops = np.array(self.ops, dtype=object).reshape(2, 2)
        for i in range(self.dimension()):
            h, w = i % H, i // H
            jacobi[i] = ops[int(h < H), int(w < W)]
            # Apply the necessary offsets to the flattened Jacobian elements
            jacobi[i, h_offsets + w_offsets if h < H else h_offsets_higher + w_offsets_higher] = jacobi[i]
        return jacobi


if __name__ == '__main__':
    import cv2
    data = cv2.imread('../sena.png', cv2.IMREAD_GRAYSCALE)
    data = cv2.resize(data, (256, 256))
    data = np.mat(data)
    dt_node = matrixslow.core.Variable(dim=data.shape, init=False, trainable=False)
    dt_node.set_value(data)
    y = WaveletHaar(dt_node)
    y.compute()
    print(y.value)
    cv2.imshow('haar', y.value.astype(np.uint8))
    cv2.waitKey(0)
