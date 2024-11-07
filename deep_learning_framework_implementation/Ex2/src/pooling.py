import numpy as np

import matrixslow
from matrixslow.ops import Operator


class Pooling(Operator):
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

        self.op = kargs.get('op', 'max')

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
                if window.shape != self.size: continue
                row.append(
                    np.average(window) if self.op == 'avg' else np.max(window)
                )
                unit = np.zeros((w, h))
                unit[top:bottom, left:right] = 1 / ((bottom - top) * (right - left))
                flag.append(unit.flatten())
            if row:
                result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):
        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag


if __name__ == '__main__':
    data = np.mat(np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]))
    dt_node = matrixslow.core.Variable(dim=(4, 4), init=False, trainable=False)
    dt_node.set_value(data)
    x = Pooling(dt_node, stride=(1, 1), size=(2, 2), op='avg')
    x.compute()
    print('Average Pooling Plus result:')
    print(x.value)
    print('-------------------------------')
    x = Pooling(dt_node, stride=(1, 1), size=(2, 2), op='max')
    x.compute()
    print('Max Pooling Plus result:')
    print(x.value)