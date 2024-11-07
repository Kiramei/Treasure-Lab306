import os

import numba as nb
import numpy as np
from alive_progress import alive_bar


@nb.njit
def basic_lbp(src):
    res = [0] * 256
    h, w = src.shape
    pd = np.zeros((h + 2, w + 2))
    pd[1:-1, 1:-1] = src
    src = pd
    for _i in range(1, h + 1):
        for _j in range(1, w + 1):
            win = [[0 for _ in range(3)] for _ in range(3)]
            rf = []
            for ii in range(_i - 1, _i + 2):
                for jj in range(_j - 1, _j + 2):
                    if src[ii][jj] >= src[_i][_j]:
                        win[ii - (_i - 1)][jj - (_j - 1)] = 1
                        rf.append((ii, jj))
            win = [val for sublist in win for val in sublist]
            win.pop(4)
            str1 = ''
            for s in win:
                str1 += str(s)
            _d = 0
            for digit in str1:
                _d <<= 1
                _d += digit == '1'
            res[_d] += 1
    return res


def block_down(img):
    h, w = img.shape
    block_h, block_w = h // 3, w // 3
    blocks = []
    for _i in range(3):
        for j in range(3):
            blocks.append(img[_i * block_h:(_i + 1) * block_h, j * block_w:(j + 1) * block_w])
    return blocks


X, y = np.load('../preprocess/train.npy', allow_pickle=True).T  # 图像数据, shape: (n_samples, height, width, channels)
V, v = np.load('../preprocess/test.npy', allow_pickle=True).T  # 类别标签, shape: (n_samples,)


def divide_into_intervals(s, n):
    interval_size = s / n
    intervals = []
    for _i in range(n):
        start = int(_i * interval_size)
        end = int((_i + 1) * interval_size)
        intervals.append((start, end))
    return intervals


def worker(X_origin, y_origin):
    X_features = []
    with alive_bar(y_origin.shape[0], title='Dividing train data') as bar:
        for ind, image in enumerate(X_origin):
            lf = block_down(image)
            fs = []
            for q in lf:
                features = basic_lbp(q)
                _max = features.max(axis=0)
                _min = features.min(axis=0)
                features = (features - _min) / (_max - _min)
                fs.append(features)

            fs = np.concatenate(fs, axis=0)
            X_features.append([fs, y_origin[ind]])
            bar()

    return X_features


if not (os.path.exists('./lbp_tr.npy') and os.path.exists('./lbp_te.npy')):
    # 提取ELBP特征
    num_points = 8  # LBP算子的采样点数
    radius = 3  # LBP算子的采样半径

    X = np.array([*X])
    V = np.array([*V])

    # X_features = []
    num_workers = 1

    ds = divide_into_intervals(len(X), num_workers)
    # multi process
    dt = None
    for i in range(len(ds)):
        p_data = X[ds[i][0]:ds[i][1]]
        y_data = y[ds[i][0]:ds[i][1]]
        dt = worker(p_data, y_data)
    # save
    assert dt is not None
    np.save('./lbp_tr_basic.npy', np.asarray(dt, dtype=object), allow_pickle=True)

    ds = divide_into_intervals(len(V), num_workers)
    dt = None
    for i in range(len(ds)):
        p_data = V[ds[i][0]:ds[i][1]]
        y_data = v[ds[i][0]:ds[i][1]]
        dt = worker(p_data, y_data)
    # save
    assert dt is not None
    np.save('./lbp_te_basic.npy', np.asarray(dt, dtype=object), allow_pickle=True)
