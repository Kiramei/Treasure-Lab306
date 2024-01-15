import os

import numba as nb
import numpy as np
from alive_progress import alive_bar


def std(_X):
    _max = _X.max(axis=0)
    _min = _X.min(axis=0)
    x_std = (_X - _min) / ((_max - _min) + 1e-7)
    return x_std


@nb.njit
def clbp(image):
    h, w = image.shape
    pd = np.zeros((h + 2, w + 2))
    pd[1:-1, 1:-1] = image
    img = pd
    c_mean = img.mean()
    C, S, M = np.zeros(2), np.zeros(256), np.zeros(256)
    for ii in range(1, h + 1):
        for j in range(1, w + 1):
            windows = [k - img[ii, j] for ks in img[ii - 1:ii + 2, j - 1:j + 2] for k in ks]
            M_windows = [(1 if abs(ks) >= c_mean else 0) for ks in windows]
            windows = [1 if ks >= 0 else 0 for ks in windows]

            windows.pop(4)
            M_windows.pop(4)

            s_str = ''
            for s in windows:
                s_str += str(s)

            M_str = ''
            for s in M_windows:
                M_str += str(s)

            aks = 0
            for digit in s_str:
                aks <<= 1
                aks += digit == '1'
            S[aks] += 1

            aks = 0
            for digit in M_str:
                aks <<= 1
                aks += digit == '1'
            M[aks] += 1
    return S, M, C


# block down an image to 9 blocks
def block_down(img):
    h, w = img.shape
    block_h, block_w = h // 3, w // 3
    blocks = []
    for _i in range(3):
        for j in range(3):
            blocks.append(img[_i * block_h:(_i + 1) * block_h, j * block_w:(j + 1) * block_w])
    return blocks


X, y = np.load('./preprocess/train.npy', allow_pickle=True).T
V, v = np.load('./preprocess/test.npy', allow_pickle=True).T


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
                a, b, c = clbp(q)
                features = np.concatenate((std(a), std(b), std(c)), axis=0)
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
    np.save('./lbp_tr_clbp.npy', np.asarray(dt, dtype=object), allow_pickle=True)

    ds = divide_into_intervals(len(V), num_workers)
    dt = None
    for i in range(len(ds)):
        p_data = V[ds[i][0]:ds[i][1]]
        y_data = v[ds[i][0]:ds[i][1]]
        dt = worker(p_data, y_data)
    # save
    assert dt is not None
    np.save('./lbp_te_clbp', np.asarray(dt, dtype=object), allow_pickle=True)
