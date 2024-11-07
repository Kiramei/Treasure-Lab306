import os

import numba as nb
import numpy as np
from alive_progress import alive_bar


def std(_):
    _max = _.max(axis=0)
    _min = _.min(axis=0)
    x_std = (_ - _min) / ((_max - _min) + 1e-7)
    return x_std


@nb.njit
def elbp(image):
    h, w = image.shape
    pd = np.zeros((h + 2, w + 2))
    pd[1:-1, 1:-1] = image
    img = pd
    c_mean = img.mean()
    cil, nil, rdl, adl = np.zeros(2), np.zeros(256), np.zeros(256), np.zeros(256)
    for __i in range(1, h + 1):
        for j in range(1, w + 1):
            cil[1 if img[__i][j] >= c_mean else 0] += 1
            win = [k for ks in img[__i - 1:__i + 2, j - 1:j + 2] for k in ks]
            t_mean = sum(win) / len(win)
            c = [1 if ks >= t_mean else 0 for ks in win]
            d = [1 if ks - img[__i][j] >= 0 else 0 for ks in win]
            win = np.delete(win, 4)
            c = np.delete(c, 4)
            d = np.delete(d, 4)
            win_0 = win[0]
            for k in range(len(win) - 1):
                win[k] = win[k + 1] - win[k]
            win[-1] = win_0 - win[-1]
            win = [1 if ks >= 0 else 0 for ks in win]
            ni_str = ''
            for _i in c:
                ni_str += str(_i)

            rd_str = ''
            for _i in d:
                rd_str += str(_i)

            ad_str = ''
            for _i in win:
                ad_str += str(_i)

            aks = 0
            for digit in ni_str:
                aks <<= 1
                aks += digit == '1'
            nil[aks] += 1

            aks = 0
            for digit in rd_str:
                aks <<= 1
                aks += digit == '1'
            rdl[aks] += 1

            aks = 0
            for digit in ad_str:
                aks <<= 1
                aks += digit == '1'
            adl[aks] += 1
    return cil, nil, rdl, adl


def block_down(img):
    h, w = img.shape
    block_h, block_w = h // 3, w // 3
    blocks = []
    for _i in range(3):
        for j in range(3):
            blocks.append(img[_i * block_h:(_i + 1) * block_h, j * block_w:(j + 1) * block_w])
    return blocks


X, y = np.load('./preprocess/train.npy', allow_pickle=True).T  # 图像数据, shape: (n_samples, height, width, channels)
V, v = np.load('./preprocess/test.npy', allow_pickle=True).T  # 类别标签, shape: (n_samples,)


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
                a, b, c, d = elbp(q)
                features = np.concatenate((std(a), std(b), std(c), std(d)), axis=0)
                fs.append(features)
            fs = np.concatenate(fs, axis=0)
            X_features.append([fs, y_origin[ind]])
            bar()

    return X_features


if not (os.path.exists('./lbp_tr.npy') and os.path.exists('./lbp_te.npy')):
    # 提取ELBP特征
    num_points = 8
    radius = 3

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
    np.save('./lbp_tr_elbp.npy', np.asarray(dt, dtype=object), allow_pickle=True)

    ds = divide_into_intervals(len(V), num_workers)
    dt = None
    for i in range(len(ds)):
        p_data = V[ds[i][0]:ds[i][1]]
        y_data = v[ds[i][0]:ds[i][1]]
        dt = worker(p_data, y_data)
    # save
    assert dt is not None
    np.save('./lbp_te_elbp', np.asarray(dt, dtype=object), allow_pickle=True)
