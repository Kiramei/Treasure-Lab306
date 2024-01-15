import os

import numpy as np
from alive_progress import alive_bar
from numba import njit, typed, float64, uint8

from .lbp_numba import basic_lbp


@njit
def convert_histogram(H):
    uniform_LBP = typed.List.empty_list(float64)
    for _ in range(59):
        uniform_LBP.append(0.0)

    uniform_dict = typed.Dict.empty(
        key_type=uint8, value_type=uint8
    )
    uniform_dict[0] = 0
    uniform_dict[255] = 1
    uniform_dict[127] = 2
    uniform_dict[191] = 3
    uniform_dict[223] = 4
    uniform_dict[239] = 5
    uniform_dict[247] = 6
    uniform_dict[251] = 7
    uniform_dict[253] = 8
    uniform_dict[254] = 9
    uniform_dict[63] = 10
    uniform_dict[159] = 11
    uniform_dict[207] = 12
    uniform_dict[231] = 13
    uniform_dict[243] = 14
    uniform_dict[249] = 15
    uniform_dict[252] = 16
    uniform_dict[126] = 17
    uniform_dict[31] = 18
    uniform_dict[143] = 19
    uniform_dict[199] = 20
    uniform_dict[227] = 21
    uniform_dict[241] = 22
    uniform_dict[248] = 23
    uniform_dict[124] = 24
    uniform_dict[61] = 25
    uniform_dict[15] = 26
    uniform_dict[135] = 27
    uniform_dict[195] = 28
    uniform_dict[225] = 29
    uniform_dict[240] = 30
    uniform_dict[120] = 31
    uniform_dict[60] = 32
    uniform_dict[30] = 33
    uniform_dict[7] = 34
    uniform_dict[131] = 35
    uniform_dict[193] = 36
    uniform_dict[224] = 37
    uniform_dict[112] = 38
    uniform_dict[56] = 39
    uniform_dict[28] = 40
    uniform_dict[14] = 41
    uniform_dict[3] = 42
    uniform_dict[129] = 43
    uniform_dict[192] = 44
    uniform_dict[96] = 45
    uniform_dict[48] = 46
    uniform_dict[24] = 47
    uniform_dict[12] = 48
    uniform_dict[6] = 49
    uniform_dict[1] = 50
    uniform_dict[128] = 51
    uniform_dict[64] = 52
    uniform_dict[32] = 53
    uniform_dict[16] = 54
    uniform_dict[8] = 55
    uniform_dict[4] = 56
    uniform_dict[2] = 57

    keys = [0, 255, 127, 191, 223, 239, 247, 251, 253, 254, 63, 159, 207, 231, 243, 249, 252, 126, 31, 143, 199, 227,
            241, 248, 124, 61, 15, 135, 195, 225, 240, 120, 60, 30, 7, 131, 193, 224, 112, 56, 28, 14, 3, 129, 192,
            96, 48, 24, 12, 6, 1, 128, 64, 32, 16, 8, 4, 2]

    for idx in range(256):
        if idx in keys:
            uniform_LBP[uniform_dict[idx] + 1] = H[idx]
        else:
            uniform_LBP[0] += H[idx]
    return uniform_LBP


# block down an image to 9 blocks
def block_down(img):
    h, w = img.shape
    block_h, block_w = h // 3, w // 3
    blocks = []
    for _i in range(3):
        for j in range(3):
            blocks.append(img[_i * block_h:(_i + 1) * block_h, j * block_w:(j + 1) * block_w])
    return blocks


# 假设有一批图像数据集，其中X为图像数据，y为对应的类别标签
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
                features = basic_lbp(q)
                features = np.array(features)
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
    np.save('./lbp_tr.npy', np.asarray(dt, dtype=object), allow_pickle=True)

    ds = divide_into_intervals(len(V), num_workers)
    dt = None
    for i in range(len(ds)):
        p_data = V[ds[i][0]:ds[i][1]]
        y_data = v[ds[i][0]:ds[i][1]]
        dt = worker(p_data, y_data)
    # save
    assert dt is not None
    np.save('./lbp_te.npy', np.asarray(dt, dtype=object), allow_pickle=True)
