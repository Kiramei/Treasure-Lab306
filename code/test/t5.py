import numpy as np
from sklearn.preprocessing import StandardScaler


def my_LBP(img, c='LBP'):
    """
    Args:
        img: 输入的图像矩阵，单通道灰度图
        c:  LBP种类，可选择：LBP, C_LBP, E_LBP, uniform_LBP
    Returns: 直方图
    """
    h, w = img.shape
    # 使用均值补充边界信息
    img = np.pad(img, ((1, 1), (1, 1)), 'mean')
    scaler = StandardScaler()

    if c == 'LBP' or c == 'uniform_LBP':
        H = np.zeros(256)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                windows = np.zeros(img[i - 1:i + 2, j - 1:j + 2].shape, dtype=np.uint8)
                windows[np.where(img[i - 1:i + 2, j - 1:j + 2] >= img[i, j])] = 1
                windows = np.delete(windows.flatten(), 4)
                str1 = ''.join(str(i) for i in windows)
                H[int(str1, 2)] += 1

        if c == 'uniform_LBP':
            # 因为uniform_LBP和LBP有特定的转换关系，所以可以通过打标完成
            uniform_LBP = np.zeros(59)
            uniform_dict = {0: 0, 255: 1, 127: 2, 191: 3, 223: 4, 239: 5, 247: 6, 251: 7, 253: 8, 254: 9,
                            63: 10, 159: 11, 207: 12, 231: 13, 243: 14, 249: 15, 252: 16, 126: 17,
                            31: 18, 143: 19, 199: 20, 227: 21, 241: 22, 248: 23, 124: 24, 61: 25,
                            15: 26, 135: 27, 195: 28, 225: 29, 240: 30, 120: 31, 60: 32, 30: 33,
                            7: 34, 131: 35, 193: 36, 224: 37, 112: 38, 56: 39, 28: 40, 14: 41,
                            3: 42, 129: 43, 192: 44, 96: 45, 48: 46, 24: 47, 12: 48, 6: 49,
                            1: 50, 128: 51, 64: 52, 32: 53, 16: 54, 8: 55, 4: 56, 2: 57}
            for idx in range(256):
                if idx in uniform_dict.keys():
                    uniform_LBP[uniform_dict[idx] + 1] = H[idx]
                else:
                    uniform_LBP[0] += H[idx]
            return scaler.transform(scaler.fit_transform(uniform_LBP))
        else:
            return scaler.transform(scaler.fit_transform(H))

    elif c == 'C_LBP':
        c_mean = img.mean()
        C_LBP_C, C_LBP_S, C_LBP_M = np.zeros(2), np.zeros(256), np.zeros(256)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                # 计算中心点与8领域之间的差值
                windows = np.array(img[i - 1:i + 2, j - 1:j + 2], dtype=np.int_) - img[i][j]
                if img[i][j] >= c_mean:
                    C_LBP_C[1] += 1
                else:
                    C_LBP_C[0] += 1
                M_windows = np.zeros(windows.shape)
                M_windows[np.abs(windows) >= c_mean] = 1
                windows[np.where(windows >= 0)] = 1
                windows[np.where(windows < 0)] = 0
                windows, M_windows = np.delete(windows.flatten(), 4), np.delete(M_windows.flatten(), 4)
                s_str, M_str = ''.join(str(i) for i in windows), ''.join(str(i) for i in M_windows)
                C_LBP_S[int(s_str, 2)] += 1
                C_LBP_M[int(M_str, 2)] += 1
        # 将三个直方图串联起来
        H = np.concatenate((guiyi(C_LBP_S), guiyi(C_LBP_M), guiyi(C_LBP_C)), axis=0)
        return H

    elif c == 'E_LBP':
        c_mean = np.mean(img)
        CI_LBP, NI_LBP, RD_LBP, AD_LBP = np.zeros(2), np.zeros(256), np.zeros(256), np.zeros(256)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                if img[i][j] >= c_mean:
                    CI_LBP[1] += 1
                else:
                    CI_LBP[0] += 1
                windows = np.array(img[i - 1:i + 2, j - 1:j + 2], dtype=np.int_)
                ni_LBP, rd_LBP = np.zeros(windows.shape, dtype=np.int_), np.zeros(windows.shape, dtype=np.int_)
                tem_mean = windows.mean()
                ni_LBP[np.where(windows >= tem_mean)] = 1
                # 计算径向相同元素之间的灰度值差
                rd_LBP[np.where((windows - img[i][j]) >= 0)] = 1
                windows = np.delete(windows.flatten(), 4)
                ni_LBP = np.delete(ni_LBP.flatten(), 4)
                rd_LBP = np.delete(rd_LBP.flatten(), 4)
                # 计算角度相同之间的灰度值差
                windows_0 = windows[0]
                windows[:-1] = windows[1:] - windows[:-1]
                windows[-1] = windows_0 - windows[-1]
                windows[windows >= 0] = 1
                windows[windows < 0] = 0
                ni_str, rd_str, ad_str = ''.join(str(i) for i in ni_LBP), ''.join(str(i) for i in rd_LBP), ''.join(
                    str(i) for i in windows)
                NI_LBP[int(ni_str, 2)] += 1
                RD_LBP[int(rd_str, 2)] += 1
                AD_LBP[int(ad_str, 2)] += 1
        H = np.concatenate((guiyi(CI_LBP), guiyi(NI_LBP), guiyi(RD_LBP), guiyi(AD_LBP)), axis=0)
        return H
