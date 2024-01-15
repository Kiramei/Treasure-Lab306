# import the necessary packages
import datetime

import cv2
import logging
import unittest
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
from numba import jit
from skimage.feature import local_binary_pattern
# scaler
from sklearn.preprocessing import StandardScaler


class LBP:
    # uniform_map为等价模式的58种特征值从小到大进行序列化编号得到的字典
    UNIFORM_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 12: 8,
                   14: 9, 15: 10, 16: 11, 24: 12, 28: 13, 30: 14, 31: 15, 32: 16,
                   48: 17, 56: 18, 60: 19, 62: 20, 63: 21, 64: 22, 96: 23, 112: 24,
                   120: 25, 124: 26, 126: 27, 127: 28, 128: 29, 129: 30, 131: 31, 135: 32,
                   143: 33, 159: 34, 191: 35, 192: 36, 193: 37, 195: 38, 199: 39, 207: 40,
                   223: 41, 224: 42, 225: 43, 227: 44, 231: 45, 239: 46, 240: 47, 241: 48,
                   243: 49, 247: 50, 248: 51, 249: 52, 251: 53, 252: 54, 253: 55, 254: 56,
                   255: 57}

    EPS = 1e-7

    METHOD_MAP = {
        'basic': 'lbp_basic',
        'uniform': 'lbp_uniform',
        'circular': 'lbp_circular',
        'completed': 'lbp_completed',
        'revolve_completed': 'lbp_revolve_completed'
    }

    METHOD_LIST = ['basic', 'revolve', 'uniform', 'circular', 'revolve_circular', 'revolve_completed']

    def __init__(self, method='revolve_uniform', radius=1, neighbors=8):
        self.src_img = None
        self.output_img = None
        self.output_hist = None

        self.method = method
        self.radius = radius
        self.neighbors = neighbors

    # 将图像载入，并转化为灰度图，获取图像灰度图的像素信息

    def describe(self, _image):
        if len(_image.shape) == 3:
            self.src_img = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        else:
            self.src_img = _image
        # 执行LBP算法
        _method = self.METHOD_MAP[self.method]
        getattr(self, _method)()
        return self
        # return _image_array

    def fetch(self):
        return self.output_img, self.output_hist

    @jit
    def lbp_basic(self, std=True):
        H = np.zeros(256)
        h, w = self.src_img.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                windows = np.zeros(self.src_img[i - 1:i + 2, j - 1:j + 2].shape, dtype=np.uint8)
                windows[np.where(self.src_img[i - 1:i + 2, j - 1:j + 2] >= self.src_img[i, j])] = 1
                windows = np.delete(windows.flatten(), 4)
                str1 = ''.join(str(i) for i in windows)
                H[int(str1, 2)] += 1
        if std:
            H = self.std(H)
        self.output_hist = H
        return H

    @jit
    def lbp_uniform(self):
        H = self.lbp_basic(std=False)
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

        H = self.std(uniform_LBP)
        self.output_hist = H
        return H

    def lbp_revolve_uniform(self):
        uniform_revolve_array = np.zeros(self.src_img.shape, dtype=np.uint8)
        basic_array = self.lbp_basic()
        width, height = self.src_img.shape[:2]

        k = np.left_shift(basic_array, 1)
        k = np.where(k > 255, k - 255, k)

        xor = np.bitwise_xor(basic_array, k)
        num = np.unpackbits(xor, axis=-1).reshape(width, height, 8).sum(axis=-1)

        less_equal_two = num <= 2
        np.count_nonzero(np.unpackbits(np.array([num[less_equal_two.all()]], dtype=np.uint8)) == 1)
        uniform_revolve_array[less_equal_two.all()] = np.count_nonzero(
            np.unpackbits(np.array([num[less_equal_two.all()]], dtype=np.uint8)) == 1)
        uniform_revolve_array[~(less_equal_two.all())] = 9
        _hist = cv2.calcHist([uniform_revolve_array], [0], None, [60], [0, 60])
        self.output_hist = cv2.normalize(_hist, _hist).flatten()
        self.output_img = uniform_revolve_array

    # 绘制指定维数和范围的图像灰度归一化统计直方图
    def show_hist(self, im_bins, im_range):
        _hist = cv2.calcHist([self.src_img], [0], None, im_bins, im_range)
        _hist = cv2.normalize(_hist, _hist).flatten()
        plt.plot(_hist, color='r')
        plt.xlim(im_range)
        plt.show()

    def lbp_circular(self):
        height, width = self.src_img.shape
        output = np.zeros((height, width), dtype=np.uint8)

        cos_vals = np.cos(2 * np.pi * np.arange(self.neighbors) / self.neighbors)
        sin_vals = np.sin(2 * np.pi * np.arange(self.neighbors) / self.neighbors)

        radius_rounded = int(np.round(self.radius))

        i_vals = np.arange(radius_rounded, height - radius_rounded)
        j_vals = np.arange(radius_rounded, width - radius_rounded)

        i_mesh, j_mesh = np.meshgrid(i_vals, j_vals, indexing='ij')

        x_coords = np.round(radius_rounded * cos_vals[:, np.newaxis, np.newaxis])
        y_coords = -np.round(radius_rounded * sin_vals[:, np.newaxis, np.newaxis])

        neighbor_coords = np.stack((x_coords, y_coords), axis=-1)
        neighbor_coords = neighbor_coords.astype(int)

        center_vals = self.src_img[i_mesh, j_mesh]
        neighbor_vals = self.src_img[i_mesh + neighbor_coords[..., 0], j_mesh + neighbor_coords[..., 1]]

        codes = np.where(neighbor_vals > center_vals, 1, 0)
        codes = np.packbits(codes, axis=-1)

        output[i_mesh, j_mesh] = np.packbits(codes, axis=-1)

        _hist, _ = np.histogram(output.ravel(), bins=np.arange(0, 11), range=(0, 10))
        _hist /= _hist.sum() + self.EPS

        self.output_img = output
        self.output_hist = _hist

    def lbp_completed(self):
        c_mean = self.src_img.mean()
        clbp_c = np.zeros(2)
        clbp_s = np.zeros(256)
        clbp_m = np.zeros(256)
        h, w = self.src_img.shape

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                win = self.src_img[i - 1:i + 2, j - 1:j + 2].astype(int) - self.src_img[i, j]
                clbp_c[1 if self.src_img[i, j] >= c_mean else 0] += 1
                M_win = np.zeros(win.shape, dtype=int)
                M_win[np.abs(win) >= c_mean] = 1
                win = np.where(win >= 0, 1, 0)
                win, M_win = np.delete(win.flatten(), 4), np.delete(M_win.flatten(), 4)
                s_str, M_str = ''.join(map(str, win)), ''.join(map(str, M_win))
                clbp_s[int(s_str, 2)] += 1
                clbp_m[int(M_str, 2)] += 1

        # 将三个直方图串联起来
        H = np.concatenate((self.std(clbp_s), self.std(clbp_m), self.std(clbp_c)), axis=0)
        self.output_hist = H
        return H

    @staticmethod
    def std(x):
        x = np.array(x)
        return (x - np.min(x)) / (np.max(x) - np.min(x) + LBP.EPS)


class ELBP:

    # Initializes the model with the given specifications
    def __init__(self, p=16, _r=2):
        self.name = "elbp"
        self.p = p
        self.r = _r
        self.method = 'uniform'

    # Computes feature vectors for the given image matrix
    def compute_features(self, _image):
        # Computes the LBP matrix for the given image
        _lbp = self.get_lbp(_image, self.p, self.r, self.method)
        return np.histogram(_lbp.ravel(), bins=self.p + 2, range=(0, self.p + 1))[0]

    def compute_elbp_features(self, _image):
        _elbp = self.get_lbp(_image)
        _var = self.get_var_mat(_image, 10)
        feature = np.zeros([self.p + 2, 10])
        x, y = _elbp.shape[0], _elbp.shape[1]
        for i in range(x):
            for j in range(y):
                feature[int(_elbp[i][j])][int(_var[i][j])] += 1
        return feature.flatten()

    def get_lbp(self, _image):
        result = local_binary_pattern(_image, self.p, self.r, self.method)
        return result

    @staticmethod
    def get_var(_mat, i, j):
        res = []
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if 0 <= x < len(_mat) and 0 <= y < len(_mat):
                    if x != i or y != j:
                        res.append(_mat[x][y])
        return np.std(res)

    def get_var_mat(self, _image, bins=10):
        _elbp = []
        for i in range(0, len(_image)):
            tmp = []
            for j in range(0, len(_image)):
                _var = self.get_var(_image, i, j)
                tmp.append(_var)
            _elbp.append(np.array(tmp))
        _elbp = np.array(_elbp)
        min_var = np.min(_elbp)
        max_var = np.max(_elbp)
        _elbp = ((_elbp - min_var) * bins) // (max_var - min_var)
        _elbp = np.where(_elbp > (bins - 1), bins - 1, _elbp)
        return _elbp

    # Computes feature vectors for all the given image matrices
    def compute_features_for_images(self, images):
        return np.array([self.compute_features(x) for x in images])



class Descriptor:
    def __init__(self, method='uniform', radius=1, neighbors=8):
        self.train_origin = np.load('./preprocess/train.npy', allow_pickle=True)
        self.test_origin = np.load('./preprocess/test.npy', allow_pickle=True)
        self.train_X, self.train_y = (np.concatenate(self.train_origin[:, 0], axis=0)
                                      .reshape(-1, 256 ** 2), self.train_origin[:, 1])
        self.test_X, self.test_y = (np.concatenate(self.test_origin[:, 0], axis=0)
                                    .reshape(-1, 256 ** 2), self.test_origin[:, 1])
        self.lbp = LBP(method=method, radius=radius, neighbors=neighbors)
        logging.basicConfig(level=logging.INFO)
        logging.log(logging.INFO, 'Start processing...')
        self.method = method

    def run_with_multi_core(self):
        pool_args = []
        num_workers = 20
        separate = len(self.test_X) // num_workers
        for i in range(num_workers + 1):
            start = i * separate
            end = (i + 1) * separate
            p_data = self.test_origin[start:min(end, len(self.test_origin))]
            # self.worker((start, end, self.test_origin, self.lbp, num_workers))
            pool_args.append((start, end, p_data, self.lbp, num_workers))
            logging.log(logging.INFO, f'Process {i + 1} added!')

        Pool().map_async(self.worker, pool_args, callback=self.process_test).get()

    @staticmethod
    def worker(args):
        _start, _end, _data, _lbp, _sup = args
        sep = min(_end, len(_data)) - _start
        X, y = np.concatenate(_data[:, 0], axis=0).reshape(-1, 256 ** 2), _data[:, 1]
        result = []
        for j in range(len(_data)):
            resolve_uniform_array = _lbp.describe(X[j].reshape(256, -1)).fetch()[1]
            result.append([resolve_uniform_array, y[j]])
            if j % 100 == 0:
                print('Info |',
                      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      'Process:', j)
        return result

    def run_core(self):
        train_parsed = []

        # with alive_bar(len(self.train_X), title='Preparing train data') as _bar:
        # with alive_bar(len(self.train_X), title='Preparing train data') as bar:
        for i in range(len(self.train_X)):
            resolve_uniform_array = self.lbp.describe(self.train_X[i].reshape(256, -1))
            train_parsed.append(resolve_uniform_array)
            if i % 10 == 0:
                print('Info |',
                      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      'Process:', i)
            # _bar()
        train_parsed = np.array(train_parsed)
        train_parsed = train_parsed.reshape(train_parsed.shape[0], -1)
        logging.log(logging.INFO, 'Train data processed!')
        test_parsed = []
        # with alive_bar(len(self.test_X), title='Preparing test data') as _bar:
        for i in range(len(self.test_X)):
            resolve_uniform_array = self.lbp.describe(self.test_X[i].reshape(256, -1))
            test_parsed.append(resolve_uniform_array)
            if i % 10 == 0:
                print('Info |',
                      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      'Process:', i)
            # _bar()
        logging.log(logging.INFO, 'Test data processed!')
        return train_parsed, test_parsed

    def run_with_one_core(self):
        train_parsed, test_parsed = self.run_core()
        parsed_train_data = []
        parsed_test_data = []
        for i in range(len(self.train_X)):
            parsed_train_data.append([self.train_X[i], self.train_y[i]])
        for i in range(len(self.test_X)):
            parsed_test_data.append([self.test_X[i], self.test_y[i]])
        np.save(f'./preprocess/train_lbp_{self.method}.npy', np.asarray(parsed_train_data, dtype=object),
                allow_pickle=True)
        np.save(f'./preprocess/test_lbp_{self.method}.npy', np.asarray(parsed_test_data, dtype=object),
                allow_pickle=True)

    @staticmethod
    def process_train(vs):
        logging.log(logging.INFO, f'Process finished!')
        data_parsed = [u for v in vs for u in v]
        data_parsed = np.asarray(data_parsed, dtype=object)
        np.save(f'./preprocess/train_lbp_clbp.npy', data_parsed, allow_pickle=True)

    @staticmethod
    def process_test(vs):
        logging.log(logging.INFO, f'Process finished!')
        data_parsed = [u for v in vs for u in v]
        data_parsed = np.asarray(data_parsed, dtype=object)
        np.save(f'./preprocess/test_lbp_clbp.npy', data_parsed, allow_pickle=True)
