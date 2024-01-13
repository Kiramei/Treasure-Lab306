# import the necessary packages

import unittest
from multiprocessing import Pool

import cv2
from matplotlib import pyplot as plt
from pylab import *
from skimage.feature import local_binary_pattern


# from sklearn.decomposition import PCA

class LBP:
    # revolve_map为旋转不变模式的36种特征值从小到大进行序列化编号得到的字典
    RESOLVE_MAP = {0: 0, 1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6, 13: 7, 15: 8, 17: 9, 19: 10, 21: 11, 23: 12,
                   25: 13, 27: 14, 29: 15, 31: 16, 37: 17, 39: 18, 43: 19, 45: 20, 47: 21, 51: 22, 53: 23,
                   55: 24, 59: 25, 61: 26, 63: 27, 85: 28, 87: 29, 91: 30, 95: 31, 111: 32, 119: 33, 127: 34,
                   255: 35}
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
        'revolve': 'lbp_revolve',
        'uniform': 'lbp_uniform',
        'revolve_uniform': 'lbp_revolve_uniform',
        'revolve_circular': 'lbp_revolve_circular',
        'revolve_completed': 'lbp_revolve_completed'
    }

    METHOD_LIST = ['basic', 'revolve', 'uniform', 'revolve_uniform', 'revolve_circular', 'revolve_completed']

    def __init__(self, method='revolve_uniform', radius=1, neighbors=8):
        self.src_img = None
        self.output_img = None
        self.output_hist = None

        self.method = method
        self.radius = radius
        self.neighbors = neighbors

    # 将图像载入，并转化为灰度图，获取图像灰度图的像素信息
    def describe(self, _image):
        self.src_img = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        # 执行LBP算法
        _method = self.METHOD_MAP[self.method]
        getattr(self, _method)()
        return self
        # return _image_array

    def fetch(self):
        return self.output_img, self.output_hist

    # 图像的LBP原始特征计算算法：将图像指定位置的像素与周围8个像素比较
    # 比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    def calculate_basic_lbp(self, i, j):
        _sum = []
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
        for offset in offsets:
            _sum.append(1 if self.src_img[i + offset[0], j + offset[1]] > self.src_img[i, j] else 0)
        return _sum

    # 获取图像的LBP原始模式特征
    # def lbp_basic(self):
    #     basic_array = np.zeros(self.src_img.shape, np.uint8)
    #     width = self.src_img.shape[0]
    #     height = self.src_img.shape[1]
    #     for i in range(1, width - 1):
    #         for j in range(1, height - 1):
    #             _sum = self.calculate_basic_lbp(i, j)
    #             bit_num = 0
    #             result = 0
    #             for s in _sum:
    #                 result += s << bit_num
    #                 bit_num += 1
    #             basic_array[i, j] = result
    #     return basic_array

    def lbp_basic(self):
        basic_array = np.zeros(self.src_img.shape, dtype=np.uint8)
        width, height = self.src_img.shape[:2]

        for i in range(1, width - 1):
            for j in range(1, height - 1):
                _sum = self.calculate_basic_lbp(i, j)
                result = np.packbits(_sum[::-1])[0]
                basic_array[i, j] = result

        return basic_array

    # 获取图像的LBP旋转不变模式特征
    def lbp_revolve(self):
        revolve_array = np.zeros(self.src_img.shape, dtype=np.uint8)
        width, height = self.src_img.shape[:2]

        circle = np.zeros((width + 2, height + 2), dtype=self.src_img.dtype)
        circle[1:-1, 1:-1] = self.src_img

        for i in range(1, width + 1):
            for j in range(1, height + 1):
                _sum = self.calculate_basic_lbp(i - 1, j - 1)
                circle_values = circle[i - 1:i + 2, j - 1:j + 2]
                values = np.unpackbits(circle_values, axis=-1).reshape(9, 8).sum(axis=-1)
                revolve_key = np.min(values)
                revolve_array[i - 1, j - 1] = self.RESOLVE_MAP[revolve_key]

        return revolve_array
        # revolve_array = np.zeros(self.src_img.shape, np.uint8)
        # width, height = self.src_img.shape[:2]
        # for i in range(1, width - 1):
        #     for j in range(1, height - 1):
        #         _sum = self.calculate_basic_lbp(i, j)
        #         values = []
        #         circle = _sum
        #         circle.extend(_sum)
        #         for _i in range(0, 8):
        #             _j = 0
        #             _sum = 0
        #             bit_num = 0
        #             while _j < 8:
        #                 _sum += circle[_i + _j] << bit_num
        #                 bit_num += 1
        #                 _j += 1
        #             values.append(_sum)
        #         revolve_key = min(values)
        #         revolve_array[i, j] = self.RESOLVE_MAP[revolve_key]
        # return revolve_array

    # 获取图像的LBP等价模式特征
    def lbp_uniform(self):
        uniform_array = np.zeros(self.src_img.shape, dtype=np.uint8)
        basic_array = self.lbp_basic()
        width, height = self.src_img.shape[:2]

        k = np.left_shift(basic_array, 1)
        k = np.where(k > 255, k - 255, k)

        xor = np.bitwise_xor(basic_array, k)
        num = np.unpackbits(xor, axis=-1).reshape(width, height, 8).sum(axis=-1)

        less_equal_two = num <= 2
        uniform_array[less_equal_two.all()] = self.UNIFORM_MAP[basic_array[less_equal_two.all()]]
        uniform_array[~less_equal_two.all()] = 58

        return uniform_array
        # uniform_array = np.zeros(self.src_img.shape, np.uint8)
        # basic_array = self.lbp_basic()
        # width = self.src_img.shape[0]
        # height = self.src_img.shape[1]
        #
        # for i in range(1, width - 1):
        #     for j in range(1, height - 1):
        #         k = basic_array[i, j] << 1
        #         if k > 255:
        #             k = k - 255
        #         xor = basic_array[i, j] ^ k
        #         # 计算二进制中1的个数
        #         num = self.calc_sum(xor)
        #         if num <= 2:
        #             uniform_array[i, j] = self.UNIFORM_MAP[basic_array[i, j]]
        #         else:
        #             uniform_array[i, j] = 58
        # return uniform_array

    # 获取图像的LBP旋转不变等价模式特征
    # def lbp_revolve_uniform(self):
    #     uniform_revolve_array = np.zeros(self.src_img.shape, np.uint8)
    #     basic_array = self.lbp_basic()
    #     width = self.src_img.shape[0]
    #     height = self.src_img.shape[1]
    #     for i in range(1, width - 1):
    #         for j in range(1, height - 1):
    #             k = basic_array[i, j] << 1
    #             if k > 255:
    #                 k = k - 255
    #             xor = basic_array[i, j] ^ k
    #             num = self.calc_sum(xor)
    #             if num <= 2:
    #                 uniform_revolve_array[i, j] = self.calc_sum(num)
    #             else:
    #                 uniform_revolve_array[i, j] = 9
    #     return uniform_revolve_array

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

    # 绘制图像原始LBP特征的归一化统计直方图
    def show_basic_hist(self):
        self.show_hist([256], [0, 256])

    # 绘制图像旋转不变LBP特征的归一化统计直方图
    def show_revolve_hist(self):
        self.show_hist([36], [0, 36])

    # 绘制图像等价模式LBP特征的归一化统计直方图
    def show_uniform_hist(self):
        self.show_hist([60], [0, 60])

    # 绘制图像旋转不变等价模式LBP特征的归一化统计直方图
    def show_revolve_uniform_hist(self):
        self.show_hist([10], [0, 10])

    """
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
                C_LBP_M[int(s_str, 2)] += 1
        # 将三个直方图串联起来
        H = np.concatenate((guiyi(C_LBP_S), guiyi(C_LBP_M), guiyi(C_LBP_C)), axis=0)
        return H
    """

    # def lbp_revolve_circular(self):
    #     height, width = self.src_img.shape
    #     output = np.zeros((height, width), np.uint8)
    #     for _i in range(self.radius, height - self.radius):
    #         for j in range(self.radius, width - self.radius):
    #             center = self.src_img[_i, j]
    #             code = 0
    #             for k in range(self.neighbors):
    #                 x = _i + int(round(self.radius * np.cos(2 * np.pi * k / self.neighbors)))
    #                 y = j - int(round(self.radius * np.sin(2 * np.pi * k / self.neighbors)))
    #                 if self.src_img[x, y] > center:
    #                     code += 1 << k
    #             output[_i][j] = code
    #     # Get histogram of uniform patterns
    #     _hist, _ = np.histogram(output.ravel(), bins=np.arange(0, 11), range=(0, 10))
    #     # Normalize the histogram
    #     _hist /= _hist.astype('float').sum() + self.EPS
    #     self.output_img = output
    #     self.output_hist = _hist

    def lbp_revolve_circular(self):
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

    def lbp_revolve_completed(self):
        height, width = self.src_img.shape
        c_mean = np.mean(self.src_img)
        clbp_c, clbp_s, clbp_m = np.zeros(2), np.zeros(256), np.zeros(256)

        windows = np.lib.stride_tricks.sliding_window_view(self.src_img, (3, 3))
        windows = windows.reshape(height, width, 9) - self.src_img[:, :, np.newaxis]

        src_img_flat = self.src_img.flatten()
        clbp_c[src_img_flat >= c_mean] += 1

        M_windows = np.zeros(windows.shape, dtype=np.int_)
        M_windows[np.abs(windows) >= c_mean] = 1
        windows = np.where(windows >= 0, 1, 0)

        windows = np.delete(windows, 4, axis=-1).reshape(-1, 8)
        M_windows = np.delete(M_windows, 4, axis=-1).reshape(-1, 8)

        s_str = np.packbits(windows, axis=-1).flatten().tobytes().decode()
        M_str = np.packbits(M_windows, axis=-1).flatten().tobytes().decode()

        clbp_s[np.frombuffer(s_str, dtype=np.uint8)] += 1
        clbp_m[np.frombuffer(M_str, dtype=np.uint8)] += 1
        clbp_s = clbp_s / (clbp_s.sum() + self.EPS)
        clbp_m = clbp_m / (clbp_m.sum() + self.EPS)
        clbp_c = clbp_c / (clbp_c.sum() + self.EPS)
        self.output_hist = np.concatenate([clbp_s, clbp_m, clbp_c], axis=0)
        self.output_img = src_img_flat


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
        _lbp = local_binary_pattern(_image, self.p, self.r, self.method)
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


class Describer:
    def __init__(self):
        self.train_origin = np.load('../preprocess/train.npy', allow_pickle=True)
        self.test_origin = np.load('../preprocess/test.npy', allow_pickle=True)
        self.train_X, self.train_y = (np.concatenate(self.train_origin[:, 0], axis=0)
                                      .reshape(-1, 256 ** 2), self.train_origin[:, 1])
        self.test_X, self.test_y = (np.concatenate(self.test_origin[:, 0], axis=0)
                                    .reshape(-1, 256 ** 2), self.test_origin[:, 1])
        self.lbp = LBP()
        logging.basicConfig(level=logging.INFO)
        logging.log(logging.INFO, 'Start processing...')

    def run_with_multi_core(self):
        pool_args = []
        num_workers = 10
        separate = len(self.train_X) // num_workers
        for i in range(num_workers + 1):
            start = i * separate
            end = (i + 1) * separate
            pool_args.append((start, end, self.train_origin, self.lbp, num_workers))
            logging.log(logging.INFO, f'Process {i + 1} added!')

        Pool().map_async(self.worker, pool_args, callback=self.process_train).get()

        pool_args = []
        num_workers = 10
        separate = len(self.test_X) // num_workers
        for i in range(num_workers + 1):
            start = i * separate
            end = (i + 1) * separate
            pool_args.append((start, end, self.test_origin, self.lbp, num_workers))
            logging.log(logging.INFO, f'Process {i + 1} added!')

        Pool().map_async(self.worker, pool_args, callback=self.process_test).get()

    def run_with_one_core(self):
        from alive_progress import alive_bar
        train_parsed = []
        with alive_bar(10, title='Preparing train data') as _bar:
            # with alive_bar(len(self.train_X), title='Preparing train data') as bar:
            for i in range(10):
                resolve_uniform_array = self.lbp.lbp_revolve_uniform(self.train_X[i].reshape(256, -1))
                train_parsed.append(resolve_uniform_array)
                _bar()
            train_parsed = np.array(train_parsed)
            train_parsed = train_parsed.reshape(train_parsed.shape[0], -1)
        logging.log(logging.INFO, 'Train data processed!')
        test_parsed = []
        with alive_bar(len(self.test_X), title='Preparing test data') as _bar:
            for i in range(len(self.test_X)):
                resolve_uniform_array = self.lbp.lbp_revolve_uniform(self.test_X[i].reshape(256, -1))
                test_parsed.append(resolve_uniform_array)
                _bar()
        logging.log(logging.INFO, 'Test data processed!')
        parsed_train_data = []
        parsed_test_data = []
        for i in range(len(self.train_X)):
            parsed_train_data.append([self.train_X[i], self.train_y[i]])
        for i in range(len(self.test_X)):
            parsed_test_data.append([self.test_X[i], self.test_y[i]])
        np.save('../preprocess/train_lbp_.npy', np.asarray(parsed_train_data, dtype=object), allow_pickle=True)
        np.save('../preprocess/test_lbp_.npy', np.asarray(parsed_test_data, dtype=object), allow_pickle=True)

    @staticmethod
    def worker(args):
        _start, _end, _data, _lbp, _sup = args
        sep = min(_end, len(_data)) - _start
        X, y = np.concatenate(_data[:, 0], axis=0).reshape(-1, 256 ** 2), _data[:, 1]
        result = []
        for j in range(_start, min(_end, len(_data))):
            resolve_uniform_array = _lbp.lbp_revolve_uniform(X[j].reshape(256, -1))
            result.append([resolve_uniform_array, y[j]])
            if j % 100 == 0:
                print('Info |',
                      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      'Process:', int(_start / (len(_data) // _sup)) + 1,
                      'Progress:', f'{j - _start + 1}/{sep}',
                      'label:', y[j])
        return result

    @staticmethod
    def process_train(vs):
        logging.log(logging.INFO, f'Process finished!')
        data_parsed = [u for v in vs for u in v]
        data_parsed = np.asarray(data_parsed, dtype=object)
        np.save(f'../preprocess/train_lbp.npy', data_parsed, allow_pickle=True)

    @staticmethod
    def process_test(vs):
        logging.log(logging.INFO, f'Process finished!')
        data_parsed = [u for v in vs for u in v]
        data_parsed = np.asarray(data_parsed, dtype=object)
        np.save(f'../preprocess/test_lbp.npy', data_parsed, allow_pickle=True)


class DescriberTest(unittest.TestCase):
    def test_lbp(self):
        data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        lbp = LBP(method='revolve_completed')
        lbp.describe(data)
        lbp.lbp_revolve_uniform()
        self.assertIsNotNone(lbp.output_img)

    # def test_clbp(self):
    #     clbp = CLBP()
    #     data = np.random.randint(0, 255, (256, 256))
    #     clbp_parsed = clbp.describe(data)
    #     self.assertIsNotNone(clbp_parsed)

    # def test_elbp(self):
    #     elbp = ELBP()
    #     data = np.random.randint(0, 255, (256, 256))
    #     elbp_parsed = elbp.compute_features(data)
    #     self.assertIsNotNone(elbp_parsed)
