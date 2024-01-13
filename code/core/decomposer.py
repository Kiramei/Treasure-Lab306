import numpy as np


class PCA:
    @staticmethod
    def centralization(dataMat):
        rows, cols = np.shape(dataMat)
        mean_val = np.mean(dataMat, 0)  # 按列求均值，即求各个特征的均值
        new_data = dataMat - np.tile(mean_val, (rows, 1))  # 减均值
        return new_data, mean_val

    @staticmethod
    def patch(data, _k):
        dataMat = np.float32(np.mat(data))  # 变矩阵
        # 中心化
        A, _meanVal = PCA.centralization(dataMat)
        # 协方差矩阵
        covMat = A * A.T
        D, V = np.linalg.eig(covMat)
        # 降维后
        _V_r = V[:, 0:_k]  # 按列取前r个特征向量
        _V_r = A.T * _V_r
        for _i in range(_k):
            _V_r[:, _i] = _V_r[:, _i] / np.linalg.norm(_V_r[:, _i])  # 归一化
        final_data = A * _V_r
        final_data = np.array(np.real(final_data))
        return final_data, _meanVal, _V_r

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data


if __name__ == '__main__':
    k = 8
    while True:
        if k == 128:
            break
        r = np.load('../preprocess/train_lbp.npy', allow_pickle=True)
        s = np.load('../preprocess/test_lbp.npy', allow_pickle=True)

        train_data = r[:, 0]
        train_label = r[:, 1]
        test_data = s[:, 0]
        test_label = s[:, 1]
        train_data = np.array([x.flatten() for x in train_data])
        test_data = np.array([x.flatten() for x in test_data])
        print(train_data.shape)
        print(test_data.shape)
        train_data, meanVal, V_r = PCA.patch(train_data, k)
        test_data, meanVal_, V_r_ = PCA.patch(test_data, k)
        print(train_data.shape)
        print(test_data.shape)
        train_data_parsed = []
        test_data_parsed = []
        for i in range(len(train_data)):
            train_data_parsed.append([train_data[i], train_label[i]])
        for i in range(len(test_data)):
            test_data_parsed.append([test_data[i], test_label[i]])
        # save
        np.save(f'../preprocess/train_pca_{k}.npy', np.asarray(train_data_parsed, dtype=object), allow_pickle=True)
        np.save(f'../preprocess/test_pca_{k}.npy', np.asarray(test_data_parsed, dtype=object), allow_pickle=True)
        print('Saved!')
        k *= 2
