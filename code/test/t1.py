from PIL import Image
import numpy as np


def pca(X):
    """主成分分析
    输入：矩阵X，存储训练数据，每一行为一条数据
    返回：投影矩阵（按维度重要性排序的）方差和均值
    """
    # 获取维数
    num_data, dim = X.shape
    # print(num_data,dim)
    # 数据中心化（减去每一维的均值）
    mean_X = X.mean(axis=0)
    X = X - mean_X
    if dim > num_data:
        # 使用紧致技巧
        # 协方差矩阵
        # M = np.dot(X,X.T)
        M = np.cov(X, rowvar=True)
        # 特征值和特征向量
        e, EV = np.linalg.eigh(M)
        # print(E,EV)
        # 紧致技巧
        tmp = np.dot(X.T, EV).T
        V = tmp[::-1]
        # 求平方根需要求其绝对值
        S = np.sqrt(np.abs(e))[::-1]
        for i in range(V.shape[1]):
            V[:, i] /= S
        print("投影矩阵", V)
        print('特征向量', EV)
        return V, S, mean_X
    else:
        # 使用SVD方法
        U, S, V = np.linalg.svd(X)
        # 返回前num_data维的数据
        V = V[:num_data]
        # 返回投影矩阵，方差和均值
        print("投影矩阵", V)
        return V, S, mean_X


hua = np.array(
    Image.open(r'E:\WorkSpace\py\Experiment\py\Last\dataset\bird\Acadian_Flycatcher_0006_2520332573.jpg').convert('L'))
pca(hua)
