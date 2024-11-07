import numpy as np
import matplotlib.pyplot as plt


def get_farthest_k_center(X, n_clusters):
    # 随机初始化第一个点的 x 和 y 坐标
    fir = np.random.choice(len(X), 1, replace=False)
    centers = []
    # 将第一个点的坐标赋值给中心点
    centers.append(*X[fir])

    for i in range(1, n_clusters):
        distances = []  # 存储到已存在中心点的距离
        for j in range(i):
            distances.append(np.linalg.norm(X - centers[j], axis=1))
        # 对于每个点，找到到已存在中心点的最小距离
        dis = np.min(np.array(distances), axis=0)
        idx = dis.argmax()
        centers.append(X[idx])
    return centers


class KMeans:
    def __init__(self):
        self.centroids = None
        self.cluster_labels = None

    def __call__(self, _X, _K):
        self.n_clusters = _K
        self.fit(_X)
        return self.cluster_labels, self.centroids

    def fit(self, X):
        # np.random.seed(0)
        # indices = np.random.choice(len(X), self.n_clusters, replace=False)
        # self.centroids = X[indices]
        self.centroids = get_farthest_k_center(X, self.n_clusters)
        while True:
            dists = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.cluster_labels = np.argmin(dists, axis=1)
            new_centroids = np.array([np.mean(X[self.cluster_labels == k], axis=0) for k in range(self.n_clusters)])

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        dists = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(dists, axis=1)


def cal_Silhouette_Coeff():
    data = np.loadtxt('Lab5.dat')
    K_values = [2, 3, 4, 5, 6]  # 指定要计算的K值
    silhouette_scores = []  # 保存每个K值对应的轮廓系数

    for k in K_values:
        S = 0
        kmeans = KMeans()
        cidx, ctrs = kmeans(data, k)
        # 计算轮廓系数
        for i in range(len(data)):
            # 同簇
            a = np.mean(np.linalg.norm(data[cidx == cidx[i]] - data[i], axis=1))
            # 最近簇
            bs = []
            for j in range(k):
                if j == cidx[i]:
                    continue
                bs.append(np.mean(np.linalg.norm(data[cidx == j] - data[i], axis=1)))
            b = np.min(bs)
            S += (b - a) / max(a, b)
        silhouette_scores.append(S / len(data))
    # 找到最佳K值
    best_k = K_values[np.argmax(silhouette_scores)]
    return best_k, silhouette_scores


best_k, silhouette_scores = cal_Silhouette_Coeff()
# 绘制轮廓系数图
plt.plot(np.arange(2, 7, 1), silhouette_scores)
plt.show()
print("Best K value:", best_k)
print("Silhouette scores:", silhouette_scores)
