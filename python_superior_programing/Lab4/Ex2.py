import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self):
        self.centroids = None
        self.cluster_labels = None

    def __call__(self, _X, _K):
        self.n_clusters = _K
        self.fit(_X)
        return self.cluster_labels, self.centroids

    def fit(self, X):
        np.random.seed(0)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices]

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


def calSSE(_X, _cidx):
    _K = len(np.unique(_cidx))
    SSE = 0.0

    for k in range(_K):
        cluster_points = _X[_cidx == k]
        centroid = np.mean(cluster_points, axis=0)
        SSE += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)

    return SSE / len(_X)


if __name__ == '__main__':
    # 读取数据
    X = np.loadtxt('Lab4.dat')
    # 测试不同的K值
    K_values = [2, 3, 4]
    fig, axes = plt.subplots(nrows=len(K_values), figsize=(4, 3 * len(K_values)))
    for i, K in enumerate(K_values):
        kmeans = KMeans()
        cidx, ctrs = kmeans(X, K)
        sse = calSSE(X, cidx)
        axes[i].scatter(X[:, 0], X[:, 1], c=cidx, marker='.')
        axes[i].scatter(ctrs[:, 0], ctrs[:, 1], c='red', marker='x')
        axes[i].set_title(f'K = {K}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
    plt.tight_layout()
    plt.show()

    # 测试不同的K值 [1,6]
    K_values = [1, 2, 3, 4, 5, 6]
    SSE_values = []
    for K in K_values:
        kmeans = KMeans()
        cidx, ctrs = kmeans(X, K)
        sse = calSSE(X, cidx)
        SSE_values.append(sse)
    plt.plot(K_values, SSE_values, marker='.')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.show()

