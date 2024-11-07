import numpy as np
import pandas as pd


class PCA:
    def __init__(self, data, n_components=None, contrib=0.9):
        """
        Args:
            data: 输入的样本 [样本数量， 维度]
            n_components: 需要降维到的维度
            contrib: 根据贡献度自动提取维度
        """
        self.data = data
        self.dimension = data.shape[1]
        self.n_components = n_components
        self.contribution = contrib
        if n_components and n_components >= self.dimension:
            raise ValueError("n_components error")
        if contrib > 1:
            raise ValueError("contribution error")

    def compute_covariance(self):
        data_T = self.data.T
        covariance_matrix = np.cov(data_T)
        return covariance_matrix

    def compute_eigenvalues_eigenvectors(self):
        covariance_matrix = self.compute_covariance()
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        m = eigenvalues.shape[0]
        combined = np.hstack((eigenvalues.reshape((m, 1)), eigenvectors))
        combined_df = pd.DataFrame(combined)
        combined_df_sorted = combined_df.sort_values(0, ascending=False)
        return combined_df_sorted

    def explained_variance_ratio(self):
        combined_df_sorted = self.compute_eigenvalues_eigenvectors()
        eigenvalues = combined_df_sorted.values[:, 0]
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        return explained_variance_ratio

    def reduce_dimension(self):
        combined_df_sorted = self.compute_eigenvalues_eigenvectors()

        if self.n_components:
            eigenvectors = combined_df_sorted.values[0:self.n_components, 1:]
            projected_data = np.dot(eigenvectors, self.data.T)
            return projected_data.T, eigenvectors

        explained_variance_ratio = self.explained_variance_ratio()

        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        dimension = np.argmax(cumulative_variance_ratio >= self.contribution) + 1

        eigenvectors = combined_df_sorted.values[0:dimension, 1:]
        projected_data = np.dot(eigenvectors, self.data.T)
        return projected_data.T, eigenvectors
