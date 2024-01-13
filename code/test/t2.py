import os
import cv2
import numpy as np
from alive_progress import alive_bar
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. 数据集准备
data_dir = '../../dataset'
categories = ['bird', 'dog', 'cat', 'snake']
image_size = (256, 256)


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


# 2. 特征提取 - LBP算法
def calculate_lbp_histogram(image):
    # 在这里实现LBP特征提取算法
    # 返回图像的LBP特征直方图
    elbp = ELBP()
    data = np.random.randint(0, 255, (256, 256))
    elbp_parsed = elbp.compute_elbp_features(data)
    return elbp_parsed


# 遍历数据集，提取每张图像的LBP特征直方图
dataset = []
labels = []
for category in categories:
    category_dir = os.path.join(data_dir, category)
    with alive_bar(len(os.listdir(category_dir)), title='Processing lbp') as bar:
        for file_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, file_name)
            image = cv2.imread(image_path, 0)  # 以灰度图像读入
            image = cv2.resize(image, image_size)
            lbp_histogram = calculate_lbp_histogram(image)
            dataset.append(lbp_histogram)
            labels.append(categories.index(category))
            bar()
np.save('../preprocess/train_lbp_1.npy', np.asarray(dataset, dtype=object), allow_pickle=True)
np.save('../preprocess/test_lbp_1.npy', np.asarray(labels, dtype=object), allow_pickle=True)

# 3. 特征降维 - PCA算法
def apply_pca(dataset, n_components):
    pca = PCA(n_components=n_components)
    transformed_dataset = pca.fit_transform(dataset)
    return transformed_dataset


# 对提取的LBP特征进行降维处理
n_components = 100  # 设置要保留的主成分数
dataset = apply_pca(dataset, n_components)

# 4. 训练分类模型 - 支持向量机（SVM）
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
model = SVC()
model.fit(X_train, y_train)

# 5. 模型性能评估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
micro_f1 = f1_score(y_test, y_pred_test, average='micro')
confusion_mat = confusion_matrix(y_test, y_pred_test)

# 打印模型性能指标
print('Train Accuracy:', train_accuracy)
print('Test Accuracy:', test_accuracy)
print('Micro F1 Score:', micro_f1)
print('Confusion Matrix:')
print(confusion_mat)

# 6. 分析模型和特征的影响
# 在这里进行模型参数和特征参数的修改和分析
