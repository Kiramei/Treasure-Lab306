import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

from core import PCA
from core.prepare import *

if not (os.path.exists('./preprocess/train.npy') and os.path.exists('./preprocess/test.npy')):
    dp = DataPreparation()

# 假设有一批图像数据集，其中X为图像数据，y为对应的类别标签
X, y = np.load('./preprocess/train.npy', allow_pickle=True).T  # 图像数据, shape: (n_samples, height, width, channels)
V, v = np.load('./preprocess/test.npy', allow_pickle=True).T  # 类别标签, shape: (n_samples,)

if not (os.path.exists('./lbp_tr_elbp.npy') and os.path.exists('./lbp_te_elbp.npy')):
    from core import lbp_elbp_numba
    lbp_elbp_numba.num_points = 8

n_components = 0
if not (os.path.exists('./train_f_pca.npy') and os.path.exists('./test_f_pca.npy')):
    X_features, y = np.load('./lbp_tr_elbp.npy', allow_pickle=True).T
    V_features, v = np.load('./lbp_te_elbp.npy', allow_pickle=True).T

    X_features = np.array([*X_features])
    V_features = np.array([*V_features])
    feat_num = X_features.shape[1]

    pca = PCA(X_features, contrib=0.999)
    X_pca, P = pca.reduce_dimension()
    X_pca = X_pca.astype(np.float32)
    V_pca = np.dot(P, np.transpose(V_features)).T.astype(np.float32)

    # save
    np.save('./train_f_pca.npy', X_pca)
    np.save('./test_f_pca.npy', V_pca)

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
if not (os.path.exists('./mlp.pkl')):
    X_pca = np.load('./train_f_pca.npy', allow_pickle=True)
    # 创建MLP分类器并进行训练
    print('Start training...')
    mlp = MLPClassifier(hidden_layer_sizes=(1000, 1000), max_iter=1000, activation='relu', solver='adam',
                        random_state=1, verbose=True)
    mlp.fit(X_pca, y)
    # save
    joblib.dump(mlp, './mlp.pkl')
    # 在测试集上进行预测

V_pca = np.load('./test_f_pca.npy', allow_pickle=True)
mlp = joblib.load('./mlp.pkl')
y_pred = mlp.predict(V_pca)

# 计算分类准确率
accuracy = accuracy_score(v, y_pred)
# 混淆矩阵
confusion_matrix = confusion_matrix(v, y_pred)
ConfusionMatrixDisplay(confusion_matrix).plot()
plt.title('Confusion Matrix of ')
plt.show()
print("Accuracy:", accuracy)
