import numpy as np
import pandas as pd
import torch
from graphviz import Digraph
from pandas import DataFrame
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import Dataset


class CancerDataset(Dataset):
    def __init__(self, csv_file_path, std=True):
        # 使用Pandas库加载CSV文件
        df = pd.read_csv(csv_file_path)
        # 将标签'M'和'B'转换为数值标签
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        # 筛选出确诊为恶性肿瘤的数据
        malignant_df = df[df['diagnosis'] == 1]
        # 筛选出确诊为健康的数据
        benign_df = df[df['diagnosis'] == 0]
        # 分别随机打乱
        malignant_df = malignant_df.sample(frac=1).reset_index(drop=True)
        benign_df = benign_df.sample(frac=1).reset_index(drop=True)
        # 两类数据均匀散步在合并数据中，每隔1到2个数据插入一条恶性肿瘤数据
        dft = DataFrame()
        a, b = 0, 0
        for i in range(0, len(df)):
            rd = np.random.randint(0, 3)
            if rd == 0 and a < len(malignant_df):
                dft = dft.append(malignant_df.iloc[a])
                a = a + 1
            elif b < len(benign_df):
                dft = dft.append(benign_df.iloc[b])
                b = b + 1
            else:
                dft = dft.append(malignant_df.iloc[a])
                a = a + 1

        # 随机化顺序
        # df = df.sample(frac=1).reset_index(drop=True)
        data = dft.drop(['id', 'diagnosis'], axis=1)
        label = dft['diagnosis']
        # 获取标签名
        self.label_names = dft.columns.tolist()
        # 获取特征数据
        self.data = torch.tensor(data.values, dtype=torch.float32)
        # 标准化
        if std:
            self.data = (self.data - self.data.mean(dim=0)) / self.data.std(dim=0)
        self.label = torch.tensor(label.values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 根据索引获取单个样本
        sample = self.data[index]
        return sample


def visualize_tree(tree, label_names):
    dot = Digraph()

    def add_node(node, parent=None, branch_label=None):
        if node.value is not None:
            node_label = f"Diagnose: {'M' if node.value == 1 else 'B'}"
        else:
            node_label = f"{label_names[node.feature_index]}\n<= {node.threshold}"
        dot.node(str(node), node_label)
        if parent is not None:
            dot.edge(str(parent), str(node), label=branch_label)
        if node.true_branch is not None:
            add_node(node.true_branch, parent=node, branch_label="True")
        if node.false_branch is not None:
            add_node(node.false_branch, parent=node, branch_label="False")

    add_node(tree)
    dot.render("decision_tree_graph", format="png", view=True)


class DecisionNode:
    def __init__(self, feature_index=None,
                 threshold=None,
                 value=None,
                 true_branch=None,
                 false_branch=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree:
    def __init__(self, max_depth=None):
        self.tree = None
        self.num_classes = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return [self._predict_sample(x, self.tree) for x in X]

    def _build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            value = self._most_common_label(y)
            return DecisionNode(value=value)

        num_features = X.shape[1]
        best_feature_index, best_threshold = self._find_best_split(X, y, num_features)

        true_indices = X[:, best_feature_index] <= best_threshold
        false_indices = ~true_indices

        true_branch = self._build_tree(X[true_indices], y[true_indices], depth + 1)
        false_branch = self._build_tree(X[false_indices], y[false_indices], depth + 1)

        return DecisionNode(feature_index=best_feature_index, threshold=best_threshold, true_branch=true_branch,
                            false_branch=false_branch)

    def _find_best_split(self, X, y, num_features):
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)
            # 遍历所有可能的分割点
            for threshold in thresholds:
                true_indices = feature_values <= threshold
                false_indices = ~true_indices
                # 计算基尼指数
                gini = self._gini_index(y[true_indices]) * len(true_indices) / len(y) + \
                       self._gini_index(y[false_indices]) * len(false_indices) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # 计算基尼指数, 公式：1 - sum(x^2)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
        most_common_label = labels[np.argmax(counts)]
        return most_common_label

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.true_branch)
        else:
            return self._predict_sample(x, node.false_branch)


def train_tree(_tree, _optimizer, _criterion, data, labels, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tree.to(device)
    data = data.to(device)
    labels = labels.to(device)
    writer = SummaryWriter(log_dir='logs')
    for epoch in range(num_epochs):
        _optimizer.zero_grad()
        outputs = _tree(data)
        loss = _criterion(outputs, labels)
        loss.backward()
        _optimizer.step()
        writer.add_scalar('loss', loss.item(), epoch)
        if epoch % 1000 == 0:
            with torch.no_grad():
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()), end='\t')
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                writer.add_scalar('accuracy', 100 * correct / total, epoch)
                print('Accuracy: {:.2f}%'.format(100 * correct / total))
    writer.close()


def predict_tree(_tree, data, labels=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tree.to(device)
    data = data.to(device)
    outputs = _tree(data)
    _, _predicted = torch.max(outputs, 1)
    # 如果没有提供标签，则直接返回预测结果
    if labels is None:
        return _predicted.cpu()
    # 计算准确率
    total = labels.size(0)
    correct = (_predicted.cpu() == labels).sum().item()
    print('Accuracy: {:.2f}%'.format(100 * correct / total))
    return _predicted.cpu()


class MLPClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLPClassifier, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.tree = nn.ModuleList()

    def forward(self, x):
        leaf_outputs = []
        for layer in self.tree:
            x = layer(x)
            leaf_outputs.append(x)
        return torch.cat(leaf_outputs, dim=1)

    def add_layer(self, layer):
        self.tree.append(layer)


class MLPLayer(nn.Module):
    def __init__(self, num_features, num_classes, hidden_feature=64):
        super(MLPLayer, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.linear_1 = nn.Linear(num_features, hidden_feature)
        self.linear_2 = nn.Linear(hidden_feature, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear_2(x)
        x = self.softmax(x)
        return x


def pred_with_decision_tree(_dt: CancerDataset):
    _tree = DecisionTree(max_depth=3)
    train_size = int(0.8 * len(_dt))
    _tree.fit(_dt.data[:train_size], _dt.label[:train_size])
    _predicted = np.array(_tree.predict(_dt.data[train_size:]))
    # 计算准确率
    total = _dt.label[train_size:].size(0)
    _ground_truth = np.array(_dt.label[train_size:])
    correct = np.sum(_predicted == _ground_truth)
    visualize_tree(_tree.tree, label_names=_dt.label_names)
    print('Accuracy: {:.2f}%'.format(100 * correct / total))


def pred_with_mlp_net(_dt: CancerDataset):
    _tree = MLPClassifier(num_features=30, num_classes=2)
    _tree.add_layer(MLPLayer(num_features=30, num_classes=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tree.to(device)
    optimizer = torch.optim.Adam(_tree.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_size = int(0.8 * len(_dt))
    train_tree(_tree, optimizer, criterion, _dt.data[:train_size], _dt.label[:train_size], 10001)


if __name__ == '__main__':
    dt = CancerDataset("./data.csv", std=False)
    # pred_with_decision_tree(dt)
    pred_with_mlp_net(dt)
