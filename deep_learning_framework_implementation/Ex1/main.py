import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 展示数据形状
print(train_data.shape)
print(test_data.shape)

# 展示部分特征
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 数据预处理
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values.astype(np.float32)).to('cuda:0')
test_features = torch.tensor(all_features[n_train:].values.astype(np.float32)).to('cuda:0')
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1).astype(np.float32)).to('cuda:0')

# 定义损失函数
loss = nn.MSELoss()


# 定义模型
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


# 定义对数均方根误差
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def log_mae(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    mae = mean_absolute_error(clipped_preds.cpu().detach().numpy(), labels.cpu().detach().numpy())
    return mae


# 定义训练函数
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls, train_mae_loss = [], [], []
    dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        train_mae_loss.append(log_mae(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
        if epoch % 10 == 0 and epoch:
            print('epoch %d, train RMSE %f, MAE %f' % (epoch, train_ls[-1], train_mae_loss[-1]))
    return train_ls, test_ls, train_mae_loss


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 8

# 在完整训练集上训练模型
net = Net(train_features.shape[1]).to('cuda:0')
train_ls, _, train_mae_ls = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay,
                                  batch_size)
# plot the training curve
plt.plot(train_ls)
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.show()

plt.plot(train_mae_ls)
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.show()

# 预测测试集
preds = net(test_features).detach().cpu().numpy()
result = pd.DataFrame({'Id': test_data.Id, 'SalePrice': pd.Series(preds.reshape(1, -1)[0])})
result.to_csv('result.csv', index=False)

import seaborn as sns

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.distplot(result['SalePrice'], bins=100, color='b')
# title
plt.title('Predicted SalePrice Distribution')
plt.subplot(1, 2, 2)
sns.distplot(train_data['SalePrice'], bins=100, color='r')
# title
plt.title('Train SalePrice Distribution')
plt.show()
