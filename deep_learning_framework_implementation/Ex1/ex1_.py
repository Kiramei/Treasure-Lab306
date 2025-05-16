import matplotlib.pyplot as plt
import numpy as np

import matrixslow as ms

# 生成训练和测试数据
np.random.seed(1234)
t_train = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
t_test = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
N_train = np.random.normal(0, 0.1, size=(1000, 1))
N_test = np.random.normal(0, 0.1, size=(100, 1))
x_train = np.cos(t_train) + np.sin(t_train) + N_train
x_test = np.cos(t_test) + np.sin(t_test) + N_test

print('Feat shape:', t_train.shape, t_test.shape)
print('Label shape:', x_train.shape, x_test.shape)

import torch
import torch.nn as nn

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# 模型
net = MLPNet()

# 损失函数
criterion = nn.MSELoss()

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

# 批大小
batch_size = 32

# 训练
for epoch in range(60):

    batch_count = 0
    i = 0
    for _t, _x in zip(t_train, x_train):

        inputs = torch.tensor(_t).reshape(-1, 1).float()
        labels = torch.tensor(_x).reshape(-1, 1).float()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:f}".format(epoch + 1, i + 1, loss.item()))
            batch_count = 0
        i += 1
    pred = []
    for _t in t_test:
        inputs = torch.tensor(_t).reshape(-1, 1).float()
        outputs = net(inputs)
        pred.append(outputs.item())

# 测试
y_test = net(torch.tensor(t_test).reshape(-1, 1).float()).detach().numpy()

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(t_test, x_test, label='True')
plt.plot(t_test, y_test, label='Predict')
plt.legend()
plt.show()