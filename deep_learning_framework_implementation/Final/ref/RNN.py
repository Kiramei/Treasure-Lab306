
import sys
import time

import matplotlib.pyplot as plt

sys.path.append('../../')

import numpy as np
import matrixslow as ms
from scipy import signal
import os

def get_data(data_root, length=10, train_set_ratio=0.7, seed=42):
    data = np.loadtxt(data_root,delimiter=",", usecols=(2, 3, 4, 5, 6, 7, 8, 9), dtype=str)[1:]
    data = data.astype(np.float_)
    number_of_examples, dimension = data.shape
    data = (data-np.min(data, axis=0))/(np.max(data, axis=0)-np.min(data, axis=0))
    train_set_size = int(number_of_examples * train_set_ratio)
    idx = train_set_size//((length+1)*dimension)*((length+1)*dimension)
    idx_ = (number_of_examples - train_set_size)//((length+1)*dimension)*((length+1)*dimension)
    return (data[:idx].reshape(-1, length+1, dimension),
            data[number_of_examples-idx_:].reshape(-1, length+1, dimension),dimension)

data_root = 'Daily_ZX.csv'
status_dimension = 12
seq_len = 50

train,test, dimension = get_data(data_root=data_root, length=seq_len)

# 输入向量节点
inputs = [ms.core.Variable(dim=(dimension, 1), init=False, trainable=False) for i in range(seq_len)]

# 输入权值矩阵
U = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

# 保存各个时刻内部状态变量的数组
last_step = None  # 上一步的输出，第一步没有上一步，先将其置为 None
for input_vec in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, input_vec), b)

    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)

    h = ms.ops.ReLU(h)
    last_step = h

fc1 = ms.layer.fc(last_step, status_dimension, 40, "ReLU")  # 第一全连接层
fc2 = ms.layer.fc(fc1, 40, 10, "ReLU")
output = ms.layer.fc(fc2, 10, 1, "None")
predict = ms.ops.ReLU(output)
label = ms.core.Variable((1, 1), trainable=False)

loss = ms.ops.loss.MAELoss(output, label)
learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)
train_MAE, test_MAE = [], []
batch_size = 16
train_loss = 0
best_loss = 1
pred_ = []
true_ = []
start_time = time.time()
for epoch in range(30):
    batch_count = 0
    for i, seq in enumerate(train):
        # seq: 10x8, 序列特征数据
        for j, x in enumerate(inputs):
            x.set_value(np.mat(seq[j]).T)

        label.set_value(np.mat(seq[-1, 2]).T)
        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))
            train_loss = loss.value[0, 0]
            optimizer.update()
            batch_count = 0

    train_MAE.append(train_loss)
    pred = []
    true = []
    for i, seq in enumerate(test):
        # 将每个样本各时刻的向量赋给相应变量
        for j, x in enumerate(inputs):
            x.set_value(np.mat(seq[j]).T)

        predict.forward()
        pred.append(predict.value.A.ravel())
        true.append(seq[-1, 2].T)

    pred_ = pred
    true_ = true
    pred = np.array(pred)
    true = np.array(true).reshape(len(test), 1)
    MAE = np.sum(np.abs(pred-true))
    test_MAE.append(MAE/len(test))
    print("epoch: {:d}, accuracy: {:.5f}".format(epoch + 1, MAE/len(test)))

cost_time = time.time()-start_time
# plt.subplot(1, 2, 1)
# plt.xlabel("Epoch")
# plt.ylabel('MAE')
# plt.title(f'{seq_len} days')
# plt.plot(np.arange(len(train_MAE)),train_MAE, np.arange(len(test_MAE)), test_MAE)
# plt.legend(['Train_MAE', 'Test_MAE'])
# plt.subplot(1, 2, 2)

plt.xlabel("day")
plt.ylabel("open")
plt.title(f'{seq_len} days')
plt.plot(np.arange(len(pred_)),pred_, np.arange(len(true_)), true_)
plt.legend(['pred', 'true'])
plt.show()

print("best_MAE=", np.min(test_MAE), "cost_time=", cost_time)
