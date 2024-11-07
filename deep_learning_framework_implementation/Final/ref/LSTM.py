
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

def pre_data(train, test, number=3):
    train_, test_ = [], []
    for i in range(len(train)):
        tem = []
        for j in range(len(train[i])//number):
            if (j+1)*number>len(train[i]):
                tem.append(train[i][-1])
            tem.append(np.mean(train[i, j*number:(j+1)*number], axis=0))
        train_.append(tem)

    for i in range(len(test)):
        tem = []
        for j in range(len(test[i]) // number):
            if (j + 1) * number > len(test[i]):
                tem.append(test_[i][-1])
            tem.append(np.mean(test[i, j * number:(j + 1) * number], axis=0))
        test_.append(tem)

    return np.array(train_), np.array(test_)

data_root = 'Daily_ZX.csv'
status_dimension = 12
seq_len = 5

train,test, dimension = get_data(data_root=data_root, length=seq_len)
if(seq_len>5):
    train, test = pre_data(train, test, seq_len//5)

# 输入向量节点
inputs = [ms.core.Variable(dim=(dimension, 1), init=False, trainable=False) for i in range(5)]

# 输入权值矩阵
U_f = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W_f = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b_f = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

U_i = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W_i = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b_i = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

U_c_ = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W_c_ = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b_c_ = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

U_o = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W_o = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b_o = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

# 保存各个时刻内部状态变量的数组
last_step = None
C_t = None
start_time = time.time()
for input_vec in inputs:
    f_t = ms.ops.Add(ms.ops.MatMul(U_f, input_vec), b_f)
    i_t = ms.ops.Add(ms.ops.MatMul(U_i, input_vec), b_i)
    c_t = ms.ops.Add(ms.ops.MatMul(U_c_, input_vec), b_c_)
    o_t = ms.ops.Add(ms.ops.MatMul(U_o, input_vec), b_o)
    if last_step is not None:
        f_t = ms.ops.Add(ms.ops.MatMul(W_f, last_step), f_t)
        i_t = ms.ops.Add(ms.ops.MatMul(W_i, last_step), i_t)
        c_t = ms.ops.Add(ms.ops.MatMul(W_c_, last_step), c_t)
        o_t = ms.ops.Add(ms.ops.MatMul(W_o, last_step), o_t)
    f_t = ms.ops.Logistic(f_t)
    i_t = ms.ops.Logistic(i_t)
    c_t = ms.ops.tanh(c_t)
    if C_t is not None:
        C_t = ms.ops.Add(ms.ops.Multiply(f_t, c_t), ms.ops.Multiply(i_t, c_t))
    else:
        C_t = ms.ops.Multiply(i_t, c_t)
    last_step = ms.ops.Multiply(o_t, ms.ops.tanh(C_t))

fc1 = ms.layer.fc(last_step, status_dimension, 40, "ReLU")  # 第一全连接层
fc2 = ms.layer.fc(fc1, 40, 10, "ReLU")
output = ms.layer.fc(fc2, 10, 1, "None")
predict = ms.ops.ReLU(output)
label = ms.core.Variable((1, 1), trainable=False)
loss = ms.ops.loss.MAELoss(output, label)
learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)
test_MAE, train_MAE = [],[]
loss_ = 1
batch_size = 16
pred_, true_ = [], []

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
            loss_ = loss.value[0, 0]
            optimizer.update()
            batch_count = 0
    train_MAE.append(loss_)
    pred = []
    true = []

    for i, seq in enumerate(test):
        # 将每个样本各时刻的向量赋给相应变量
        for j, x in enumerate(inputs):
            x.set_value(np.mat(seq[j]).T)

        predict.forward()
        pred.append(predict.value.A.ravel())
        true.append(seq[-1, 2].T)
    pred = np.array(pred)
    true = np.array(true).reshape(len(test), 1)
    pred_ = pred
    true_ = true
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