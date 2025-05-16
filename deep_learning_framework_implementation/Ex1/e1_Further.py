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

# 输入输出的数据大小
input_shape = (1, 1)
output_shape = (1, 1)

# 输入网络
x = ms.core.Variable(input_shape, init=False, trainable=False)
y = ms.core.Variable(output_shape, init=False, trainable=False)

# 全连接层 1 =RELU=> 256 =RELU=> 64 ==> 1
fc1 = ms.layer.fc(x, 1, 256, "ReLU")
fc2 = ms.layer.fc(fc1, 256, 128, "ReLU")
output = ms.layer.fc(fc2, 128, 1, "None")

# 均方差损失
loss = ms.ops.loss.MeanSquaredErrorLoss(output, y)
LAMBDA = ms.Variable(dim=(1,1), init=False, trainable=False)
LAMBDA.set_value(np.mat([1]))
norm = ms.ops.loss.L2NormLoss(fc1, LAMBDA)
total_loss = ms.ops.Add(loss, norm, name='total_loss')

# 学习率，该项的设置较为重要
learning_rate = 0.01

# 优化器
optimizer = ms.optimizer.Adam(ms.default_graph, total_loss, learning_rate)

# 批大小，由于每个样本代表的时间较为稀疏，为了模型能够捕捉全局能力，批大小设置为训练集大小
batch_size = 1000

best_accuracy = 9999999
# 训练
for epoch in range(60):
    batch_count = 0
    i = 0
    for _t, _x in zip(t_train, x_train):
        # 输入特征为训练集时间轴数据
        x.set_value(np.mat(_t).reshape(input_shape))
        # 标签为训练集函数值数据
        y.set_value(np.mat(_x).reshape(output_shape))
        # 前向传播，更新损失值
        optimizer.one_step()
        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:f}".format(epoch + 1, i + 1, loss.value[0, 0]))
            # 反向传播，更新参数
            optimizer.update()
            batch_count = 0
        i += 1
    # 在测试集上验证模型效果
    pred = []
    for _t in t_test:
        x.set_value(np.mat(_t).reshape(input_shape))
        output.forward()
        pred.append(output.value.A.ravel())
    pred = np.array(pred)
    # 计算均方误差
    accuracy = ((x_test - pred) ** 2).sum() / len(x_test)
    # 只输出最佳模型效果
    if abs(accuracy) < best_accuracy:
        best_accuracy = abs(accuracy)
        plt.plot(t_test, pred, label='pred')
        plt.plot(t_test, x_test, label='true')
        plt.legend()
        plt.show()
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
