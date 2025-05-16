import sys

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matrixslow as ms

sys.path.append('../..')
# 加载MNIST数据集，取一部分样本并归一化
X, y = np.load('X_data.npy', allow_pickle=True), np.load('Y_data.npy', allow_pickle=True)
# for p in range(len(X)):
#     # import matplotlib.pyplot as plt
#     import cv2
#
#     v = (X[p].reshape(28, 28) * 255).astype(np.uint8)
#     s =str(y[p]) + ""
#     cv2.imshow(s, v)
#     cv2.waitKey(0)

X, y = X[:1000] / 255, y.astype(np.int)[:1000]

# 将整数形式的标签转换成One-Hot编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(y.reshape(-1, 1))

# 输入图像尺寸
img_shape = (28, 28)

# 输入图像
x = ms.core.Variable(img_shape, init=False, trainable=False)

# One-Hot标签
one_hot = ms.core.Variable(dim=(10, 1), init=False, trainable=False)

# 第一卷积层
conv1 = ms.layer.conv([x], img_shape, 3, (5, 5), "ReLU")

# 第一池化层
pooling1 = ms.layer.pooling(conv1, (3, 3), (2, 2))

# 第二卷积层
conv2 = ms.layer.conv(pooling1, (14, 14), 3, (3, 3), "ReLU")

# 第二池化层
pooling2 = ms.layer.pooling(conv2, (3, 3), (2, 2))

# 全连接层
fc1 = ms.layer.fc(ms.ops.Concat(*pooling2), 147, 120, "ReLU")

# 输出层
output = ms.layer.fc(fc1, 120, 10, "None")

# 分类概率
predict = ms.ops.SoftMax(output)

# 交叉熵损失
loss = ms.ops.loss.FocalLoss(output, one_hot, alpha=np.array(
    [0.110, 0.131, 0.112, 0.105, 0.119, 0.104, 0.106, 0.000, 0.099, 0.113]
), gamma=2.0)
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)
# 学习率
learning_rate = 0.01

# 优化器
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

# 批大小
batch_size = 1000

# 训练
for epoch in range(60):

    batch_count = 0

    for i in range(len(X)):

        feature = np.mat(X[i]).reshape(img_shape)
        label = np.mat(one_hot_label[i]).T

        x.set_value(feature)
        one_hot.set_value(label)

        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(X)):
        feature = np.mat(X[i]).reshape(img_shape)
        x.set_value(feature)

        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    accuracy = (y == pred).astype(np.int).sum() / len(X)
    pred_fail = y[pred != y]
    # 统计预测错误的样本概率
    for i in range(10):
        print("预测错误的样本中，标签为{:d}的样本概率为{:.3f}".format(i, (pred_fail == i).astype(np.int).sum() / len(
            pred_fail)))
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
