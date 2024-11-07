import sys

import numpy as np

sys.path.append('../MatrixSlowLib')
import matrixslow as ms

batchsize = 16
status_dimension = 32

train_X = []
train_y = []

with open('./train_emoji.csv', 'r') as f:
    data = f.read().split('\n')
    for datum in data:
        train_data = datum.split(',')
        train_X.append(train_data[0])
        train_y.append(int(train_data[1]))

MAX_STRING_LENGTH = 60
for ind, string in enumerate(train_X):
    length = len(string)
    if length < MAX_STRING_LENGTH:
        for i in range(MAX_STRING_LENGTH - length):
            string += ' '
    train_X[ind] = string

seqlength = MAX_STRING_LENGTH
classes = list(set(train_y))  # 不重复的字符
out_classes = len(classes)

# mapping data
# ascii 的反函数为 chr()
chars = np.concatenate(
    (np.array([chr(i) for i in range(65, 91)]),
     np.array([chr(i) for i in range(97, 123)]),
     np.array([chr(i) for i in range(48, 58)]),
     np.array([' ']))
)

M = len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}


inputs = [ms.core.Variable(dim=(M, 1), init=False, trainable=False) for i in range(seqlength)]
U = ms.core.Variable(dim=(status_dimension, M), init=True, trainable=True)
W = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)
b = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

# 构造计算图
last_step = None
for input_vec in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, input_vec), b)

    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)

    h = ms.ops.ReLU(h)
    last_step = h

fc1 = ms.layer.fc(last_step, status_dimension, 40, "ReLU")
output = ms.layer.fc(fc1, 40, out_classes, "ReLU")
predict = ms.ops.Logistic(output)
label = ms.core.Variable((out_classes, 1), trainable=False)

# 训练
loss = ms.ops.CrossEntropyWithSoftMax(output, label)
learning_rate = 0.01
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)
batch_size = 32
for epoch in range(50):
    for batch_k in range(batch_size):
        seq = train_X[batch_k: batch_k + seqlength]
        for j, x in enumerate(inputs):
            # 用one-hot 向量表示当前字母
            d = np.zeros((M, 1))
            d[char_to_ix[seq[j]]] = 1
            x.set_value(np.mat(d))

        d = np.zeros((M, 1))
        target_ch = data[start + seqlength]
        d[char_to_ix[target_ch]] = 1
        label.set_value(np.mat(d))
        optimizer.one_step()
    optimizer.update()

    # pick a random sub-sequence, and test
    pred = []
    trues = []
    pos = np.random.randint(len(data) - 1 - seqlength - batch_size)
    for k in range(1000):
        start = pos + k
        seq = data[start: start + seqlength]
        for j, x in enumerate(inputs):
            d = np.zeros((M, 1))
            d[char_to_ix[seq[j]]] = 1
            x.set_value(np.mat(d))
        predict.forward()
        pred.append(predict.value.A.ravel())
        target_c = data[start + seqlength]
        trues.append(char_to_ix[target_c])

    pred = np.array(pred).argmax(axis=1)
    acc = 0
    for k in range(len(trues)):
        if trues[k] == pred[k]:
            acc += 1.0
    acc = acc / len(trues)
    print("epoch %3f acc: %.3f" % (epoch, acc))
