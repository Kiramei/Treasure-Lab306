import sys

import numpy as np

sys.path.append('../MatrixSlowLib')
import matrixslow as ms

BATCH_SIZE = 16
STATUS_DIM = 32
HIDDEN_FEATURE = 256
CLASS_TO_PRED = 5
MAX_STRING_LENGTH = 30
NUM_EPOCHS = 1000

train_X = []
train_y = []

with open('./train_emoji.csv', 'r') as f:
    data = f.read().split('\n')
    for datum in data:
        train_data = datum.split(',')
        train_X.append(train_data[0]
                       .replace('\n', '')
                       .replace('\r', '')
                       .replace('\"', '')
                       .replace('\t', ''))
        train_y.append(int(train_data[1]))

test_X = []
test_y = []

with open('./test_emoji.csv', 'r') as f:
    data = f.read().split('\n')
    for datum in data:
        test_data = datum.split(',')
        test_X.append(test_data[0]
                      .replace('\n', '')
                      .replace('\"', '')
                      .replace('\r', '')
                      .replace('\t', ''))
        test_y.append(int(test_data[1]))

for ind, string in enumerate(train_X):
    length = len(string)
    if length < MAX_STRING_LENGTH:
        for i in range(MAX_STRING_LENGTH - length):
            string += ' '
    train_X[ind] = string

for ind, string in enumerate(test_X):
    length = len(string)
    if length < MAX_STRING_LENGTH:
        for i in range(MAX_STRING_LENGTH - length):
            string += ' '
    test_X[ind] = string

seqlength = MAX_STRING_LENGTH
classes = list(set(train_y))  # 不重复的字符
out_classes = len(classes)

# mapping data
# ascii 的反函数为 chr()
chars = np.concatenate(
    (np.array([' ', '\"']),
     np.array([chr(i) for i in range(65, 91)]),
     np.array([chr(i) for i in range(97, 123)]),
     np.array([chr(i) for i in range(48, 58)]))
)

np.random.shuffle(chars)

CLASS_CHARS = len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

inputs = [ms.core.Variable(dim=(CLASS_CHARS, 1), init=False, trainable=False) for i in range(seqlength)]
U = ms.core.Variable(dim=(STATUS_DIM, CLASS_CHARS), init=True, trainable=True)
W = ms.core.Variable(dim=(STATUS_DIM, STATUS_DIM), init=True, trainable=True)
b = ms.core.Variable(dim=(STATUS_DIM, 1), init=True, trainable=True)

# 构造计算图
last_step = None
for input_vec in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, input_vec), b)
    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)
    h = ms.ops.ReLU(h)
    last_step = h

fc1 = ms.layer.fc(last_step, STATUS_DIM, HIDDEN_FEATURE, "ReLU")
output = ms.layer.fc(fc1, HIDDEN_FEATURE, CLASS_TO_PRED, "None")
predict = ms.ops.Logistic(output)
# predict = output
label = ms.core.Variable((CLASS_TO_PRED, 1), trainable=False)

# 训练
loss = ms.ops.CrossEntropyWithSoftMax(output, label)
learning_rate = 0.02
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)
batch_size = 32

losses = []
preds = []
best_acc = 0

for epoch in range(NUM_EPOCHS):
    print('================== Epoch %d ===================' % epoch)
    pred = []
    losses = []
    for ind, (X, y) in enumerate(zip(train_X, train_y)):
        for j, x in enumerate(inputs):
            # 用one-hot 向量表示当前字母
            d = np.zeros((CLASS_CHARS, 1))
            d[char_to_ix[X[j]]] = 1
            x.set_value(np.mat(d))
        input_y = np.zeros((CLASS_TO_PRED, 1))
        input_y[y] = 1
        label.set_value(np.mat(input_y))
        optimizer.one_step()
        losses.append(loss.value[0, 0])
        if ind % 32 == 0:
            optimizer.update()
    for X in test_X:
        for j, x in enumerate(inputs):
            # 用one-hot 向量表示当前字母
            d = np.zeros((CLASS_CHARS, 1))
            d[char_to_ix[list(X)[j]]] = 1
            x.set_value(np.mat(d))
        predict.forward()
        pred.append(predict.value.A.ravel())
    los = np.mean(losses)
    acc = (np.array(pred).argmax(axis=1) == test_y).mean()
    losses.append(los)
    preds.append(acc)
    print("epoch %d: \t loss: %.3f \t acc: %.3f" % (epoch, los, acc))
    if best_acc < acc or epoch % 100 == 0:
        if best_acc < acc:
            best_acc = acc
        # 作图，训练曲线和测试曲线，混淆矩阵
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(losses)
        plt.title("Train Loss")
        plt.subplot(122)
        plt.plot(preds)
        plt.title("Test Accuracy")
        plt.show()

        from sklearn.metrics import confusion_matrix

        confusion = confusion_matrix(test_y, np.array(pred).argmax(axis=1))
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
