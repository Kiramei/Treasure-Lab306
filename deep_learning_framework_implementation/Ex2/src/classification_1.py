import numpy as np
import matrixslow as ms


def preprocess_data(data):
    # 编码数据
    encoding = np.zeros((len(data), len(data[0].split(','))))
    unique_values = [[] for _ in range(len(data[0].split(',')))]
    for i, row in enumerate(data):
        values = row.split(',')
        for j, value in enumerate(values):
            if value in unique_values[j]:
                encoding[i, j] = np.where(unique_values[j] == value)[0]
            else:
                unique_values[j] = np.append(unique_values[j], value)
                encoding[i, j] = len(unique_values[j]) - 1

    # 划分训练集和测试集
    train_size = len(data) // 10 * 7
    train_data, train_label = encoding[:train_size, :-1], encoding[:train_size, -1]
    test_data, test_label = encoding[train_size:, :-1], encoding[train_size:, -1]

    # 标签编码为one-hot
    train_label_enc = np.zeros((len(train_label), int(max(np.max(test_label), np.max(train_label)) + 1)))
    for i, label in enumerate(train_label):
        train_label_enc[i, int(label)] = 1

    return train_data, test_data, train_label_enc, test_label


def generate_neural_network(input_node, num_classes=4, hidden_units=6, hidden_layers=1, activation='ReLU'):
    """
    生成神经网络，为了简单，这里只生成多层感知机模型
    :param input_node:     输入节点
    :param num_classes:    输出节点数
    :param hidden_units:   隐藏层神经元个数
    :param hidden_layers:  隐藏层层数
    :param activation:     激活函数
    :return:
    """
    if hidden_layers < 1:
        return ms.layer.fc(input_node, hidden_units, num_classes, None)
    hidden_1 = ms.layer.fc(input_node, 6, hidden_units, activation)
    if hidden_layers > 1:
        for i in range(hidden_layers - 1):
            hidden_1 = ms.layer.fc(hidden_1, hidden_units, hidden_units, activation)
    output = ms.layer.fc(hidden_1, hidden_units, num_classes, None)
    return output


def loop(train_data, test_data, train_label, test_label, x, one_hot, predict, loss, optimizer, batch_size):
    batch_count = 0
    losses = []
    for i in range(len(train_data)):
        feature = np.mat(train_data[i]).T
        label = np.mat(train_label[i]).T
        x.set_value(feature)
        one_hot.set_value(label)
        optimizer.one_step()
        batch_count += 1
        if batch_count >= batch_size:
            # print(f'epoch: {epoch + 1}, loss: {loss.value[0, 0]}')
            losses.append(loss.value[0, 0])
            optimizer.update()
            batch_count = 0
    loss_value = np.mean(losses)
    # 评估模型
    pred = []
    for i in range(len(test_data)):
        feature = np.mat(test_data[i]).T
        x.set_value(feature)
        predict.forward()
        pred.append(predict.value.A.ravel())
    pred = np.array(pred).argmax(axis=1)

    true_positives = np.sum((test_label == 1) & (pred == 1))
    false_positives = np.sum((test_label == 0) & (pred == 1))
    false_negatives = np.sum((test_label == 1) & (pred == 0))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score, loss_value


def draw_pic(accuracies, losses):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(accuracies)), accuracies, label='F1 Score')
    plt.plot(range(len(losses)), losses, label='Loss')
    plt.legend()
    plt.show()
    # for i in range(len(test_data)


def main():
    # 加载数据
    data = np.loadtxt('car.data', dtype=str)
    # 预处理数据
    train_data, test_data, train_label, test_label = preprocess_data(data)

    # 定义模型
    x = ms.core.Variable(dim=(6, 1), init=False, trainable=False)
    one_hot = ms.core.Variable(dim=(4, 1), init=False, trainable=False)
    # 生成神经网络，2层隐藏层，每层32个神经元，输出层4个神经元
    output = generate_neural_network(input_node=x,
                                     num_classes=4,
                                     hidden_units=32,
                                     hidden_layers=2,
                                     activation='ReLU')
    # 利用SoftMax函数将输出转化为概率
    predict = ms.ops.SoftMax(output)
    # 使用交叉熵损失
    loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

    # 定义优化器
    learning_rate = 0.00825
    optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)
    batch_size = 64

    f1_score, losses = [], []
    # 训练模型
    for epoch in range(100):
        f1, loss_value = loop(train_data, test_data, train_label,
                               test_label, x, one_hot, predict, loss,
                               optimizer, batch_size)
        print(f'epoch: {epoch}, acc: {f1}, loss: {loss_value}')
        f1_score.append(f1)
        losses.append(loss_value)

    draw_pic(f1_score, losses)

    # print best accuracy and its loss
    print(f'Best F1 Score: {max(f1_score)}, Loss: {losses[f1_score.index(max(f1_score))]}')
