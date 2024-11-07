import sys
import numpy as np

sys.path.append('../MatrixSlowLib')
import matrixslow as ms

batchsize = 16
status_dimension = 32
seqlength = 25

# data I/O
data = open('./alice29.txt', 'r').read()
chars = list(set(data))
data_size, M = len(data), len(chars)
print(f"data has {data_size} characters, {M} unique.")
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

inputs = [ms.core.Variable(dim=(M, 1), init=False, trainable=False) for _ in range(seqlength)]
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
output = ms.layer.fc(fc1, 40, M, "ReLU")
predict = ms.ops.Logistic(output)
label = ms.core.Variable((M, 1), trainable=False)

# 训练
loss = ms.ops.CrossEntropyWithSoftMax(output, label)
learning_rate = 0.01
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)
batch_size = 16

# 主循环遍历训练周期（epoch）
for epoch in range(50):
    # 对于每个epoch，进行数据集中的若干次迭代（itr_count）
    for itr_count in range(500):
        # 随机选择一个起始位置，确保不会超出数据边界和考虑序列长度及批次大小
        pos = np.random.randint(len(data) - 1 - seqlength - batch_size)
        # 遍历批次中的每个样本
        for batch_k in range(batch_size):
            # 计算该样本在数据集中的实际起始位置
            start = pos + batch_k
            # 提取一个序列片段用于训练
            seq = data[start: start + seqlength]
            # 针对该序列的每个字符，设置输入层的值（one-hot编码）
            for j, x in enumerate(inputs):
                d = np.zeros((M, 1))  # 初始化一个零向量
                d[char_to_ix[seq[j]]] = 1  # 根据字符映射表设置相应位置为1，实现one-hot
                x.set_value(np.mat(d))  # 将该向量设置为当前输入层的值
            # 设置目标字符的one-hot编码作为标签
            d = np.zeros((M, 1))
            target_ch = data[start + seqlength]  # 获取目标字符
            d[char_to_ix[target_ch]] = 1  # 目标字符one-hot编码
            label.set_value(np.mat(d))  # 设置标签值
            # 执行一次梯度下降步骤更新模型参数
            optimizer.one_step()
        # 每100次迭代打印损失信息
        if itr_count % 100 == 0 and itr_count > 0:
            print(f"epoch {epoch:3d} itr_count : {itr_count:4d} loss: {loss.value[0, 0]:.3f}")
    # 在每个epoch末尾，更新学习率等优化器状态
    optimizer.update()
    # 验证阶段：随机选取一个子序列进行预测准确率评估
    pred = []  # 用于存储预测字符的索引
    trues = []  # 用于存储真实字符的索引
    # 重新选择一个起始位置用于验证
    pos = np.random.randint(len(data) - 1 - seqlength - batch_size)
    # 进行1000次字符预测
    for k in range(1000):
        start = pos + k
        seq = data[start: start + seqlength]
        # 为验证序列中的每个字符设置输入
        for j, x in enumerate(inputs):
            d = np.zeros((M, 1))
            d[char_to_ix[seq[j]]] = 1
            x.set_value(np.mat(d))
        # 前向传播得到预测结果
        predict.forward()
        pred.append(predict.value.A.ravel())  # 添加预测结果到列表
        # 记录真实字符的索引
        target_c = data[start + seqlength]
        trues.append(char_to_ix[target_c])
    # 将预测结果转化为索引并计算准确率
    pred = np.array(pred).argmax(axis=1)
    acc = sum(1 for i in range(len(trues)) if trues[i] == pred[i]) / len(trues)
    # 打印当前epoch的预测准确率
    print(f"epoch {epoch:.3f} acc: {acc:.3f}")
