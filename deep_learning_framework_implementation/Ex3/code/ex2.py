import imageio
import matplotlib.pyplot as plt
import numpy as np

import matrixslow as ms

# 定义超参数
INPUT_SIZE = 1  # 初始参数尺寸（输入维度）
TIME_STEP = 10  # 时间步长(序列长度)
LR = 0.02  # 学习效率

STATUS_DIM = 6  # 状态维度（隐藏层参数）
# 输入向量节点
inputs = [ms.core.Variable(dim=(INPUT_SIZE, 1), init=False, trainable=False) for i in range(TIME_STEP)]
# 输入权值矩阵，需要初始化，可训练
U = ms.core.Variable(dim=(STATUS_DIM, INPUT_SIZE), init=True, trainable=True)
# 状态权值矩阵，需要初始化，可训练
W = ms.core.Variable(dim=(STATUS_DIM, STATUS_DIM), init=True, trainable=True)
# 偏置向量，需要初始化，可训练
b = ms.core.Variable(dim=(STATUS_DIM, 1), init=True, trainable=True)

last_step = None  # 上一步的输出，第一步没有上一步，先将其置为 None
for iv in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, iv), b)
    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)
    h = ms.ops.ReLU(h)
    last_step = h

fc1 = ms.layer.fc(last_step, STATUS_DIM, 40, "ReLU")  # 第一全连接层
fc2 = ms.layer.fc(fc1, 40, 10, "ReLU")  # 第二全连接层
output = ms.layer.fc(fc2, 10, 10, "None")  # 输出层
predicted = ms.ops.Logistic(output)  # 将输出转换为概率，方便交叉熵损失计算
class_gt = ms.core.Variable((TIME_STEP, 1), trainable=False)  # 输出标签，为Ground Truth
loss = ms.ops.loss.MeanSquaredErrorLoss(output, class_gt)  # 交叉熵损失作为模型损失
optimizer = ms.optimizer.Adam(ms.default_graph, loss, LR)  # Adam优化器，利于收敛
# 初始化hidden state
h_state = None  # 设置h_state初始值为None，因为初始时没有任何学习记忆包
# 可视化
fig = plt.figure(1, figsize=(12, 8))
plt.ion()

# 开始机器学习
figList = []
losses = []  # 记忆损失值
for step in range(1000):
    print(step)
    start, end = step * np.pi, (step + 1) * np.pi
    sampled_steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    input_seq, gt = np.sin(sampled_steps), np.cos(sampled_steps)
    input_seq = input_seq.reshape(TIME_STEP, INPUT_SIZE)
    gt = gt.reshape(TIME_STEP, INPUT_SIZE)
    # 设置输入层的值
    for j, x in enumerate(inputs):
        x.set_value(np.mat(input_seq[j]))
    # 设置Ground Truth
    class_gt.set_value(np.mat(gt))
    # 执行一次梯度下降步骤更新模型参数
    optimizer.one_step()
    accuracy = output.value - gt  # 误差
    value = output.value
    losses.append(loss.value[0, 0])
    # 优化器更新，更新学习率等优化器状态
    optimizer.update()

    # 出图
    plt.subplot(3, 1, 1)
    plt.plot(sampled_steps, gt.flatten())  # 被学习数据
    plt.ylabel('Target')

    plt.subplot(3, 1, 2)
    plt.plot(sampled_steps, input_seq.flatten())  # 被学习数据
    plt.ylabel('Input')

    plt.subplot(3, 1, 3)
    plt.ylabel('Prediction')
    plt.plot(sampled_steps, value, label='prediction')  # 学习后的数据
    plt.plot(sampled_steps, accuracy, 'k--', linewidth=3)  # 误差
    plt.draw()
    plt.pause(0.05)

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    figList.append(image)

plt.ioff()
plt.close()

imageio.mimsave("test.gif", figList, fps=10)

plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()
