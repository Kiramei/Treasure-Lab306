"""
Created on Tue May 26 00:20:24 2020

@author: ScramJet
"""
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional

# 定义超参数
INPUT_SIZE = 1  # 初始参数尺寸
TIME_STEP = 10  # 时间步长
LR = 0.02
# 学习效率
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 搭建RNN神经网络
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(  # 此处直接使用RNN不使用LSTM就可以达到很好的效果，注意RNN参数定义和LSTM一致
            input_size=INPUT_SIZE,
            hidden_size=32,  # 注意TIME_STEP并不是RNN中的参数，TIME_STEP是后续数据处理时使用的
            num_layers=1,  # 注意是num_layers
            batch_first=True,  # 此时x为（batch_size,time_step,input_size)三个维度的参数，batch_size优先，位于第一位
        )
        self.output = torch.nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x,
                                  h_state)  # RNN返回的值包含x本身的值（batch_size,time_step,
        # input_size），和每次叠加的学习内容h_state，也就是hidden state，学习记忆包。。。
        outs = []  # 把每个时间步长计算得到的r_out都收集起来
        for time_step in range(r_out.size(1)):  # r_out.size(1)是返回的x中时间步长的个数
            outs.append(self.output(r_out[:, time_step, :]))  # 将r_out中每个时间步长对应的值加入列表outs中
        return torch.stack(outs, dim=1), h_state  # 将列表打包成张量返回，并且返回h_state


rnn = RNN()

# 选取优化器
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()  # 定义误差函数
# 初始化hidden state
h_state = None  # 设置h_state初始值为None，因为初始时没有任何学习记忆包

# 可视化
fig = plt.figure(1, figsize=(12, 8))
plt.ion()
# 开始机器学习
figList = []
for step in range(100):
    print(step)
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # 将x_np从一维数据变为（batch_size,time_step,input_size）三维形式
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])  # 将y_np从一维数据变为（batch_size,time_step,input_size）三维形式

    prediction, h_state = rnn(x, h_state)  # 初始h_state设置为None
    h_state = h_state.data
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = prediction - y  # 误差

    # 出图
    plt.subplot(3, 1, 1)
    plt.plot(steps, y_np.flatten())  # 被学习数据
    plt.ylabel('Target')

    plt.subplot(3, 1, 2)
    plt.plot(steps, x_np.flatten())  # 被学习数据
    plt.ylabel('Input')

    plt.subplot(3, 1, 3)
    plt.ylabel('Prediction')
    plt.plot(steps, prediction.data.numpy().flatten(), label='prediction')  # 学习后的数据
    plt.plot(steps, accuracy.data.numpy().flatten(), 'k--', linewidth=3)  # 误差
    plt.draw()
    plt.pause(0.05)

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    figList.append(image)

plt.ioff()
plt.close()

imageio.mimsave("test.gif", figList, fps=10)
