import numpy as np
import matplotlib.pyplot as plt


def P3_4(data, a0, b0, r, num_iterations):
    X, Y = data[:, 1], data[:, 0]  # 第二列为X
    n, errors = len(X), []
    a, b = a0, b0

    train_epochs = num_iterations
    while train_epochs > 0:
        # 计算误差函数对于a和b的梯度
        delta_a = (-2 / n) * np.sum(X * (Y - (a * X + b)))
        delta_b = (-2 / n) * np.sum(Y - (a * X + b))
        # 更新参数
        a -= r * delta_a
        b -= r * delta_b
        # 计算当前模型的误差
        error = (1 / n) * np.sum((Y - (a * X + b)) ** 2)
        errors.append(error)
        train_epochs -= 1

    # 绘制误差曲线
    plt.figure(figsize=(15, 6))
    plt.grid(True)
    plt.scatter(range(num_iterations), np.log10(errors), marker='o', facecolors='none', edgecolors='#E95176')
    plt.xlabel('Iterations')
    plt.ylabel('Error.log10(e)')
    plt.show()

    # 绘制最终模型拟合曲线
    plt.scatter(X, Y, facecolors='#6176E9', edgecolors='none', label='DataPoint')
    plt.plot(X, a * X + b, color='#E95176', label='Curve')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    return [a, b]


# 从data.txt加载数据
data = np.loadtxt('data.txt')

# 设置参数初始值和学习步长
a0 = np.mean(data[:, 0]) / np.mean(data[:, 1])
b0 = data[0, 0] - a0 * data[0, 1]
r = 0.04

# 训练模型并绘制图像
parameters = P3_4(data, a0, b0, r, num_iterations=10000)
print('Parameters [a, b]:', parameters)
