import numpy as np
import matplotlib.pyplot as plt

# 生成 x 的数据
x = np.linspace(0, 2 * np.pi, 100)

# 计算 y 的数据
y = np.sin(2 * x) * np.exp(-2 * x) + np.cos(2 * x)

# 创建一个新的图形
plt.figure()
# 设置图形的大小
plt.gcf().set_size_inches(6, 8)

# 绘制火柴杆图
plt.subplot(311)
plt.stem(x, y, linefmt='C0-', markerfmt='C0o', basefmt='k-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stem Plot')
plt.legend(['Stem'], loc='lower right')

# 绘制阶梯图
plt.subplot(312)

plt.stairs(y[:-1], x, baseline=y[-1], color='C1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stairs Plot')
plt.legend(['Stairs'], loc='lower right')

# 绘制散点图
plt.subplot(313)
plt.scatter(x, y, c='C2', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')

# 添加图例
plt.legend(['Scatter'], loc='lower right')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
