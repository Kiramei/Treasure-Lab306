# import numpy as np
# import matplotlib.pyplot as plt

# # 产生一个3x3的随机矩阵，并把它标准化(减去均值，再除以标准方差)
# X = np.random.random((3, 3))
# print('我们的随机矩阵为：\n', X)
# Xmean = X.mean(axis=0)
# print('我们的均值为：', Xmean)
# Xstd = X.std(axis=0)
# print('我们的标准方差是：', Xstd)
# X1 = (X - Xmean) / Xstd
# print('应用公式得到的标准化矩阵是：\n', X1)


# 已知 A = np.random.randint(1,20,10)，写出语句打印出大于1，小于7的偶数。
# A = np.random.randint(1, 20, 10)
# print('筛选前：', A)
# print('筛选后：', A[(A > 1) & (A < 7) & (A % 2 == 0)])

# 已知a = np.array([[1, 2], [3, 4]])， b = np.array([[5, 6]])，使用concatenate得到array([[1, 2], [3, 4], [5, 6]]) 以及array([[1, 2, 5],[3, 4, 6]])
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6]])
# print('1、使用concatenate得到：\n',
#       np.concatenate((a, b), axis=0))
# print('2、使用concatenate得到：\n',
#       np.concatenate((a, b.T), axis=1))



# # 画出函数 y = sin(x) 在区间 [-5, 5] 的图像
# x = np.linspace(-5, 5, 100)
# y = np.sin(x)
# plt.plot(x, y)
# #  在图形指定位置加标注命令
# plt.text(0, 0, 'y=sin(x)')
# plt.show()


# 一个图形窗口上绘制多个图形可以用       _______函数进行分割窗口。如果需要多个图形窗口同时打开时，可以使用         _______语句
# plt.subplot(2, 1, 1)
# plt.subplot(2, 1, 2)
# plt.show()
# plt.figure(1)
# plt.figure(2)

