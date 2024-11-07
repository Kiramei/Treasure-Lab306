import matplotlib.pyplot as plt
import numpy as np

# 利用fill绘制一个外接圆半径为1的红色的等边六边形。要求隐藏坐标轴，圆形看起来是标准的圆。
t0 = np.linspace(0, 2 * np.pi, 361)
x0 = np.cos(t0)
y0 = np.sin(t0)
# 创建一个等边六边形的顶点坐标
# 生成六个角度，最后一个角度与第一个角度相同
angle = np.linspace(0, 2*np.pi, 7)[:-1]
x = np.cos(angle)
y = np.sin(angle)
# 绘制等边六边形
fig, ax = plt.subplots()
ax.fill(x, y, 'red')
ax.axis('off')
# 设置坐标轴相等，否则看起来是椭圆
ax.set_aspect('equal')
plt.plot(x0, y0)
# 显示图形
plt.show()
