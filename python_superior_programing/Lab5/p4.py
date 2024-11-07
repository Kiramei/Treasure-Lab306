# 4. 绘制函数y=x+cosx,\ x\in[0\ 50]描述的曲线图。y轴上只显示当x等于 0，10,20,30,40,50时对应的函数值，将横坐标的范围设置为[-10 60]。参考命令xlim，设置ytick属性。

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 50, 100)
yf = lambda x: x + np.cos(x)
y = x + np.cos(x)
plt.plot(x, y)
plt.xlim(-10, 60)
# y轴上只显示当x等于 0,10,20,30,40,50时对应的函数值
plt.yticks(yf([0, 10, 20, 30, 40, 50]))
plt.show()
