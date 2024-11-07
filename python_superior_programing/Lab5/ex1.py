import matplotlib.pyplot as plt
import numpy as np

# Lab5.csv 是一个公司销售数据。导入数据，做如下的分析。

# 3.1. 画出每个月份月份的盈利情况。曲线具有如下的特征：
# 	Line Style dotted and Line-color should be red
# 	Show legend at the lower right location.
# 	X label name = Month Number
# 	Y label name = Sold units number
# 	Add a circle marker.
# 	Line width should be 3

labels = np.loadtxt('Lab5.csv', delimiter=',', dtype='str', max_rows=1)
data = np.loadtxt('Lab5.csv', delimiter=',', skiprows=1)
month = data[:, 0]
sales = data[:, 1]
# plt.plot(month,
#          sales,
#          'ro--',
#          linewidth=3,
#          label='Sales')
# plt.xlabel('Month Number')
# plt.ylabel('Sold units number')
# plt.legend(loc='lower right')
# plt.title('Company Sales data of last year')
# plt.show()


# 3.2 用散点图展示toothpaste 产品，每个月的销量。 如下图所示。
# toothpaste = data[:, 2]
# plt.scatter(month,
#             toothpaste,
#             label='Tooth paste Sales Data')
# plt.xlabel('Month Number')
# plt.ylabel('Number of units Sold')
# plt.legend(loc='upper left')
# plt.title('Tooth paste Sales data')
# plt.xticks(month)
# plt.grid(True, linewidth=1, linestyle='--')
#
# plt.show()


# 3.3 对比facecream 和facewash 产品的销量，用柱状图展示他们每个月的销量差异。如下图所示。
# facecream_sales = data[:, 1]
# facewash_sales = data[:, 2]
# months = data[:, 0]
# # 绘制柱状图
# plt.gcf().set_size_inches(7, 4)
# plt.bar(months+0.12, facecream_sales,
#         width=0.24,
#         label='Face Cream Sales Data',)
# plt.bar(months-0.12, facewash_sales,
#         width=0.24,
#         label='Face Wash Sales Data')
# # 网格
# plt.grid(True, linewidth=1, linestyle='--')
# plt.xticks(months)
# plt.legend(loc='upper left')
# plt.xlabel('Month Number')
# plt.ylabel('Sales units in number')
# plt.title('Face Wash and Face Cream Sales Data')
# plt.show()

# 3.4 用stack plot 展示所有产品的销量，如下图所示。
# Better Color Style
# plt.figure()
# ax = plt.gca()
# dts = data[:, 1:7].transpose()
# ax.stackplot(data[:, 0], *[x.transpose() for x in dts], labels=labels[1:7])
# ax.legend(loc='upper left')
# plt.xticks(data[:, 0])
# plt.xlabel('Month Number')
# plt.ylabel('Sales units in number')
# plt.title("All product sales data using stack plot")
# plt.show()

# 3.5 用pie chart 展示每个产品对公司盈利的贡献。
plt.figure()
ax = plt.gca()
total = np.sum(data, axis=0)
ax.pie(total[1:7],
       labels=labels[1:7],
       autopct='%1.1f%%')
plt.title('Contributions of each product to company profit')
plt.show()
