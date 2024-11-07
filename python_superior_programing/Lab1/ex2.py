# 基本数字
base = 0.31415926
# 等式左边的数字
left = 1 / (1 - base)
# 等式右边的数字，初始值为0
right = 0
# n的初始值为1
n = 0
# 循环计算，知道两边差值小于5e-3
while abs(left - right) > 5e-3:
    # 等式右边的数字加上base的n次方
    right += base ** n
    # n加1
    n += 1
# 输出结果
print(f'右侧总共需要{n}项')
