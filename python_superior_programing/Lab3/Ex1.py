import numpy as np


def P3_1(N):
    """
    判断一个数是否为丑数，如果是丑数返回True，否则返回False
    判断方法：不断除以2、3、5，如果最后得到的数为1，则为丑数，否则不是丑数
    """
    while N % 2 == 0: N = N // 2
    while N % 3 == 0: N = N // 3
    while N % 5 == 0: N = N // 5
    return N == 1


# np.arange(1, 101) 表示生成一个1-100的数组
nums = np.arange(1, 101)
# [P3_1(x) for x in nums] 表示对nums中的每个元素x，都执行P3_1(x)函数
# nums[[P3_1(x) for x in nums]] 表示nums中所有满足P3_1(x)的元素
ugly = nums[[P3_1(x) for x in nums]]
print("1-100之间的丑数:")
print(ugly)
