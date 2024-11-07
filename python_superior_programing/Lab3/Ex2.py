import numpy as np


def P3_2(nums):
    # 将非零元素筛选出来
    nums = np.array(nums)
    non_zero = np.nonzero(nums)
    non_zero_nums = nums[non_zero]

    # 创建一个与 nums 相同形状的数组，将非零元素放到前面
    res = np.zeros_like(nums)
    res[:len(non_zero_nums)] = non_zero_nums

    return list(res)


# 测试样例
test_cases = [[0], [0, 1, 0, 2, 0], [2, 1, 0, 0, 3]]

for nums in test_cases:
    result = P3_2(nums)
    print(result)
