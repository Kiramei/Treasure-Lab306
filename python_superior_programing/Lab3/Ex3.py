def P3_3(nums, x):
    low = 0
    high = len(nums) - 1

    while low <=high:
        mid = (low + high) // 2

        if nums[mid] == x:
            return mid
        elif nums[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    return -1

# 测试样例
test_cases = [
    ([1, 2, 3, 4, 5, 6], 4),  # 数组长度为偶数，找到元素
    ([1, 2, 3, 4, 5, 6], 7),  # 数组长度为偶数，找不到元素
    ([1, 2, 3, 4, 5], 3),     # 数组长度为奇数，找到元素
    ([1, 2, 3, 4, 5], 6)      # 数组长度为奇数，找不到元素
]

for numList, x in test_cases:
    result = P3_3(numList, x)
    print(result)