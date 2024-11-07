# Desc: 计算数组的前缀和
def sum_of_array(array: list) -> list:
    """
    计算数组的前缀和，使用列表推导式，对于每一个索引i，计算array[:i]的和，
    并将结果存入列表中，最后返回列表
    :param array: 数组
    :return: 数组的前缀和
    """
    return [sum(array[:i]) for i in range(1, len(array) + 1)]


# 测试函数
def test():
    array = range(0, 10)
    print(sum_of_array(array))


if __name__ == '__main__':
    test()
