def num_reverse(num: int) -> int:
    """
    反转数字，负数符号不变
    :param num: 要转换的数字
    :return: 转换后的数字
    """
    # 如果是负数，符号不变，先转换成正数，再转换回来
    if num < 0:
        return -num_reverse(-num)
    # 如果是正数，直接转换,将数字从个位开始取出，然后乘以10的位数次方，再相加
    else:
        res = 0
        while num > 0:
            res = res * 10 + num % 10
            num //= 10
        return res


# 测试函数
def test():
    num = 12334456
    print(num_reverse(num))


if __name__ == '__main__':
    test()
