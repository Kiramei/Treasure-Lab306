def solution():
    """
    买鸡问题
    :return: 有多少种买法
    """
    solutions = []
    # cocks: 公鸡 hens: 母鸡 chicks: 小鸡 100元买100只鸡 5元一只公鸡 3元一只母鸡 1元三只小鸡
    # 最多买20只公鸡 最多买33只母鸡 最多买300只小鸡
    # 两层循环，穷举买的公鸡和母鸡的数量，小鸡的数量由公鸡和母鸡的数量确定
    # 枚举可能方案，添加到列表中
    for cocks in range(0, 21):
        for hens in range(0, 34):
            chicks = 100 - cocks - hens
            if (cocks * 5 + hens * 3 + chicks / 3 == 100 and
                    chicks % 3 == 0):
                solutions.append([cocks, hens, chicks])
    return solutions


if __name__ == '__main__':
    print(solution())
