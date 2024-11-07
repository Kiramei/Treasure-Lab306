# import numpy as np
# a = np.array([4, 2, -6, 7, 5, 4, 3, 4, 9]).reshape(3, 3)
# print(np.linalg.det(a))  # 行列式
# print(np.linalg.inv(a))  # 逆矩阵

# import numpy as np
#
# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)
# b = a
# for i in range(1, 5):
#     a = np.rot90(a, 1, (0, 1))
#     k = np.flipud(np.fliplr(b))  # 上下翻转
#     # 判断是否相等
#     if (a == k).all():
#         print(f'需要旋转{i}次才能使得两者相等')
#         break

# import numpy as np
# a = np.random.rand(5, 5)
# print('行列式：', np.linalg.det(a))  # 行列式
# print('秩：', np.linalg.matrix_rank(a))  # 求矩阵的秩
# print('范数：', np.linalg.norm(a, ord=2))  # ord=2表示求二范数

# import numpy as np

# A = np.array([2, 9, 0, 3, 4, 11, 2, 2, 6]).reshape(3, 3)
# b = np.array([13, 6, 6])
# x_1 = np.linalg.inv(A).dot(b)
# x_2 = np.linalg.solve(A, b)
# print('逆矩阵求解', x_1)
# print('numpy直接解', x_2)
# # 因为python精度问题，所以需要用阈值来判断两个数组是否相等
# print('两者'+('相等' if np.sum([x_1 - x_2]) < 1e-12 else '不相等'))

# 写出代码，给出例子,验证两个方阵A，B满足|AB| =|A||B|
# A = np.array([1123123, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)
# B = np.array([1, 2, 2, 4, 23123, 4, 7, 0, 0]).reshape(3, 3)

# 写成函数，定义|AB| ，|A||B|变量
# def func(_A, _B):
#     C = np.linalg.det(_A.dot(_B))
#     D = np.linalg.det(_A) * np.linalg.det(_B)
#     #     判断CD是否相等
#     return np.isclose(C, D)
#
#
# # 生成验证数据测试函数
# result = True
# for i in range(100):
#     rand_size_1 = np.random.randint(1, 100)
#     scale_1 = np.random.randint(1, 100)
#     scale_2 = np.random.randint(1, 10)
#     A = np.random.rand(rand_size_1, rand_size_1) * scale_1
#     B = np.random.rand(rand_size_1, rand_size_1) * scale_2
#     result &= func(A, B)
# print('验证通过' if result else '验证失败')


# a = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
# a = a[a.astype(np.int16) != a]
# print(a)


# 举例计算 ((A+B)*(-A/2))
# A = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)
# B = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)
# print(np.matmul((A + B), (-A / 2)))


# import random
# vector = [random.randint(0, 20) for _ in range(10)]
# 取反操作
# negated_vector = [-x if 3 <= x <= 8 else x for x in vector]
# 打印结果
# print("原始向量:", vector)
# print("取反后的向量:", negated_vector)


# 给定一个数值范围在[0, 20]之间的一维随机向量，把值在[3, 8]之间的所有数取反，即乘以-1.
# matrix = np.random.randint(low=0, high=100, size=(3, 5))
# print("原始矩阵：")
# print(matrix)
# print("排序后的矩阵：")
# print(matrix[matrix[:, 1].argsort()])
