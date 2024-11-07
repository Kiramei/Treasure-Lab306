import numpy as np

# print(np.arange(0, 41, 4) * 0.1 * np.pi)
# print(np.linspace(4*np.pi, 0, 12, endpoint=True))

# print(np.array([1, 2, 3, 4, 5])[::-1])
# import numpy as np
#
# A = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])
#
# reversed_A = A[:,::-1]
#
# print(reversed_A)

# import numpy as np
#
# arr = np.array([1, 2, 0, 0, 4, 0])
# nonzero_indices = np.where(arr != 0)[0]
#
# print(nonzero_indices)

# import numpy as np
#
# arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# arr[arr % 2 != 0] = -1
#
# print(arr)

# a = np.array([1, 2, 3, 4, 5, 6])
# b = np.array([4, 5, 6, 7, 8, 9])
# print(np.isin(a, b))


import numpy as np

A = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])

# print(A)
#
# # 得到 4
# element_4 = A[1, 0]
# print(element_4)
#
# # 得到 [4 6]
# subarray_46 = A[1, [0, 2]]
# print(subarray_46)
#
# # 得到 [7 4 1]
# subarray_741 = A[:, 0]
# print(subarray_741)
#
# # 得到 [4 5; 7 8]
# submatrix_45_78 = A[:2, :2][::-1]
# print(submatrix_45_78)

# 得到 [1 2 3; 4 5 6]
# b = np.where(A == 5)
# a = list(A[1][A[1] != 5])
# A = A.tolist()
# A[b[0][0]] = a
#
# print(A)
# # A[b[0]] = np.delete(A[b[0]])

diag = [1, 2, 3, 4]
diag = np.diag(diag, k=-1)
print(diag)
