# ans = 0
# for k in range(1, 101):
#     unit = 1/((k+1)**2)
#     ans += unit
# print(ans)

# 方案二、使用列表推导式
ans = sum([1/((k+1)**2) for k in range(1, 101)])
print(ans)