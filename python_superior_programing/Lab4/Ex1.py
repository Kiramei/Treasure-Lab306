import numpy as np

# 原始数据
years = np.array([1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984])
distances = np.array([13.75, 15.28, 16.59, 17.32, 18.14, 19.61, 21.03, 21.16, 22.41, 23.57])

# 使用线性插值预测1970年的最佳成绩
predicted_distance = np.interp(1970, years, distances)
k, b = np.polyfit(years, distances, deg=1)
print("预测的1970年奥运会女子铅球最佳成绩：", predicted_distance, "米")
predicted_distance = k * 2000 + b
# 两位，输出权重，偏置
k = round(k, 2)
b = round(b, 2)
print("线性回归的权重：", k, "偏置：", b)
predicted_distance = round(predicted_distance, 2)
print("预测的2000年奥运会女子铅球最佳成绩：", predicted_distance, "米")
