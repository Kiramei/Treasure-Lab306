import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 从CSV文件读取数据并创建DataFrame

# 统计分析
# statistics = df.describe()
# print(statistics)
#
# # 查找缺失值
# missing_values = df.isnull().sum()
# print(missing_values)
#
# # 填充缺失值
# df['workclass'].fillna('Unknown', inplace=True)
# df['occupation'].fillna('Unknown', inplace=True)
# df['native-country'].fillna('Unknown', inplace=True)
#
# # 独热编码
# df_encoded = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
#
# # 确保各个属性使用计算机可识别的数值形式
# df_encoded['class'] = df_encoded['class'].map({'<=50K': 0, '>50K': 1})
#
# # 关联性分析
# correlation = df_encoded.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation, annot=False, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()
#
# # 发掘有意义的性质或结论
# # 示例：分析年龄与收入的关系
# age_income = df_encoded.groupby('age')['class'].mean()
# plt.plot(age_income.index, age_income.values)
# plt.xlabel('Age')
# plt.ylabel('Income Proportion')
# plt.title('Proportion of High Income by Age')
# plt.show()

# 从CSV文件读取数据并创建DataFrame

columns = ['Age','Workclass','fnlgwt','Education','EduNo','MaritalStatus',
           'Job','Relationship','Race','Gender','CapitalGain',
           'CapitalLoss','HoursPerWeek','Country','Income']
df = pd.read_csv('data.csv', names=columns)
df.head(10)
df.info()
print(df.sort_values(by='fnlgwt', ascending=True)[:10])

# 统计分析
for column in df.columns:
    counts = df[column].value_counts()
    print(f"Column: {column}")
    if column == 'fnlwgt':
        continue

    threshold = 0.02 * df[column].count()  # 假设阈值为1%
    if len(counts[counts < threshold]) > 0:
        counts_other = counts[counts < threshold]
        counts_other_sum = counts_other.sum()
        counts_other_combined = pd.Series(counts_other_sum, index=["Other"])
        counts_filtered = pd.concat([counts[counts >= threshold], counts_other_combined])

        plt.pie(counts_filtered, labels=counts_filtered.index, autopct='%1.1f%%')
    else:
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title(f"Counts of {column}")
    plt.show()
    print(counts)
    print()

