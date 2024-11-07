import numpy as np
import matplotlib.pyplot as plt

# 1.用条形图表示每个人的各科成绩，并用饼形图表示每个人各科成绩占总成绩的百分比。参考命令bar, pie。
# 	Science	Math	English
# Tom	80	92	75
# Lily	96	88	90
# Neki	90	96	85

data = np.array([[80, 92, 75], [96, 88, 90], [90, 96, 85]])
label_x = np.array(['Tom', 'Lily', 'Neki'])
label_y = np.array(['Science', 'Math', 'English'])

# better color style
plt.style.use('ggplot')

# for Tom
plt.subplot(1, 2, 1)
plt.bar(label_y, data[0])
plt.title(label_x[0])
plt.xlabel('Subject')
plt.ylabel('Score')
plt.text(0, 80, str(data[0][0]))
plt.text(1, 92, str(data[0][1]))
plt.text(2, 75, str(data[0][2]))
plt.subplot(1, 2, 2)
plt.pie(data[0], labels=label_y, autopct='%1.1f%%')
plt.title(label_x[0])
plt.tight_layout()
plt.show()

# for Lily
plt.subplot(1, 2, 1)
plt.bar(label_y, data[1])
plt.title(label_x[1])
plt.xlabel('Subject')
plt.ylabel('Score')
plt.text(0, 96, str(data[1][0]))
plt.text(1, 88, str(data[1][1]))
plt.text(2, 90, str(data[1][2]))
plt.subplot(1, 2, 2)
plt.pie(data[1], labels=label_y, autopct='%1.1f%%')
plt.title(label_x[1])
plt.tight_layout()
plt.show()

# for Neki
plt.subplot(1, 2, 1)
plt.bar(label_y, data[2])
plt.title(label_x[2])
plt.xlabel('Subject')
plt.ylabel('Score')
plt.text(0, 90, str(data[2][0]))
plt.text(1, 96, str(data[2][1]))
plt.text(2, 85, str(data[2][2]))
plt.subplot(1, 2, 2)
plt.pie(data[2], labels=label_y, autopct='%1.1f%%')
plt.title(label_x[2])
plt.tight_layout()
plt.show()

