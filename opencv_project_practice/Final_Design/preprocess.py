import os
import cv2
import numpy as np

data_dir = 'data_test_tube'
categories = ['OK', 'NG/daqipao', 'NG/jiaodai']
arc = {'OK': 0, 'NG/daqipao': 1, 'NG/jiaodai': 2}
image_size = (256, 256)

data_list = []
i = 0
for category in categories:
    category_dir = os.path.join(data_dir, category)
    for file_name in os.listdir(category_dir):
        # if i == 10:
        data = cv2.imread(os.path.join(category_dir, file_name), cv2.IMREAD_GRAYSCALE)
        data = cv2.resize(data, image_size), arc[category]
        data_list.append(data)


data_list = np.asarray(data_list, dtype=object)
template_tube = data_list[0]
np.random.shuffle(data_list[1:])
data_X = data_list[1:].transpose()[0]
data_label = data_list[1:].transpose()[1]
data_X = np.array([template_tube[0], *data_X])
data_label = np.array([template_tube[1], *data_label])
np.save('data_test_tube_X.npy', data_X)
np.save('data_test_tube_y.npy', data_label)
