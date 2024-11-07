import os
import cv2
import numpy as np


def read_and_preprocess_image(fn):
    img = cv2.imread(fn)
    img = cv2.resize(img, (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


# Use train-dir to make train data, and test data
def generate_data(root_dir = './dataset/train/'):
    list_f = os.listdir(root_dir)
    raw_name_data_list = []
    for ind,i in enumerate(list_f):
        fn = root_dir + i
        processed_data = read_and_preprocess_image(fn)
        raw_name_data_list.append((processed_data ,int(i.split('.')[0] == 'dog')))
        if ind % 1000 == 0:
            print(f'Processed {ind}/{len(list_f)} images')
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    np.save('preprocessed/data_ResNet.npy', np.asarray(raw_name_data_list, dtype=object), allow_pickle=True)

if __name__ == '__main__':
    generate_data('../dataset/train/')
    print('Done')