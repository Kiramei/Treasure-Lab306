import os
import cv2
import json
import logging

import numpy as np
from alive_progress import alive_bar

"""
Divide the dataset into a training set and a test set, with no overlap. 
In the test set, there should be at least 500 images per class. 
The training set should have at least 3000 images per class. 
It is recommended to convert all color images to grayscale 
and resize all images to 256x256. 
The following experiments are based on this training set and test set.
"""


class Config:
    def __init__(self, config_path='./config/config.json'):
        self.config_path = config_path
        self.config = {}
        self.__folder_prepare__()
        self.__logger_prepare__()
        self.__read_config__()

    def __read_config__(self):
        if os.path.exists(self.config_path):
            logging.log(logging.INFO, 'Config.json exists! Now Loading...')
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            logging.log(logging.ERROR, 'Config.json not exists! Now Creating...')
            DEFAULT_CONFIG = {
                "origin_dir": "../dataset",
                "label_name": [
                    "bird",
                    "cat",
                    "dog",
                    "snake"
                ],
                "train_data_path": "./preprocess/train.npy",
                "test_data_path": "./preprocess/test.npy",
                "train_ratio": 0.8
            }
            with open(self.config_path, 'w') as f:
                json.dump(DEFAULT_CONFIG, f)
            self.__read_config__()

    @staticmethod
    def __logger_prepare__():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    @staticmethod
    def __folder_prepare__():
        if not os.path.exists('./preprocess'):
            os.mkdir('./preprocess')


class DataPreparation(Config):
    def __init__(self):
        super().__init__()
        for x in self['label_name']:
            for y in os.listdir(str(os.path.join(self['origin_dir'], x))):
                if y.split('.')[-1] not in ['jpg', 'png', 'jpeg', 'JPEG']:
                    raise Exception('Invalid file type: {}'.format(y))
        self.__load_all__()

    def get_train_data(self):
        """
        This function returns train data.
        :return: [[image, label]...]
        """
        return np.load(self['train_data_path'], allow_pickle=True)

    def get_test_data(self):
        """
        This function returns test data.
        :return: [[image, label]...]
        """
        return np.load(self['test_data_path'], allow_pickle=True)

    def __load_all__(self):
        if os.path.exists(self['test_data_path']) and os.path.exists(self['train_data_path']):
            logging.log(logging.INFO, 'Dataset exists! Now Loading...')
            return
        else:
            logging.log(logging.INFO, 'Train data not exists! Now Creating...')
            mapping_data = []
            for ind, x in enumerate(self['label_name']):
                path_to_origin = os.path.join(self['origin_dir'], x)
                for z in os.listdir(str(path_to_origin)):
                    mapping_data.append([os.path.join(self['origin_dir'], x, z), ind])
            logging.log(logging.INFO, 'Mapping relationship created!')
            self.__generate_dataset__(mapping_data)

    def __generate_dataset__(self, mapping_data):
        logging.log(logging.INFO, 'Dataset not exists! Now Creating...')
        train_data = []
        test_data = []
        for ind, x in enumerate(self['label_name']):
            kind = list(filter(lambda _x: _x[1] == ind, mapping_data))
            kind = np.array(kind)
            np.random.shuffle(kind)
            train_data.append(kind[:int(len(kind) * self['train_ratio'])].tolist())
            test_data.append(kind[int(len(kind) * self['train_ratio']):])
        train_data = np.concatenate(train_data)
        test_data = np.concatenate(test_data)

        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

        parsed_train_data = []
        parsed_test_data = []
        with alive_bar(len(train_data), title='Preparing train data') as bar:
            for x in train_data:
                im = self.__transform__(x[0])
                parsed_train_data.append([im, x[1]])
                bar()
        with alive_bar(len(test_data), title='Preparing test data') as bar:
            for x in test_data:
                im = self.__transform__(x[0])
                parsed_test_data.append([im, x[1]])
                bar()
        logging.log(logging.INFO, 'Saving train and test data...')
        np.save(self['train_data_path'], np.asarray(parsed_train_data, dtype=object), allow_pickle=True)
        np.save(self['test_data_path'], np.asarray(parsed_test_data, dtype=object), allow_pickle=True)
        logging.log(logging.INFO, 'Train and test data created!')

    @staticmethod
    def __transform__(path_to_data: str) -> np.ndarray:
        im_data = cv2.imread(path_to_data, cv2.IMREAD_GRAYSCALE)
        im_data = cv2.resize(im_data, (256, 256))
        return im_data

    def __getitem__(self, item):
        return self.config[item]
