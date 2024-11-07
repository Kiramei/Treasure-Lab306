import os
import shutil

import cv2


class Solution:
    DATASET_DIR_OK = 'data_test_tube/OK'
    DATASET_DIR_NG_1 = 'data_test_tube\\NG\\daqipao'
    DATASET_DIR_NG_2 = 'data_test_tube\\NG\\jiaodai'

    def __init__(self):
        super().__init__()
        self.pkg_ok = []
        self.pkg_ng = []

    @staticmethod
    def read_image(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img

    def read_all_images(self):
        for path in os.listdir(self.DATASET_DIR_OK):
            self.pkg_ok.append(self.read_image(os.path.join(os.getcwd(),self.DATASET_DIR_OK, path)))
        for path in os.listdir(self.DATASET_DIR_NG_1):
            self.pkg_ng.append(self.read_image(os.path.join(os.getcwd(),self.DATASET_DIR_NG_1, path)))
        for path in os.listdir(self.DATASET_DIR_NG_2):
            self.pkg_ng.append(self.read_image(os.path.join(os.getcwd(),self.DATASET_DIR_NG_2, path)))


if __name__ == '__main__':
    s = Solution()
    s.read_all_images()
    print(len(s.pkg_ok))
    print(len(s.pkg_ng))