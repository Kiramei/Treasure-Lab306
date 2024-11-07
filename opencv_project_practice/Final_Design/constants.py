import random
import cv2
import matplotlib.pyplot as plt

TEST_IMG_1 = './data_test_tube/OK/OK_0001.bmp'
TEST_IMG_2 = './data_test_tube/OK/OK_0002.bmp'
TEST_IMG_3 = './data_test_tube/OK/OK_0003.bmp'
TEST_IMG_4 = './data_test_tube/NG/daqipao/da_qipao_0001.bmp'
TEST_IMG_5 = './data_test_tube/NG/daqipao/da_qipao_0013.bmp'
TEST_IMG_6 = './data_test_tube/NG/daqipao/da_qipao_0014.bmp'
TEST_IMG_7 = './data_test_tube/NG/daqipao/da_qipao_0022.bmp'
TEST_IMG_8 = './data_test_tube/NG/daqipao/da_qipao_0012.bmp'
TEST_IMG_9 = './data_test_tube/NG/jiaodai/jiaodai_0004.bmp'
TEST_IMG_10 = './data_test_tube/NG/jiaodai/jiaodai_0005.bmp'
TEST_IMG_11 = './data_test_tube/NG/jiaodai/jiaodai_0013.bmp'
TEST_IMG_12 = './data_test_tube/NG/jiaodai/jiaodai_0015.bmp'
TEST_IMG_13 = './data_test_tube/NG/jiaodai/jiaodai_0010.bmp'

ALL_IMG_ARRAY = [TEST_IMG_1, TEST_IMG_2, TEST_IMG_3, TEST_IMG_4, TEST_IMG_5, TEST_IMG_6, TEST_IMG_7, TEST_IMG_8,
                 TEST_IMG_9, TEST_IMG_10, TEST_IMG_11, TEST_IMG_12, TEST_IMG_13]

DIFF_ARRAY = random.sample(ALL_IMG_ARRAY, 5)


def get_a_random_img():
    return cv2.imread(random.choice(ALL_IMG_ARRAY), cv2.IMREAD_GRAYSCALE)
