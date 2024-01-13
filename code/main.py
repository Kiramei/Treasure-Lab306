from core import *

# def main():
#     print('main')
#     get_train_data()


if __name__ == '__main__':
    dp = DataPreparation()
    d = dp.get_test_data()

    # cv2.imshow('s', s)
    # cv2.waitKey(0)
    print(len(d))
