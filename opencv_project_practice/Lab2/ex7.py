import cv2
import numpy as np
from PIL import Image


def read_image(path: str):
    '''
    Read image from path
    :param path: The path of image
    :return: The data of image
    '''
    return cv2.imread(path)


def transform_idnumber_to_binary(idnumber: str):
    '''
    Transform id number to binary
    :param idnumber: The id number
    :return: The binary of id number
    '''
    binary = ''
    for i in idnumber:
        binary += bin(ord(i) - ord('0'))[2:].zfill(4)
    return binary


def change_last_bit_of_the_image(img: np.ndarray):
    '''
    Change the last bit of the image
    :param img: The data of image
    :return: The data of image after changing the last bit
    '''
    img = img.copy()
    height, width = img.shape[:2]
    # fill a mask fully with 254 presented the same size of image
    mask = np.full((height, width), 254, dtype=np.uint8)
    # change the last bit of the image to 0
    img = cv2.bitwise_and(img, mask)
    return img


def encoder(img, idnumber: str) -> np.ndarray:
    '''
    Encoder
    :param img: The data of image
    :param idnumber: The id number
    :return: The data of image after encoding
    '''
    img = img.copy()
    height, width = img.shape[:2]
    binary = transform_idnumber_to_binary(idnumber) + '11111111'
    # create an array that store the binary of id number repeatedly
    binary = np.tile(list(binary), height * width // len(binary) + 1)
    # change the last bit of the image to the binary of id number
    for i in range(height):
        for j in range(width):
            # s is the value of the last bit of the image
            s = img[i, j, 0]
            # t is the value of the last bit of the binary of id number
            t = int(binary[i * width + j])
            # change the last bit of the image to the binary of id number
            img[i, j] = t | (s & 254)
    return img


def decoder(img_encoded) -> str:
    '''
    Decoder
    :param img_encoded: The data of image after encoding
    :return: The id number
    '''
    img_encoded = img_encoded.copy()
    height, width = img_encoded.shape[:2]
    binary = ''
    for i in range(height):
        for j in range(width):
            binary += bin(img_encoded[i, j, 0])[-1]
    idnumber = ''
    for i in range(0, len(binary), 4):
        if binary[i:i + 4] == '1111' and binary[i + 4:i + 8] == '1111':
            break
        idnumber += chr(int(binary[i:i + 4], 2) + ord('0'))
    return idnumber


if __name__ == '__main__':
    img = read_image('data/girl.png')
    cv2.imshow('img', img)
    img_encoded = encoder(img, '2022280594')
    cv2.imshow('img_encoded', img_encoded)
    img_decoded = decoder(img_encoded)
    print(img_decoded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
