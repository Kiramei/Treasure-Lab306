import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取左右两张图像
img_left = cv2.imread('./images/house_left.png', cv2.IMREAD_COLOR)
img_right = cv2.imread('./images/house_right.png', cv2.IMREAD_COLOR)


def alignImages(im1, im2, option_1='norm', option_2='match'):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # feature matching
    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2)
    if option_1 == 'norm':
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    if option_2 == 'match':
        matches = matcher.match(descriptors1, descriptors2)
    else:
        matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    if option_2 == 'knnMatch':
        matches = [m for (m, n) in matches]

    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # 仅保留前15%分数较高的匹配特征
    numGoodMatches = int(len(matches))
    matches = matches[:numGoodMatches]

    # 分別將兩張相片匹配的特徵點匯出
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # 找到 homography。
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # 套用homography
    im1Height, im1Width, channels = im1.shape
    im2Height, im2Width, channels = im2.shape
    im2Aligned = cv2.warpPerspective(im2, h, (im2Width + im1Width, im2Height))

    # 将image1圖像置換到對齊好的圖片中
    stitchedImage = np.copy(im2Aligned)
    stitchedImage[0:im1Height, 0:im1Width] = im1

    return im2Aligned, stitchedImage


res = [None] * 4
_, res[0] = alignImages(img_left, img_right, option_1='norm', option_2='match')
_, res[1] = alignImages(img_left, img_right, option_1='norm', option_2='knnMatch')
_, res[2] = alignImages(img_left, img_right, option_1='hamming', option_2='match')
_, res[3] = alignImages(img_left, img_right, option_1='hamming', option_2='knnMatch')

for r in res:
    plt.imshow(r)
    plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.show()
