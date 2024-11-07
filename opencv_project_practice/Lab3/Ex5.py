import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image():
    return cv2.imread("./data/tr.png", cv2.IMREAD_GRAYSCALE)


# 去除噪声
def remove_noise(image):
    image = cv2.medianBlur(image, 5)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.medianBlur(image, 5)


def border_detect(image):
    return cv2.Canny(image, 50, 100)


def linearize(img):
    height, width = img.shape
    # 按高度处理像素
    for i in range(height):
        img[i] = process_row(img[i], 9)
    return img


def process_row(row, threshold):
    btm_pt, cnt_1, cnt_0 = 0, 0, 0
    for j in range(len(row)):
        # 如果不是白色像素，跳过
        if row[j] != 255:
            continue
        row[j] = 0
        cnt_1 += 1
        # 如果是白色像素，向右扫描
        for k in range(j + 1, len(row)):
            if row[k] == 255:
                row[k], cnt_0 = 0, 0
                cnt_1 += 1
                btm_pt = k
            else:
                cnt_0 += 1
            if cnt_0 >= 10:
                break
        # 如果白色像素数超过阈值，将这些像素置为白色
        if cnt_1 >= threshold:
            row[j:btm_pt + 1] = 255
            cnt_1, cnt_0, j = 0, 0, btm_pt
    return row


def check_line_parallelism(img, threshold=5):
    # 按高度处理像素
    sum_ = np.sum(img, axis=1)
    idx = np.argsort(sum_)[::-1]
    idx_low, idx_high = idx[0], idx[1]
    flag_low, flag_high = True, False
    # 处理处于上下边界的直线
    scale = (idx_high + threshold < idx) & (idx < idx_low - threshold)
    lines = img[scale]
    cols = np.any(lines == 255, axis=0)
    rows = np.any(lines == 255, axis=1)
    # 检查上下边界是否平行，使用np.argmax()函数找到第一个为True的位置
    if np.any(cols):
        if np.any(rows[:np.argmax(cols)]):
            flag_high = False
        if np.any(rows[np.argmax(cols):]):
            flag_low = False
    sum_ = np.sum(img, axis=0)
    idx = np.where(sum_ == 510)[0]
    # 检查上下边界的宽度是否相同
    find_idx = [np.sum(x) for x in img[:, idx]]
    find_idx = np.nonzero(find_idx)[0]
    width = np.mean(np.diff(find_idx))
    return flag_low, flag_high, width


if __name__ == '__main__':
    img = read_image()
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    img = remove_noise(img)
    img = border_detect(img)
    ps = linearize(img)
    plt.subplot(122), plt.imshow(ps, cmap='gray'), plt.title('Denoised')
    ps = check_line_parallelism(ps)
    print(f'上边界是否平行：{"是" if ps[0] else "否"}\n'
          f'下边界是否平行：{"是" if ps[1] else "否"}\n'
          f'宽度：{ps[2]}。')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
