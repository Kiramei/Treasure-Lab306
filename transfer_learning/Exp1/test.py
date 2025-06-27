# from src.dataset_process import face_extract
# import os
# import cv2
#
# for root, _, files in os.walk("dataset/CUFSF/test//origin"):
#     output_name = "00001"
#     for file in files:
#         if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".webp"):
#             path = os.path.join(root, file)
#             p = face_extract(path)
#             cv2.imwrite(output_name + ".png", p)
#             output_name = str(int(output_name) + 1).zfill(5)
#             os.remove(path)
import os
from pathlib import Path

import cv2
import numpy as np

#
# a = Path("dataset/FS2K/test/origin")
# b = Path("dataset/FS2K/test/target")
#
# for f1 in a.glob("*"):
#     _a = cv2.imread(f1)
#     _b = cv2.imread(str(b / f1.name))
#
#     _a= cv2.resize(_a, (224, 224))
#     _b = cv2.resize(_b, (224, 224))
#
#     cv2.imwrite(f1, _a)
#     cv2.imwrite(str(b / f1.name), _b)


# for i, x in enumerate(os.listdir(a)):
#     name = str(i + 1 + 1529).zfill(5)
#     ix = x[-8:-4]
#     os.rename(a / x, a / (name + ".jpg"))
#     os.rename(b / ("sketch" + ix + ".jpg"), b / (name + ".jpg"))

# at = []
# for file in a.iterdir():
#     at.append(file)
#
# bt = []
# for file in b.iterdir():
#     bt.append(file)
#
# att = [str(x).split('.')[0] for x in at]
# btt = [str(x).split('.')[0] for x in bt]
# att = [x[x.rindex('/') + 1:] for x in att]
# btt = [x[:x.rindex('-')][x.rindex('/') + 1:] for x in btt]
#
# att = sorted(att)
# btt = sorted(btt)
#
# # if att == btt:
# #     print("True")
#
# for ind, f in enumerate(att):
#     name = str(ind + 1).zfill(5)
#     p = f.split('-')[0]
#     p2 = "-".join(f.split('-')[1:])
#     f1 = a / (f + ".jpg")
#     f2 = b / (p + "2-" + p2 + "-sz1.jpg")
#     os.rename(f1, a / (name + ".jpg"))
#     os.rename(f2, b / (name + ".jpg"))

# import cv2
#
# def orb_align(a, b):
#     # 使用 ORB 进行特征检测和描述
#     orb = cv2.ORB_create()
#     keypoints_a, descriptors_a = orb.detectAndCompute(a, None)
#     keypoints_b, descriptors_b = orb.detectAndCompute(b, None)
#
#     # 使用 BFMatcher 进行特征匹配
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(descriptors_a, descriptors_b)
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     # 选择前 50 个最佳匹配
#     good_matches = matches[:50]
#
#     # 提取匹配点
#     points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#
#     # 计算单应性矩阵
#     H, mask = cv2.findHomography(points_b, points_a, cv2.RANSAC, 5.0)
#
#     # 进行透视变换
#     h, w = a.shape
#     b_aligned = cv2.warpPerspective(b, H, (w, h))
#
#     return b_aligned
#
# for i in range(len(os.listdir(a))):
#     name = str(i+1).zfill(5)
#
#     a_img = cv2.imread(str(a / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
#     b_img = cv2.imread(str(b / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
#
#     b_aligned = orb_align(a_img, b_img)
#
#     cv2.imwrite(str(b / f"{name}.jpg"), b_aligned)
#     print(f"Processed {name}/{len(os.listdir(a))}")

# import glob
# # 删除b 下所有除开以0开头的文件的所有文件
#
# for file in glob.glob(str(b / "*.jpg")):
#     if not file.startswith("0"):
#         os.remove(file)
#         print(f"Deleted {file}")




from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from pathlib import Path
def psnr_ssim(path):
    # 读取拼接图像
    grid_image_path = Path(path)
    if not grid_image_path.exists():
        raise FileNotFoundError(f"未找到图片 {grid_image_path}")

    grid_img = cv2.imread(str(grid_image_path))

    # 获取图像尺寸
    h, w, _ = grid_img.shape

    # 去除网格Grid，网格为4行5列，每个图片padding为2
    real_A = grid_img[2:h // 4 - 2, 2:w // 5 - 2]
    fake_A = grid_img[h // 4 + 2:h // 2 - 2, 2:w // 5 - 2]
    real_B = grid_img[h // 2 + 2:3 * h // 4 - 2, 2:w // 5 - 2]
    fake_B = grid_img[3 * h // 4 + 2:h - 2, 2:w // 5 - 2]

    # 转换为灰度图用于 SSIM 计算
    real_B_gray = cv2.cvtColor(real_B, cv2.COLOR_BGR2GRAY)
    fake_B_gray = cv2.cvtColor(fake_B, cv2.COLOR_BGR2GRAY)
    real_A_gray = cv2.cvtColor(real_A, cv2.COLOR_BGR2GRAY)
    fake_A_gray = cv2.cvtColor(fake_A, cv2.COLOR_BGR2GRAY)

    # 计算 PSNR
    psnr_B = psnr(real_B, fake_B, data_range=255)
    psnr_A = psnr(real_A, fake_A, data_range=255)

    # 计算 SSIM
    ssim_B = ssim(real_B_gray, fake_B_gray, data_range=255)
    ssim_A = ssim(real_A_gray, fake_A_gray, data_range=255)

    # 记录结果
    results_df = pd.DataFrame({
        "Comparison": ["fake_B vs real_B", "fake_A vs real_A"],
        "PSNR": [psnr_B, psnr_A],
        "SSIM": [ssim_B, ssim_A]
    })

    # 打印结果
    print(results_df)

    # 可选：保存结果到 CSV 文件
    results_df.to_csv("psnr_ssim_results.csv", index=False)

if __name__ == '__main__':
    psnr_ssim(r"F:\WorkSpace\py\Experiment\tl\Exp1\logs\T-1742044576\preds\000500.png")