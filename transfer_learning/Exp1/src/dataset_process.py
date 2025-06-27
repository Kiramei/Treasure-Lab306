import bz2
import os

import cv2
import face_recognition
import numpy as np
from PIL import Image


def init():
    # 设定输入和输出目录
    input_dir = "./dataset/pre"  # 你的数据存放的根目录
    output_dir = "./dataset/output_png"  # 输出的 PNG 文件存放目录
    final_dir = "./dataset/origin"
    target_dir = "./dataset/target"

    if not os.path.exists(input_dir):
        raise FileNotFoundError("请将数据放在 pre 目录下！")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(final_dir):
        os.makedirs(final_dir)


def decompress_bz2(file_path, output_path):
    """解压 .bz2 文件"""
    with bz2.BZ2File(file_path, 'rb') as bz2_file:
        data = bz2_file.read()
        with open(output_path, 'wb') as output_file:
            output_file.write(data)


def convert_ppm_to_png(ppm_path, png_path):
    """转换 .ppm 文件为 .png"""
    try:
        img = Image.open(ppm_path)
        img.save(png_path, "PNG")
        print(f"转换成功: {ppm_path} → {png_path}")
    except Exception as e:
        print(f"转换失败: {ppm_path}, 错误: {e}")


# 遍历 pre 目录下的所有子文件夹
def process_all_files():
    global input_dir
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".ppm.bz2"):
                # 构造完整路径
                bz2_file_path = os.path.join(root, file)
                ppm_file_path = bz2_file_path[:-4]  # 去掉 .bz2 后缀
                png_file_name = os.path.splitext(file)[0] + ".png"  # 变成 .png
                relative_path = os.path.relpath(root, input_dir)  # 获取相对路径
                png_output_folder = os.path.join(output_dir, relative_path)  # 目标文件夹
                os.makedirs(png_output_folder, exist_ok=True)  # 创建输出文件夹

                # 解压 .bz2 文件
                decompress_bz2(bz2_file_path, ppm_file_path)

                # 转换 .ppm 为 .png
                png_output_path = os.path.join(png_output_folder, png_file_name)
                convert_ppm_to_png(ppm_file_path, png_output_path)

                # 删除解压出的 .ppm 文件，节省空间
                os.remove(ppm_file_path)

    print("所有文件转换完成！")


def remove_all_not_frontal_face():
    global output_dir
    for root, _, files in os.walk(output_dir):
        if len(files) == 0: continue
        selected = min([int(x.split("_")[1]) for x in files])
        for file in files:
            if file.endswith(".png"):
                # 构造完整路径
                png_file_path = os.path.join(root, file)
                if (not (file.__contains__("_fa") or file.__contains__("_fb"))) or int(file.split("_")[1]) != selected:
                    print(f"删除图片: {png_file_path}")
                    os.remove(png_file_path)

    print("删除所有非正脸图片完成！")


def image_preprocess():
    from tensorflow import keras
    model = keras.models.load_model("./dataset/model_v6_23.hdf5")
    emotion_dict = {'生气': 0, '悲伤': 5, '中性': 4, '厌恶': 1, '惊讶': 6, '恐惧': 2, '高兴': 3}
    label_map = dict((v, k) for k, v in emotion_dict.items())
    global output_dir
    for root, _, files in os.walk(output_dir):
        # l_list = []
        sl = False
        fx = ""
        fi = None

        for file in files:
            if file.endswith(".png"):
                png_file_path = os.path.join(root, file)
                image_cropped = face_extract(png_file_path)

                image_input = cv2.resize(image_cropped.copy(), (48, 48))
                image_input = np.reshape(image_input, (1, 48, 48, 1))
                predicted = np.argmax(model.predict(image_input))
                if predicted != 3:
                    print("添加图片: ", png_file_path)
                    cv2.imwrite(os.path.join(final_dir, file.split("_")[0] + ".png"), image_cropped)
                    sl = True
                    break
                fi = image_cropped
                fx = file
                # l_list.append(label)
        if not sl and fx:
            cv2.imwrite(os.path.join(final_dir, fx.split("_")[0] + ".png"), fi)


def face_extract(png_file_path):
    image = cv2.imread(png_file_path)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 0:

        _h = face_locations[0][2] - face_locations[0][0]
        _w = face_locations[0][1] - face_locations[0][3]

        t = max(face_locations[0][0] - int(_h * 0.5), 0)
        l = max(face_locations[0][3] - int(_w * 0.2), 0)
        b = min(face_locations[0][2] + int(_h * 0.1), h)
        r = min(face_locations[0][1] + int(_w * 0.2), w)

        image_cropped = image[t:b, l:r]
    else:
        image_cropped = image
    # 直方图均衡化
    image_cropped = cv2.equalizeHist(image_cropped)
    image_cropped = cv2.resize(image_cropped, (64, 80))
    return image_cropped


def delete_all_not_found():
    global target_dir
    global final_dir

    # Change JPG to PNG
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".jpg"):
                image = cv2.imread(os.path.join(root, file))
                cv2.imwrite(os.path.join(root, file[:-4] + ".png"), image)
                os.remove(os.path.join(root, file))
                print("Change JPG to PNG: ", file)

    # Forward
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".png"):
                target_file_path = os.path.join(root, file)
                if not os.path.exists(os.path.join(final_dir, file)):
                    print(f"删除图片: {target_file_path}")
                    os.remove(target_file_path)

    # Backward
    for root, _, files in os.walk(final_dir):
        for file in files:
            if file.endswith(".png"):
                origin_file_path = os.path.join(root, file)
                if not os.path.exists(os.path.join(target_dir, file)):
                    print(f"删除图片: {origin_file_path}")
                    os.remove(origin_file_path)


def main():
    # process_all_files()
    # remove_all_not_frontal_face()
    # image_preprocess()
    # delete_all_not_found()
    pass


if __name__ == "__main__":
    init()
    main()
