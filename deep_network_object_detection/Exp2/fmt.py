import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_boxes = []
    gt_labels = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        gt_boxes.append([xmin, ymin, xmax, ymax])
        gt_labels.append(label)
    return gt_boxes, gt_labels

ann_dir = r'F:\WorkSpace\py\Experiment\od\Exp3\data\VOCdevkit\VOC2007\Annotations'
image_ids = []
num_objects = []

for file in os.listdir(ann_dir):
    if file.endswith(".xml"):
        img_id = file.replace(".xml", "")
        xml_path = os.path.join(ann_dir, file)
        boxes, labels = parse_voc_xml(xml_path)
        image_ids.append(img_id)
        num_objects.append(len(boxes))

df = pd.DataFrame({
    "ImageID": image_ids,
    "NumObjects": num_objects
})

print(df.head())
