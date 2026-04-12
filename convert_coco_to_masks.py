import os
import cv2
import numpy as np
from pycocotools.coco import COCO

dataset_dir = "Archaeological-Site-Segmentation.v1-seg-v1.coco-segmentation/train"
annotation_file = os.path.join(dataset_dir, "_annotations.coco.json")

coco = COCO(annotation_file)

mask_dir = os.path.join(dataset_dir, "masks")
os.makedirs(mask_dir, exist_ok=True)

for img_id in coco.getImgIds():

    img_info = coco.loadImgs(img_id)[0]
    height = img_info["height"]
    width = img_info["width"]

    mask = np.zeros((height, width), dtype=np.uint8)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:

        class_id = ann["category_id"]
        m = coco.annToMask(ann)

        mask[m == 1] = class_id

    mask_path = os.path.join(mask_dir, img_info["file_name"].replace(".jpg", ".png"))

    cv2.imwrite(mask_path, mask)

print("Masks generated successfully")