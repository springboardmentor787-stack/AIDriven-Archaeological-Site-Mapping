import os
import cv2
import numpy as np
from pycocotools.coco import COCO

# CHANGE THIS PATH
DATASET_PATH = "seg_dataset"

splits = ["train", "valid", "test"]

for split in splits:

    print(f"Processing {split}...")

    split_path = os.path.join(DATASET_PATH, split)
    annotation_file = os.path.join(split_path, "_annotations.coco.json")
    images_dir = os.path.join(split_path, "images")
    masks_dir = os.path.join(split_path, "masks")

    os.makedirs(masks_dir, exist_ok=True)

    coco = COCO(annotation_file)

    img_ids = coco.getImgIds()

    for img_id in img_ids:

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

        mask_path = os.path.join(masks_dir, img_info["file_name"].replace(".jpg", ".png"))

        cv2.imwrite(mask_path, mask)

print("✅ Masks generated successfully!")