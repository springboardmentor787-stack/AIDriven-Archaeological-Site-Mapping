import cv2
import torch
import numpy as np
import pandas as pd
import os
import random
import segmentation_models_pytorch as smp


def resolve_segmentation_checkpoint():
    candidates = [
        os.path.join("models", "deeplab_model.pth"),
        "deeplab_model.pth",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("No segmentation checkpoint found.")

# ----------------------------
# Load DeepLab Segmentation Model
# ----------------------------

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=6
)

model.load_state_dict(torch.load(resolve_segmentation_checkpoint(), map_location="cpu"))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ----------------------------
# Image Dataset Path
# ----------------------------

IMAGE_FOLDER = "dataset/train/images"

data = []

print("Extracting terrain features...\n")

for img_name in os.listdir(IMAGE_FOLDER):

    img_path = os.path.join(IMAGE_FOLDER, img_name)

    image = cv2.imread(img_path)

    if image is None:
        continue

    image = cv2.resize(image,(512,512))

    img_tensor = torch.tensor(
        image.transpose(2,0,1)/255.0,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(img_tensor)
        mask = torch.argmax(pred,dim=1).squeeze().cpu().numpy()

    # class indices: 1=boulders, 3=ruins, 4=structures, 5=vegetation
    boulders_pixels = np.sum(mask == 1)
    vegetation_pixels = np.sum(mask == 5)
    ruins_pixels = np.sum(mask == 3)
    structures_pixels = np.sum(mask == 4)
    total_pixels = mask.size

    boulders_ratio = boulders_pixels / total_pixels
    vegetation_ratio = vegetation_pixels / total_pixels
    ruins_ratio = ruins_pixels / total_pixels
    structures_ratio = structures_pixels / total_pixels

    # simulate slope
    slope = np.random.uniform(0,45)

    # simulate elevation
    elevation = np.random.uniform(350,450)

    score = (
        0.5 * slope +
        -40 * vegetation_ratio +
        15 * boulders_ratio +
        30 * ruins_ratio +
        20 * structures_ratio +
        0.1 * elevation +
        random.uniform(-5, 5)
    )

    erosion_risk = 1 if score > 10 else 0

    data.append([
        slope,
        vegetation_ratio,
        elevation,
        boulders_ratio,
        ruins_ratio,
        structures_ratio,
        erosion_risk
    ])

print("Feature extraction complete")

df = pd.DataFrame(
    data,
    columns=[
        "slope",
        "vegetation_ratio",
        "elevation",
        "boulders_ratio",
        "ruins_ratio",
        "structures_ratio",
        "erosion_risk"
    ]
)

df.to_csv("terrain_model/erosion_dataset.csv", index=False)

print("\nDataset saved as terrain_model/erosion_dataset.csv")