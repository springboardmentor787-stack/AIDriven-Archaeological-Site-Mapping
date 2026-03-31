AIDriven-Archaeological-Site-Mapping Project

# Importing Google Colab files module to upload files from local system
from google.colab import files
uploaded = files.upload()

# Listing all files currently present in the working directory
import os
os.listdir()

# Creating dataset folders for images and labels if they do not already exist
import os
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

# Moving uploaded image files (.png or .jpg) into the dataset/images folder
import shutil
import os

for file in os.listdir():
    if file.endswith(".png") or file.endswith(".jpg"):
        shutil.move(file, "dataset/images/"+file)

# Renaming the exported annotation file to a simpler name for easier processing
import os

os.rename(
"Export project - Archaeology project - 3_13_2026 (1).ndjson",
"export.ndjson"
)

# Reading the NDJSON annotation file and converting bounding box annotations into label files
import json
import os

os.makedirs("dataset/labels", exist_ok=True)

classes = {
    "Ancient Structure":0,
    "Buried Wall":1,
    "Ancient Road":2,
    "Temples/ Mountains":3
}

with open("export.ndjson") as f:
    for line in f:
        data = json.loads(line)
        image_url = data["data_row"]["row_data"]
        image_name = os.path.basename(image_url)
        label_path = "dataset/labels/" + image_name.replace(".png",".txt")
        with open(label_path,"w") as label_file:
        for project in data["projects"].values():
        if "labels" not in project:
                    continue
                    for label in project["labels"]:
                    for obj in label["annotations"]["objects"]:
                    name = obj["name"]
                    bbox = obj["bounding_box"]
                    x = bbox["left"]
                        y = bbox["top"]
                        w = bbox["width"]
                        h = bbox["height"]
                        label_file.write(f"{classes[name]} {x} {y} {w} {h}\n")

# Preprocessing images by normalizing pixel values and resizing them to 640x640
import cv2
import os

image_folder = "dataset/images"

for img_name in os.listdir(image_folder):
path = os.path.join(image_folder,img_name)
    img = cv2.imread(path)
    img = img/255.0
    img = cv2.resize(img,(640,640))
    cv2.imwrite(path,(img*255).astype("uint8"))

# Splitting dataset into training and validation sets (80% training, 20% validation)
from sklearn.model_selection import train_test_split
import os, shutil
images = os.listdir("dataset/images")
train,val = train_test_split(images,test_size=0.2)

# Creating folders for training and validation image datasets
os.makedirs("dataset/train/images", exist_ok=True)
os.makedirs("dataset/val/images", exist_ok=True)

# Copying training images into the training folder
for img in train:
    shutil.copy("dataset/images/"+img,"dataset/train/images/"+img)

# Copying validation images into the validation folder
for img in val:
    shutil.copy("dataset/images/"+img,"dataset/val/images/"+img)

# Installing required libraries for segmentation model training and image processing
!pip install segmentation-models-pytorch
!pip install torch torchvision
!pip install opencv-python

# Importing required libraries for dataset handling, image processing, and deep learning
import os
import cv2
import torch
from torch.utils.data import Dataset

# Creating a custom dataset class for loading images and corresponding segmentation masks
class SegDataset(Dataset):
def __init__(self,image_dir,mask_dir):
        self.images=os.listdir(image_dir)
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        def __len__(self):
        return len(self.images)
        def __getitem__(self,idx):

 img_path=os.path.join(self.image_dir,self.images[idx])
        mask_path=os.path.join(self.mask_dir,self.images[idx])
        image=cv2.imread(img_path)
        mask=cv2.imread(mask_path,0)
        image=image/255.0
        image=torch.tensor(image).permute(2,0,1).float()
        mask=torch.tensor(mask).long()
        return image,mask

# Importing segmentation models library for implementing deep learning segmentation architectures
import segmentation_models_pytorch as smp

# Creating a U-Net segmentation model using ResNet34 as the encoder backbone
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)

# Importing NumPy for numerical operations used in evaluation metrics
import numpy as np

# Defining the Intersection over Union (IoU) metric to evaluate segmentation performance
def iou_score(pred,mask):
pred = pred.argmax(1)
intersection = (pred & mask).float().sum()
union = (pred | mask).float().sum()
return (intersection + 1e-6)/(union + 1e-6)

# Defining Dice Score metric to measure overlap between predicted and ground truth masks
def dice_score(pred,mask):
pred = pred.argmax(1)
intersection = (pred * mask).sum()
dice = (2.*intersection)/(pred.sum()+mask.sum()+1e-6)
return dice


# Creating a DeepLabV3+ segmentation model with ResNet34 backbone for improved segmentation performance
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=3,
    in_channels=3
)
# Setting the model to evaluation mode before performing validation
model.eval()

# Evaluating the model on the validation dataset and printing IoU and Dice scores
with torch.no_grad():
for img,mask in val_loader:
img=img.to(device)
        mask=mask.to(device)
        pred=model(img)
        print("IoU:",iou_score(pred,mask))
        print("Dice:",dice_score(pred,mask))

# Cloning the YOLOv5 repository from GitHub and installing its dependencies
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt

# Uploading dataset files from local system to Google Colab
from google.colab import files
files.upload()

# Unzipping the uploaded dataset file
!unzip dataset.zip

# Creating a data configuration file for YOLOv5 training
# This file specifies training images, validation images, number of classes, and class names
%%writefile data.yaml
train: dataset/images/train
val: dataset/images/val
nc: 3
names: ['ruins','vegetation','artifact']

# Training the YOLOv5 model using the specified dataset and configuration
# Image size: 640, Batch size: 16, Epochs: 50, Pretrained weights: yolov5s
!python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt

# Evaluating the trained YOLOv5 model using the best saved weights
!python val.py --weights runs/train/exp/weights/best.pt --data data.yaml




Milestone 3:
terrain-erosion-project/
│
├── data/
│   ├── raw/
│   ├── features/
│   └── labels/
│
├── scripts/
│   ├── extract_features.js
│   └── generate_labels.js
│
├── notebooks/
│   └── visualization.ipynb
│
├── README.md
└── requirements.txt

// ===============================
// TERRAIN FEATURE EXTRACTION
// ===============================

// Step 1: Define Region of Interest (ROI)
var roi = ee.Geometry.Rectangle([76.3, 15.2, 76.6, 15.5]); // Example: Hampi
Map.centerObject(roi, 10);

// Step 2: Load Sentinel-2 Data
var s2 = ee.ImageCollection('COPERNICUS/S2')
  .filterBounds(roi)
  .filterDate('2023-01-01', '2023-12-31')
  .median();

// Step 3: Compute NDVI (Vegetation Index)
var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');

// Step 4: Load Elevation Data
var dem = ee.Image('USGS/SRTMGL1_003').rename('Elevation');

// Step 5: Compute Slope
var slope = ee.Terrain.slope(dem).rename('Slope');

// Step 6: Combine Features
var features = ndvi.addBands(slope).addBands(dem);

// Step 7: Visualization
Map.addLayer(ndvi, {min: -1, max: 1, palette: ['blue','white','green']}, 'NDVI');
Map.addLayer(slope, {min: 0, max: 60}, 'Slope');
Map.addLayer(dem, {min: 0, max: 3000}, 'Elevation');

// Step 8: Export Features
Export.image.toDrive({
  image: features,
  description: 'terrain_features',
  region: roi,
  scale: 30,
  fileFormat: 'GeoTIFF'
});

// ===============================
// LABEL GENERATION (EROSION MAP)
// ===============================

// Step 1: Define ROI
var roi = ee.Geometry.Rectangle([76.3, 15.2, 76.6, 15.5]);

// Step 2: Load Data Again
var s2 = ee.ImageCollection('COPERNICUS/S2')
  .filterBounds(roi)
  .filterDate('2023-01-01', '2023-12-31')
  .median();

var dem = ee.Image('USGS/SRTMGL1_003');

// Step 3: Compute NDVI
var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');

// Step 4: Compute Slope
var slope = ee.Terrain.slope(dem).rename('Slope');

// Step 5: Define Rules
// Erosion-prone → 1
var erosion = slope.gt(20).and(ndvi.lt(0.3));

// Stable → 0
var stable = slope.lt(10).and(ndvi.gt(0.5));

// Step 6: Combine Labels
var labels = erosion.multiply(1).add(stable.multiply(0)).rename('Label');

// Step 7: Visualization
Map.addLayer(labels, {min: 0, max: 1, palette: ['green','red']}, 'Erosion Map');

// Step 8: Export Labels
Export.image.toDrive({
  image: labels,
  description: 'erosion_labels',
  region: roi,
  scale: 30,
  fileFormat: 'GeoTIFF'
});

Requirements:
rasterio
numpy
matplotlib
scikit-learn
xgboost

# Terrain Erosion Prediction using Remote Sensing & Machine Learning

## Overview
This project focuses on identifying erosion-prone areas using terrain features such as NDVI (vegetation index), slope, and elevation derived from satellite data.

## Tools Used
- Google Earth Engine
- Sentinel-2 Satellite Data
- SRTM Elevation Data
- Python (Machine Learning)

## Feature Extraction
- NDVI calculated from Sentinel-2 imagery
- Slope derived from elevation data
- Elevation from SRTM dataset

## Label Generation
Rule-based classification:
- Erosion-prone: Slope > 20 AND NDVI < 0.3
- Stable: Slope < 10 AND NDVI > 0.5

## Output
- Terrain feature maps (GeoTIFF)
- Labeled erosion dataset

## Folder Structure
data/
scripts/
notebooks/

## Future Work
- Train ML models (Random Forest / XGBoost)
- Improve labeling using manual annotation


