# AIDriven-Archaeological-Site-Mapping
Project: AI-Driven Archaeological Site Mapping
Milestone 1: Dataset Collection & Preparation (Weeks 1–2)
1. Data Collection
Objective:

Gather high-resolution satellite and drone imagery for archaeological analysis.

Sources Used:

Google Earth Pro

OpenAerialMap

Custom drone imagery (if available)

Process:

Identified archaeological regions of interest.

Downloaded high-resolution imagery.

Ensured:

Clear terrain visibility

Minimal cloud cover

Consistent resolution

Organized images into structured folders.

Example structure:

Dataset/
    Ruins/
    Vegetation/
    Artifacts/
    Background/

Outcome:
✔ Collected structured raw imagery dataset.

2. Annotation Schema Definition
Objective:

Define how images will be labeled.

Defined Classes:
Category	Description
Ruins	Ancient walls, foundations, structural remains
Vegetation	Grass, trees, shrubs covering site
Artifacts	Visible structures, pillars, carved remains
Background	Non-relevant terrain

For segmentation:

Pixel-level mask labeling (ruins vs vegetation vs soil)

For detection:

Bounding boxes around artifacts

Outcome:
✔ Clear labeling strategy for classification, segmentation, and detection tasks.

3. Image Annotation
Tools Used:

QGIS (for spatial annotation)

Labelbox (for bounding box labeling)

Folder-based labeling (for classification task)

What Was Done:

✔ Classification labeling:
Images organized into folders representing classes.

✔ Segmentation labeling:
Binary or multi-class masks created (if required).

✔ Object detection:
Bounding boxes defined around artifact structures.

Outcome:
✔ Images paired with labels / masks / annotations.

4. Dataset Preparation (What You Implemented in Code)

You performed:

✔ Automatic label extraction from folder names
✔ Created annotation DataFrame
✔ Train-Test split (80-20 stratified)
✔ Image resizing & normalization
✔ Generator creation for deep learning

This prepares dataset for:

CNN classification

Future segmentation models

Object detection models

5. Preprocessing Steps Applied

Image resizing (64x64 / 128x128)

Pixel normalization (rescale 1./255)

Train-test split

Class encoding (automatic via Keras
MILE STONE 2
AI Archaeological Site Mapping using YOLOv8
1. Project Overview

This project uses Artificial Intelligence and Computer Vision to detect archaeological structures from aerial or satellite images. The system is built using the YOLOv8 object detection model and trained on annotated archaeological images.

The goal of this project is to help archaeologists automatically identify ruins, structures, and historical remains from satellite imagery. This reduces manual effort and speeds up archaeological site analysis.

2. Objectives

Detect archaeological structures from aerial imagery.

Train an object detection model using YOLOv8.

Use labeled datasets for supervised learning.

Visualize detected structures using bounding boxes.

Improve archaeological site mapping using AI.

3. Technologies Used
Technology	Purpose
Python	Programming language
Google Colab	Cloud environment for training
YOLOv8	Object detection model
Roboflow	Dataset management and annotation
OpenCV	Image processing
Matplotlib	Image visualization
4. Project Workflow

Dataset Collection
↓
Dataset Annotation
↓
Dataset Download from Roboflow
↓
Model Training using YOLOv8
↓
Model Evaluation
↓
Prediction on Test Images

5. Dataset Structure

The dataset is downloaded in YOLO format and has the following folder structure:

Archaeological-Site-Mapping-6

train
 ├── images
 └── labels

valid
 ├── images
 └── labels

test
 ├── images
 └── labels

data.yaml
6. Images

This folder contains satellite or aerial images of archaeological locations used for training and testing the model.

7. Labels

Each image has a corresponding .txt file containing bounding box coordinates.

Example:

0 0.52 0.61 0.20 0.15
Value	Description
0	Class ID
0.52	X center
0.61	Y center
0.20	Width
0.15	Height

Coordinates are normalized values between 0 and 1.

8. Step 1: Setup Environment

Run the following commands to install required libraries.

!pip install ultralytics
!pip install roboflow

These libraries allow us to train YOLO models and download datasets.

9. Step 2: Import Libraries
from ultralytics import YOLO
from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
import os
import random

Explanation

Ultralytics YOLO – Used to train and run the detection model

Roboflow – Used to download datasets

OpenCV – Used for image processing

Matplotlib – Used to display images

10. Step 3: Download Dataset
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("harshils-workspace").project("archaeological-site-mapping")
version = project.version(6)
dataset = version.download("yolov8")

This code connects to Roboflow and downloads the dataset in YOLO format.

11. Step 4: Train YOLO Model
model = YOLO("yolov8n.pt")

model.train(
    data="/content/Archaeological-Site-Mapping-6/data.yaml",
    epochs=50,
    imgsz=640
)
Parameter	Meaning
yolov8n.pt	Pretrained YOLO model
epochs	Number of training cycles
imgsz	Image resolution

Training produces output files such as:

runs/detect/train/

Inside this folder:

best.pt
last.pt
results.png
confusion_matrix.png
12. Step 5: View Training Results
from IPython.display import Image
Image("runs/detect/train/results.png")

This shows training graphs such as:

Loss

Precision

Recall

mAP accuracy

13. Step 6: Run Prediction
model = YOLO("runs/detect/train/weights/best.pt")

model.predict(
    source="/content/Archaeological-Site-Mapping-6/test/images",
    save=True
)

The trained model predicts objects in test images.

Results are saved in:

runs/detect/predict/
14. Step 7: Display Prediction Images
from IPython.display import Image, display
import glob

for img in glob.glob("/content/runs/detect/predict/*.jpg")[:5]:
    display(Image(filename=img))

This displays images with detected archaeological structures and bounding boxes.

15. Step 8: Visualize Dataset Labels

To verify dataset annotations:

img_dir = "/content/Archaeological-Site-Mapping-6/train/images"
label_dir = "/content/Archaeological-Site-Mapping-6/train/labels"

img_file = random.choice(os.listdir(img_dir))
img = cv2.imread(f"{img_dir}/{img_file}")

h, w = img.shape[:2]

label_file = img_file.replace(".jpg",".txt").replace(".png",".txt")

with open(f"{label_dir}/{label_file}") as f:
    for line in f:
        c,x,y,bw,bh = map(float,line.split())

        x1 = int((x-bw/2)*w)
        y1 = int((y-bh/2)*h)
        x2 = int((x+bw/2)*w)
        y2 = int((y+bh/2)*h)

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

This code displays bounding boxes from the dataset labels.
