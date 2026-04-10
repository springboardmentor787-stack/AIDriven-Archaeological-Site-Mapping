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

MILESTONE 4:
import random
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AI Archaeological Platform",
    layout="wide",
)

# Application style with enhanced visuals
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #f8fafc;
    }
    .css-1d391kg {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }
    .stSidebar {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
        border-right: 2px solid #334155;
    }
    .stSidebar .css-1d391kg {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #f8fafc;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .stButton>button {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: #f8fafc;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    .card {
        border-radius: 20px;
        padding: 25px;
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .metric-card {
        border-radius: 16px;
        padding: 20px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .feature-chip {
        display: inline-block;
        background: linear-gradient(45deg, rgba(59, 130, 246, 0.2), rgba(147, 51, 234, 0.2));
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 999px;
        color: #e0e7ff;
        padding: 8px 16px;
        margin: 4px 8px 4px 0;
        font-size: 13px;
        font-weight: 500;
    }
    .warning-box {
        background: linear-gradient(45deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.15));
        color: #fbbf24;
        border-left: 4px solid #fbbf24;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.1);
    }
    .success-box {
        background: linear-gradient(45deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.15));
        color: #86efac;
        border-left: 4px solid #22c55e;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.1);
    }
    .error-box {
        background: linear-gradient(45deg, rgba(248, 113, 113, 0.15), rgba(220, 38, 38, 0.15));
        color: #fecaca;
        border-left: 4px solid #ef4444;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(248, 113, 113, 0.1);
    }
    .progress-bar {
        width: 100%;
        height: 8px;
        background: rgba(148, 163, 184, 0.2);
        border-radius: 4px;
        overflow: hidden;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .glow-text {
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.5), 0 0 20px rgba(59, 130, 246, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page navigation
page = st.sidebar.selectbox("Navigation", ["Home", "Analysis", "About"])

# Custom components

def segmentation_model(image: np.ndarray, blur_level: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_level * 2 + 1, blur_level * 2 + 1), 0)
    edges = cv2.Canny(blurred, threshold1=70, threshold2=140)
    segmented = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    segmented = cv2.addWeighted(image, 0.5, segmented, 0.7, 0)
    return segmented


def detection_model(image: np.ndarray, confidence: float) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    colors = [(255, 159, 28), (34, 197, 94), (59, 130, 246)]
    num_boxes = max(1, int(confidence * 5))

    for i in range(num_boxes):
        x1 = random.randint(0, max(0, width - width // 4))
        y1 = random.randint(0, max(0, height - height // 4))
        x2 = random.randint(x1 + width // 8, min(width, x1 + width // 2))
        y2 = random.randint(y1 + height // 8, min(height, y1 + height // 2))
        color = colors[i % len(colors)]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            output,
            f"Feature {i + 1}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return output


def erosion_model(sensitivity: float) -> tuple[str, str]:
    if sensitivity < 0.3:
        risk = random.choice(["Low", "Low", "Medium"])
    elif sensitivity < 0.7:
        risk = random.choice(["Medium", "Medium", "High"])
    else:
        risk = random.choice(["High", "High", "Medium"])
    color = {"Low": "green", "Medium": "orange", "High": "red"}[risk]
    return risk, color


def artifact_classification_model() -> tuple[str, str, str]:
    artifacts = [
        ("Pottery Shard", "Bronze Age", "Ceramic vessel fragment"),
        ("Stone Tool", "Paleolithic Era", "Flint tool for cutting"),
        ("Ceramic Fragment", "Iron Age", "Decorative pottery piece"),
        ("Ancient Coin", "Roman Era", "Currency from ancient Rome"),
        ("Bone Relic", "Medieval Period", "Animal bone artifact"),
        ("Metal Artifact", "Classical Period", "Bronze figurine")
    ]
    artifact, era, description = random.choice(artifacts)
    return artifact, era, description


def confidence_score_model() -> int:
    return random.randint(85, 98)


def gemini_analyze_image(image: Image.Image, api_key: str) -> str:
    if not api_key:
        return "API key not provided. Enable Gemini analysis by entering your API key."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """
        Analyze this archaeological image and provide a detailed report including:
        1. Description of visible features and artifacts
        2. Potential historical context and era
        3. Preservation condition assessment
        4. Recommendations for further study
        5. Cultural significance insights
        """
        response = model.generate_content([prompt, image], generation_config={"temperature": 0.7})
        return response.text
    except Exception as e:
        return f"Error with Gemini AI: {str(e)}"


def plot_erosion_distribution():
    labels = ["Low", "Medium", "High"]
    values = [random.randint(25, 50), random.randint(20, 40), random.randint(15, 35)]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#22c55e", "#f97316", "#ef4444"], alpha=0.9, edgecolor='white', linewidth=1)
    ax.set_title("Erosion Risk Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Risk Score", fontsize=12)
    ax.set_ylim(0, max(values) + 15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(value), ha="center", va="bottom", fontsize=11, fontweight='bold')
    st.pyplot(fig)


def plot_artifact_pie():
    labels = ['Pottery', 'Tools', 'Ceramics', 'Coins', 'Bones', 'Metals']
    sizes = [random.randint(10, 30) for _ in range(6)]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#22c55e', '#06b6d4'])
    ax.axis('equal')
    ax.set_title("Artifact Type Distribution", fontsize=14, fontweight='bold')
    st.pyplot(fig)


def plot_feature_heatmap():
    data = np.random.rand(8, 8)
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.imshow(data, cmap="plasma", interpolation="nearest")
    ax.set_title("Feature Intensity Heatmap", fontsize=14, fontweight='bold')
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def ai_insight_cards(risk: str, confidence: int, artifact: str, era: str, terrain_type: str, scan_mode: str, description: str):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Erosion Risk", risk, delta=None)
    col2.metric("AI Confidence", f"{confidence}%", delta="High")
    col3.metric("Primary Artifact", artifact, delta=era)
    col4.metric("Terrain Type", terrain_type, delta=scan_mode)
    st.markdown(f"**Artifact Description:** {description}")


def generate_report(analysis_data: dict) -> str:
    report = f"""
# AI Archaeological Analysis Report

## Site Overview
- **Terrain Type:** {analysis_data['terrain']}
- **Scan Mode:** {analysis_data['scan_mode']}
- **Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings
- **Primary Artifact:** {analysis_data['artifact']} ({analysis_data['era']})
- **Erosion Risk Level:** {analysis_data['erosion_risk']}
- **AI Confidence Score:** {analysis_data['confidence']}%
- **Artifact Description:** {analysis_data['description']}

## Visual Analysis Summary
- Segmentation: Edge detection applied with blur level {analysis_data['blur']}
- Detection: {analysis_data['num_boxes']} potential artifacts identified
- Erosion Assessment: Based on sensitivity threshold {analysis_data['sensitivity']}

## Recommendations
{analysis_data['recommendations']}

## Gemini AI Insights
{analysis_data['gemini_insight']}

---
*Report generated by AI Archaeological Platform v2.0*
"""
    return report


def show_home():
    st.markdown("<h1 class='glow-text'>Welcome to the AI Archaeological Platform</h1>", unsafe_allow_html=True)
    st.markdown(
        "Discover the future of archaeological research with our advanced AI-powered analysis tools."
    )

    # Feature showcase
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Platform Capabilities")
    features = [
        "**Advanced Image Segmentation** - Precise edge detection and feature extraction",
        "**Smart Object Detection** - AI-powered artifact identification with confidence scoring",
        "**Erosion Risk Assessment** - Predictive modeling for site preservation",
        "**Gemini AI Integration** - Contextual analysis and historical insights",
        "**Data Visualization** - Interactive charts and heatmaps for comprehensive analysis",
        "**Automated Reporting** - Export detailed analysis reports"
    ]
    for feature in features:
        st.markdown(f"<div class='feature-chip'>{feature}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Analysis Types")
        st.markdown("**4 Core Modules**")
        st.markdown("Segmentation, Detection, Erosion, AI Insights")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Visual Outputs")
        st.markdown("**Interactive Charts**")
        st.markdown("Heatmaps, Distributions, Pie Charts")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### AI Integration")
        st.markdown("**Gemini Powered**")
        st.markdown("Contextual Image Analysis & Reports")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Get Started")
    if st.button("Start Analysis", type="primary"):
        st.session_state.page = "Analysis"
        st.rerun()


def show_about():
    st.markdown("## About This Demo")
    st.markdown(
        "This application is a demo of an AI Archaeological Platform built with Streamlit. It uses simulated vision models and optional Gemini AI integration for presentation purposes."
    )
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Why this works for demos")
    st.markdown(
        "- Clean page structure with Home, Analysis, and About sections.\n"
        "- Professional dark user interface suitable for presentations.\n"
        "- Configurable analysis parameters for interactive exploration.\n"
        "- Optional Gemini AI connection for real language-based insights."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("### Notes")
    st.markdown(
        "- Gemini AI requires a valid API key to generate insights.\n"
        "- This demo simulates computer vision behavior using OpenCV.\n"
        "- The platform is intended for presentation and learning, not clinical archaeological analysis."
    )


def display_analysis():
    st.markdown("<h1 class='glow-text'>Archaeological Analysis Workspace</h1>", unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("## Analysis Configuration")
    terrain_type = st.sidebar.selectbox("Terrain Type", ["Rocky", "Soil", "Mixed"], index=0)
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["All", "Segmentation", "Detection", "Erosion"],
        index=0,
        help="Select All to view complete analysis suite."
    )
    scan_mode = st.sidebar.radio("Scan Mode", ["Quick Scan", "Deep Scan", "Full AI Suite"], index=0)

    st.sidebar.markdown("### Advanced Parameters")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.7, help="Higher values detect more features.")
    erosion_sensitivity = st.sidebar.slider("Erosion Sensitivity", 0.1, 1.0, 0.5, help="Adjusts erosion risk calculation.")
    segmentation_blur = st.sidebar.slider("Segmentation Blur", 1, 10, 4, help="Controls edge detection smoothness.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Gemini AI Integration")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key.")
    use_gemini = st.sidebar.checkbox("Enable Gemini AI Analysis", value=True, help="Activate AI-powered insights.")

    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload Archaeological Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.warning("Please upload an archaeological image to begin analysis.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Image Preview")
    st.image(image, width=800)
    st.markdown("</div>", unsafe_allow_html=True)

    # Progress simulation
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Processing archaeological AI analysis..."):
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("Preprocessing image...")
            elif i < 60:
                status_text.text("Running AI segmentation...")
            elif i < 80:
                status_text.text("Analyzing artifacts...")
            else:
                status_text.text("Generating insights...")

        segmentation_result = None
        detection_result = None
        erosion_risk = None
        artifact_type = None
        artifact_era = None
        artifact_desc = None
        confidence = None
        gemini_insight = None
        num_boxes = 0

        if analysis_mode in ["Segmentation", "All"]:
            segmentation_result = segmentation_model(image_cv, segmentation_blur)
        if analysis_mode in ["Detection", "All"]:
            detection_result = detection_model(image_cv, detection_confidence)
            num_boxes = max(1, int(detection_confidence * 6))
        if analysis_mode in ["Erosion", "All"]:
            erosion_risk, _ = erosion_model(erosion_sensitivity)
        artifact_type, artifact_era, artifact_desc = artifact_classification_model()
        confidence = confidence_score_model()
        if use_gemini and gemini_api_key:
            gemini_insight = gemini_analyze_image(image, gemini_api_key)

    progress_bar.empty()
    status_text.empty()

    st.markdown("---")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Analysis Results Dashboard")

    tabs = st.tabs(["Visual Analysis", "AI Insights", "Data Charts", "Gemini AI", "Full Report"])

    with tabs[0]:
        st.markdown("#### Visual Analysis Results")
        if analysis_mode == "All":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Segmentation Overlay")
                if segmentation_result is not None:
                    segmented_rgb = cv2.cvtColor(segmentation_result, cv2.COLOR_BGR2RGB)
                    st.image(segmented_rgb, width=700, caption="AI-generated segmentation mask")
            with col2:
                st.markdown("##### Object Detection")
                if detection_result is not None:
                    detected_rgb = cv2.cvtColor(detection_result, cv2.COLOR_BGR2RGB)
                    st.image(detected_rgb, width=700, caption=f"Detected {num_boxes} potential artifacts")
        else:
            st.info(f"Showing {analysis_mode} analysis only. Select 'All' for complete visual suite.")

    with tabs[1]:
        st.markdown("#### AI-Powered Insights")
        ai_insight_cards(erosion_risk if erosion_risk else "N/A", confidence, artifact_type, artifact_era, terrain_type, scan_mode, artifact_desc)

        if erosion_risk == "High":
            st.markdown("<div class='error-box'>**High Erosion Risk Detected** - Immediate preservation measures recommended.</div>", unsafe_allow_html=True)
        elif erosion_risk == "Medium":
            st.markdown("<div class='warning-box'>**Medium Erosion Risk Detected** - Schedule monitoring and follow-up surveys.</div>", unsafe_allow_html=True)
        elif erosion_risk == "Low":
            st.markdown("<div class='success-box'>**Low Erosion Risk Detected** - Conditions appear stable for excavation.</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("#### Data Visualizations")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("##### Erosion Risk Distribution")
            plot_erosion_distribution()
        with col2:
            st.markdown("##### Artifact Type Breakdown")
            plot_artifact_pie()

        st.markdown("##### Feature Intensity Heatmap")
        plot_feature_heatmap()

    with tabs[3]:
        st.markdown("#### Gemini AI Analysis")
        if use_gemini:
            if gemini_api_key:
                if gemini_insight:
                    st.markdown(f"<div class='card'>{gemini_insight}</div>", unsafe_allow_html=True)
                else:
                    st.info("Gemini analysis in progress...")
            else:
                st.warning("Please enter your Gemini API key to activate AI insights.")
        else:
            st.info("Enable Gemini AI in the sidebar for contextual analysis.")

    with tabs[4]:
        st.markdown("#### Comprehensive Analysis Report")

        analysis_data = {
            'terrain': terrain_type,
            'scan_mode': scan_mode,
            'artifact': artifact_type,
            'era': artifact_era,
            'erosion_risk': erosion_risk,
            'confidence': confidence,
            'description': artifact_desc,
            'blur': segmentation_blur,
            'num_boxes': num_boxes,
            'sensitivity': erosion_sensitivity,
            'recommendations': "1. Document all findings with high-resolution photography\n2. Conduct soil analysis for preservation assessment\n3. Consult with local archaeological authorities\n4. Plan controlled excavation if conditions permit",
            'gemini_insight': gemini_insight if gemini_insight else "Gemini analysis not performed"
        }

        report = generate_report(analysis_data)
        st.markdown(f"<div class='card' style='font-family: monospace; white-space: pre-wrap;'>{report}</div>", unsafe_allow_html=True)

        # Export functionality
        if st.button("Download Report as Text File"):
            buffer = BytesIO()
            buffer.write(report.encode('utf-8'))
            buffer.seek(0)
            st.download_button(
                label="Download Report",
                data=buffer,
                file_name=f"archaeological_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


# Page routing
if page == "Home":
    show_home()
elif page == "Analysis":
    display_analysis()
else:
    show_about()



