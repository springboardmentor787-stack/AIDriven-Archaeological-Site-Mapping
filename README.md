# AIDriven-Archaeological-Site-Mapping

An AI-based system for detecting archaeological features (ruins, vegetation, artifacts) from satellite images and predicting terrain erosion risk using machine learning.

This project combines **Computer Vision** and **Machine Learning** to assist archaeological site monitoring and preservation.

---

# Project Overview

This system performs the following tasks:

1. Detect archaeological features from satellite images
2. Segment ruins and vegetation areas
3. Predict terrain erosion risk
4. Visualize results using an interactive dashboard

---

# Project Pipeline

```
Satellite Image
      ↓
YOLO Object Detection
      ↓
U-Net Segmentation
      ↓
Random Forest Erosion Prediction
      ↓
Streamlit Dashboard Visualization
```

---

# Features

- Archaeological ruin detection
- Vegetation detection
- Artifact detection
- Terrain segmentation
- Terrain erosion prediction
- Interactive visualization dashboard

---

# Technologies Used

| Category | Tools |
|--------|--------|
| Computer Vision | YOLO (Ultralytics) |
| Segmentation | U-Net |
| Machine Learning | Random Forest |
| Dataset Platform | Roboflow |
| Visualization | Streamlit |
| Programming Language | Python |

---

# Dataset

Satellite and drone images were used to train the models.

Dataset preparation included:

- Image collection from satellite sources
- Annotation of ruins, vegetation, and artifacts
- Dataset augmentation
- Train / validation / test split

Total images used:

```
254 images
```

Dataset split:

```
Train: 225 images
Validation: 18 images
Test: 11 images
```

---

# Milestones Completed

## Milestone 1 — Dataset Collection and Preparation

Tasks completed:

- Satellite imagery collection
- Image annotation
- Data preprocessing
- Dataset splitting

Tools used:

- Roboflow
- Annotation tools
- Data augmentation

---

## Milestone 2 — Segmentation and Object Detection

### Week 3 — Segmentation Model

Implemented:

```
U-Net segmentation model
```

Purpose:

- Segment ruins
- Segment vegetation

Evaluation metrics:

```
IoU Score
Dice Score
```

Libraries used:

```
segmentation_models_pytorch
PyTorch
```

---

### Week 4 — Object Detection

Implemented:

```
YOLO object detection model
```

Detected classes:

```
Ruins
Vegetation
Artifacts
```

Evaluation metrics:

```
mAP
Precision
Recall
```

Training performed using:

```
Ultralytics YOLO
Roboflow dataset
```

---

## Milestone 3 — Terrain Erosion Prediction

A machine learning model predicts whether an area is:

```
Stable
or
Erosion-prone
```

Features used:

```
Slope
Vegetation Index
Elevation
```

Model used:

```
Random Forest Classifier
```

Evaluation metric:

```
Accuracy
```

Example prediction:

```
Slope: 28
Vegetation Index: 0.35
Elevation: 130

Prediction:
Erosion Prone Area
```

---

## Milestone 4 — Visualization Dashboard

A Streamlit dashboard was created to visualize the results.

Dashboard features:

- Upload satellite image
- Display satellite image
- Adjust terrain parameters
- Predict erosion risk

The dashboard integrates:

```
Computer Vision + Machine Learning results
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/krishnan4/archaeological-site-Mapping.git
```

Navigate to the project folder:

```bash
cd archaeological-site-detection
```

Install required libraries:

```bash
pip install ultralytics
pip install segmentation-models-pytorch
pip install streamlit
pip install opencv-python
pip install scikit-learn
pip install matplotlib
```

---

# Running the Dashboard

Run the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

# Example Outputs

Object detection example:

```
Ruins detected
Vegetation detected
Artifacts detected
```

Segmentation output:

```
Ruins segmentation mask
Vegetation segmentation mask
```

Erosion prediction example:

```
Slope: 28
Vegetation Index: 0.35
Elevation: 130

Prediction:
Erosion Prone Area
```

---

# Applications

This system can be used for:

- Archaeological site monitoring
- Heritage conservation
- Remote sensing analysis
- Terrain risk assessment

---

# Future Improvements

Possible improvements include:

- Using higher resolution satellite datasets
- Improving segmentation accuracy
- Integrating GIS mapping
- Real-time satellite monitoring

---

# Author

Developed as part of an AI/ML milestone-based project on archaeological site detection and terrain analysis.

 
