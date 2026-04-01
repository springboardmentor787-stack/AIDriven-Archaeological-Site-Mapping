# AI-Driven Archaeological Site Mapping

**Tech Stack:** Python | PyTorch | Computer Vision

---

##  Project Overview

AI-Driven Archaeological Site Mapping is a computer vision-based system designed to analyze satellite and aerial imagery to identify archaeological features such as ruins, vegetation, and terrain structures.

The system integrates object detection and semantic segmentation techniques to assist archaeologists in faster site discovery, mapping, and analysis.

---

##  System Architecture

Input Image
↓
YOLOv8 Object Detection
↓
DeepLabV3+ Segmentation
↓
Combined Feature Mapping
↓
Visualization (Streamlit)

---

## 📁 Project Structure

```
AIDriven-Archaeological-Site-Mapping/
│
├── Milestone1/
│   ├── dataset/
│   │   ├── raw_images/
│   │   └── annotations/
│   ├── preprocessing/
│   
│
├── Milestone2/
│   ├── week3.ipynb/
│   ├── week4.ipynb/
│
├── Milestone3/
│   ├── week5.ipynb/
│   ├── week6.ipynb/
│   ├── erosion_dataset (1).csv/
└── README.md
```

---

#  Milestone 1 – Dataset Collection & Preparation

##  Objective

To collect, annotate, and preprocess satellite imagery for training models.

##  Tasks Performed

* Collected satellite images from Google Earth and OpenAerialMap
* Defined annotation schema for:

  * ruins
  * vegetation
  * structures
* Annotated dataset using tools like Labelbox/QGIS
* Organized dataset into train, validation, and test sets
* Applied preprocessing:

  * resizing
  * normalization

---

#  Milestone 2 – Model Development

##  Objective

To build segmentation and detection models for archaeological feature extraction.

---

## 🔍 Object Detection (YOLOv8)

* Detects:

  * ruins
  * vegetation
  * structures
* Trained using labeled dataset
* Evaluated using:

  * mAP
  * precision
  * recall

---

##  Semantic Segmentation (DeepLabV3+)

* Segments landscape into:

  * ruins
  * vegetation
  * background
* Evaluated using:

  * IoU
  * Dice Score

---

##  Combined Pipeline

* Integrated detection + segmentation
* Generates complete archaeological mapping

---

#  Tech Stack

## Core

* Python
* PyTorch
* NumPy

## Computer Vision

* YOLOv8
* DeepLabV3+
* OpenCV

## Visualization

* Streamlit
* Matplotlib

---

#  How to Run

## Install Dependencies

```
pip install -r requirements.txt
```

## Run Pipeline

```
python demo_pipeline.py
```

## Launch App

```
streamlit run app.py
```

---

#  Results

* Detection accuracy evaluated using mAP
* Segmentation evaluated using IoU & Dice Score
* Visual outputs generated for archaeological mapping

---

#  Milestone 3 – Terrain Erosion Prediction

##  Objective

To predict erosion-prone areas using terrain features derived from satellite imagery.

---

##  Week 5: Feature Extraction & Dataset Preparation

* Reused satellite images from previous milestones
* Extracted terrain features:

  * Slope (using gradient)
  * Vegetation Index (NDVI)
  * Elevation (approximated)
* Created pixel-wise dataset from images
* Applied random sampling (200,000 samples) for optimization
* Generated labels using rule-based conditions:

  * High slope + low vegetation → erosion-prone

---

##  Week 6: Model Training & Evaluation

* Split dataset into training and testing sets
* Trained models:

  * Random Forest
  * XGBoost
* Evaluated using:

  * RMSE
  * R² Score
* Converted regression output to classification using thresholding
* Achieved high accuracy (~99.9%)

---

## 📊 Results

* Random Forest: RMSE ≈ 0.0, R² ≈ 1.0
* XGBoost: RMSE ≈ 0.027, R² ≈ 0.995
* Classification Accuracy: ~99.9%

---

## Key Insight

Terrain features such as slope and vegetation index are strong indicators of erosion-prone regions.

---
