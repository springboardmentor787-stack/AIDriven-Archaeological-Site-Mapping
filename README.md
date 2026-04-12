#  AI-Driven Archaeological Site Mapping

An end-to-end AI-powered system for analyzing archaeological sites using satellite imagery.
The project integrates **semantic segmentation, object detection, and erosion prediction** into a unified **interactive Streamlit dashboard**.

---

##  Project Overview

This system automates archaeological analysis by:

* 🧩 Segmenting terrain into meaningful classes (vegetation, soil, ruins, etc.)
* 📦 Detecting artifacts and structural elements using object detection
* 🔥 Predicting erosion-prone regions using terrain features
* 📊 Visualizing insights through a professional dashboard

---

##  Objectives

* Detect archaeological structures from satellite imagery
* Perform terrain segmentation
* Predict erosion risk using ML models
* Build an interactive and user-friendly dashboard

---

##  Project Architecture

```
Input Image
     ↓
Preprocessing
     ↓
Segmentation (U-Net / DeepLabV3+)
     ↓
Object Detection (YOLOv5 / YOLOv8)
     ↓
Feature Extraction (Slope, NDVI, Elevation)
     ↓
Erosion Prediction (XGBoost / Random Forest)
     ↓
Streamlit Dashboard Visualization
```

---

##  Project Structure

```
Internship/
│
├── Milestone 1/
│   ├── raw_data/
│   ├── preprocessing/
│   ├── yolo_data/
│
├── Milestone 2/
│   ├── segmentation/
│   ├── detection/
│
├── Milestone 3/
│   ├── erosion_model/
│   ├── dataset/
│
├── models/
│
├── app.py
├── README.md
└── .gitignore
```

---

##  Milestone Breakdown

### 🔹 Milestone 1: Data Collection & Preparation

* Collected satellite imagery dataset
* Performed preprocessing (resizing, normalization)
* Organized dataset for training

---

### 🔹 Milestone 2: Segmentation & Object Detection

* Implemented **semantic segmentation** (U-Net / DeepLabV3+)
* Classified terrain into:

  * Vegetation
  * Soil
  * Ruins
* Implemented **object detection (YOLO)**:

  * Detected artifacts like pottery, tools, structures
* Evaluated using:

  * IoU
  * Dice Score
  * mAP

---

### 🔹 Milestone 3: Terrain Erosion Prediction

* Extracted terrain features:

  * Slope
  * NDVI (vegetation index)
  * Elevation
* Built ML models:

  * Random Forest
  * XGBoost
* Evaluated using:

  * RMSE
  * R² Score
* Generated erosion risk maps

---

### 🔹 Milestone 4: Dashboard & Integration

* Built interactive dashboard using **Streamlit**
* Integrated:

  * Segmentation
  * Detection
  * Erosion prediction
* Features:

  * Image upload
  * Model selection
  * Parameter tuning
  * Multi-tab visualization

---

## 📊 Dashboard Features

###  Site Analysis

* Ruins coverage
* Artifact count
* Erosion risk score
* Model performance metrics

###  Segmentation

* Pixel-wise terrain classification
* Coverage breakdown
* IoU, Dice Score, Precision

###  Object Detection

* Bounding box visualization
* Confidence scores
* Precision vs Recall

###  Erosion Prediction

* Heatmap visualization
* Feature importance
* Temporal forecast
* Metrics: RMSE, R², MAE

### 📄 Final Report

* Summary of findings
* Recommendations for conservation

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Machine Learning (XGBoost, Random Forest)**

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📸 Sample Outputs

* Segmentation Maps
* Object Detection Results
* Erosion Heatmaps
* Interactive Dashboard

---

## ⚠️ Challenges

* Limited labeled dataset
* Synthetic data for erosion modeling
* Model generalization

---

##  Future Work

* Deploy on cloud
* Add real-time satellite data

---

## 📌 Conclusion

This project demonstrates how AI can assist archaeologists by:

* Automating site analysis
* Improving detection accuracy
* Predicting environmental risks
* Supporting conservation efforts

---

## 👨‍💻 Author

**Nikhil Dacharla**
BTech CSE | AI & Web Development Enthusiast

---
