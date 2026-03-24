# AIDriven-Archaeological-Site-Mapping
# 🏛️ AI-Based Archaeological Site Mapping & Erosion Prediction (Chettinad Region)

## 📌 Project Overview

This project presents an AI-powered system designed to detect archaeological features and predict terrain erosion risk using satellite imagery from the **Chettinad region (India)**.

The system integrates **Computer Vision** and **Machine Learning** techniques to support archaeological monitoring, preservation, and terrain risk analysis.

---

## 🚀 Project Pipeline

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

## ✨ Features

* 🏺 Archaeological ruin detection
* 🌿 Vegetation detection
* 🧱 Artifact detection
* 🗺️ Terrain segmentation
* ⚠️ Terrain erosion prediction
* 📊 Interactive visualization dashboard

---

## 🛠️ Technologies Used

| Category         | Tools Used         |
| ---------------- | ------------------ |
| Computer Vision  | YOLO (Ultralytics) |
| Segmentation     | U-Net              |
| Machine Learning | Random Forest      |
| Dataset Platform | Roboflow           |
| Visualization    | Streamlit          |
| Programming      | Python             |

---

## 📂 Dataset

The dataset consists of **satellite images from the Chettinad region**, used for detecting archaeological structures and terrain features.

### Dataset Preparation

* Satellite image collection
* Annotation of:

  * Ruins
  * Vegetation
  * Artifacts
* Data augmentation
* Dataset splitting

### 📊 Dataset Details

* **Total Images:** 250
* **Train Set:** 220 images
* **Validation Set:** 20 images
* **Test Set:** 10 images

---

## 🧩 Milestones Completed

### 📍 Milestone 1 — Dataset Collection & Preparation

**Tasks Completed:**

* Satellite imagery collection (Chettinad region)
* Image annotation
* Data preprocessing
* Dataset splitting

**Tools Used:**

* Roboflow
* Annotation tools
* Data augmentation

---

### 🤖 Milestone 2 — Segmentation & Object Detection

#### Week 3 — Segmentation Model

**Implemented:**

* U-Net segmentation model

**Purpose:**

* Segment ruins
* Segment vegetation

**Evaluation Metrics:**

* IoU Score
* Dice Score

**Libraries Used:**

* segmentation_models_pytorch
* PyTorch

---

#### Week 4 — Object Detection

**Implemented:**

* YOLO object detection model

**Detected Classes:**

* Ruins
* Vegetation
* Artifacts

**Evaluation Metrics:**

* mAP
* Precision
* Recall

**Training Tools:**

* Ultralytics YOLO
* Roboflow dataset

---

### 🌍 Milestone 3 — Terrain Erosion Prediction

A machine learning model predicts whether terrain is:

* Stable
* Erosion-prone

**Features Used:**

* Slope
* Vegetation Index
* Elevation

**Model Used:**

* Random Forest Classifier

**Evaluation Metric:**

* Accuracy

**Example Prediction:**

```
Slope: 28
Vegetation Index: 0.35
Elevation: 130

Prediction:
Erosion Prone Area
```

---

### 📊 Milestone 4 — Visualization Dashboard

A **Streamlit dashboard** was developed to visualize model outputs.

**Dashboard Features:**

* Upload satellite images
* Display processed images
* Adjust terrain parameters
* Predict erosion risk

**Integration:**

* Computer Vision outputs
* Machine Learning predictions

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/archaeological-site-mapping.git
```

### 2️⃣ Navigate to Project Folder

```
cd archaeological-site-mapping
```

### 3️⃣ Install Dependencies

```
pip install ultralytics
pip install segmentation-models-pytorch
pip install streamlit
pip install opencv-python
pip install scikit-learn
pip install matplotlib
```

---

## ▶️ Running the Application

Run the Streamlit dashboard:

```
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Example Outputs

### 🔍 Object Detection

* Ruins detected
* Vegetation detected
* Artifacts detected

### 🧠 Segmentation Output

* Ruins segmentation mask
* Vegetation segmentation mask

### ⚠️ Erosion Prediction

```
Slope: 28
Vegetation Index: 0.35
Elevation: 130

Prediction:
Erosion Prone Area
```

---

## 🌍 Applications

* Archaeological site monitoring
* Heritage conservation
* Remote sensing analysis
* Terrain risk assessment

---

## 🔮 Future Improvements

* Use higher-resolution satellite datasets
* Improve segmentation accuracy
* Integrate GIS mapping
* Enable real-time satellite monitoring

---

## 👨‍💻 Author

Developed as part of an AI/ML milestone-based academic project focused on the **Chettinad archaeological region**.
