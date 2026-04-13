# 🏛️ AI-Driven Archaeological Site Mapping

> An end-to-end AI platform that analyses satellite and drone imagery to assist archaeologists in segmenting ancient ruins and vegetation, detecting artifact structures, and predicting erosion-prone zones — delivered through an interactive Streamlit dashboard.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/YOLOv5-Ultralytics-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/Google%20Colab-GPU%20T4-yellow?style=flat-square&logo=googlecolab" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [What the System Does](#what-the-system-does)
- [System Architecture](#system-architecture)
- [Complete Project Flow](#complete-project-flow)
- [Algorithms Used and Why](#algorithms-used-and-why)
- [Evaluation Scores](#evaluation-scores)
- [Tech Stack](#tech-stack)
- [Dataset Sources](#dataset-sources)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Dashboard Preview](#dashboard-preview)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Archaeological sites across the world are threatened by vegetation encroachment, soil erosion, and undiscovered artifact structures that go undetected without expensive manual survey work. This project builds a complete AI pipeline that automates three core archaeological analysis tasks using satellite and drone imagery:

- Identify and colour-code every pixel as ruins, vegetation, or erosion zone
- Locate and label individual artifact structures with bounding boxes
- Score terrain patches for erosion risk from 0 (safe) to 1 (critical)

All three outputs are combined into a single Streamlit web dashboard that any archaeologist can use by simply uploading an image — no coding required.

---

## What the System Does

```
Archaeologist uploads a satellite image
              |
              v
+----------------------------+
|   AI Processing Pipeline   |
|                            |
|  Step 1: Segmentation      |  <-- U-Net labels every pixel
|  Step 2: Detection         |  <-- YOLOv5 finds artifacts
|  Step 3: Erosion scoring   |  <-- XGBoost predicts risk
+----------------------------+
              |
              v
  Interactive Streamlit Dashboard
  - Segmentation mask overlay
  - Bounding boxes on artifacts
  - Erosion risk score + colour map
  - Folium geographic map
```

---

## System Architecture

```
+------------------------------------------------------------------+
|                     COMPLETE SYSTEM FLOW                        |
|                                                                  |
|  [Satellite / Drone Images]  +  [Annotation Labels]             |
|              |                                                   |
|              v                                                   |
|  +------------------------+                                      |
|  |   DATA PIPELINE        |                                      |
|  |  Download images        |                                     |
|  |  Annotate (COCO/YOLO)  |                                      |
|  |  Resize + Normalise    |                                      |
|  |  Augment (3x dataset)  |                                      |
|  |  Split 75/15/10        |                                      |
|  +----------+-------------+                                      |
|             |                                                    |
|             | split/ + metadata.json                            |
|             v                                                    |
|  +-----------------------------+                                 |
|  |   SEGMENTATION MODEL        |   Algorithm: U-Net             |
|  |   Labels every pixel        |   Scores: IoU, Dice            |
|  +-----------------------------+                                 |
|  +-----------------------------+                                 |
|  |   DETECTION MODEL           |   Algorithm: YOLOv5            |
|  |   Finds artifact locations  |   Scores: mAP, P, R            |
|  +-----------------------------+                                 |
|             |                                                    |
|             | model weights (.pth, .pt)                         |
|             v                                                    |
|  +-----------------------------+                                 |
|  |   EROSION PREDICTION        |   Algorithm: XGBoost + RF      |
|  |   13 terrain features       |   Scores: RMSE, R2, MAE        |
|  |   Risk score 0 to 1         |   SHAP explainability           |
|  |   Folium HTML map           |                                 |
|  +-----------------------------+                                 |
|             |                                                    |
|             | pkl models + erosion map                          |
|             v                                                    |
|  +-----------------------------+                                 |
|  |   STREAMLIT DASHBOARD       |                                 |
|  |   Upload image              |                                 |
|  |   Get all 3 predictions     |                                 |
|  |   Interactive Folium map    |                                 |
|  +-----------------------------+                                 |
+------------------------------------------------------------------+
```

---

## Complete Project Flow

### How Google Drive connects all notebooks

Every notebook saves its outputs to Google Drive so the next notebook can read them even in a new Colab session.

```
+------------------+     saves     +--------------------------------+
|  Milestone 1     | ------------> | MyDrive/Archaeological_AI/     |
|  Data pipeline   |               |   project/split/segmentation/  |
|  (Weeks 1-2)     |               |   project/split/detection/     |
+------------------+               |   project/processed/           |
                                   |     milestone1_metadata.json   |
                                   |     artifacts_data.yaml        |
                                   +--------------------------------+
                                          |
                                          | reads split/ + metadata
                                          v
+------------------+     saves     +--------------------------------+
|  Milestone 2     | ------------> | MyDrive/Archaeological_AI/     |
|  Seg + Detection |               |   Milestone2_Output/           |
|  (Weeks 3-4)     |               |     best_seg_model.pth         |
+------------------+               |     yolo_artifacts/best.pt     |
                                   |     milestone2_metrics.json    |
                                   +--------------------------------+
                                          |
                                          | reads images + M2 weights
                                          v
+------------------+     saves     +--------------------------------+
|  Milestone 3     | ------------> | MyDrive/Archaeological_AI/     |
|  Erosion model   |               |   Milestone3_Output/           |
|  (Weeks 5-6)     |               |     random_forest.pkl          |
+------------------+               |     xgboost.pkl                |
                                   |     scaler.pkl                 |
                                   |     imputer.pkl                |
                                   |     erosion_risk_map.html      |
                                   |     milestone3_metrics.json    |
                                   +--------------------------------+
                                          |
                                          | reads ALL above outputs
                                          v
+------------------+
|  Milestone 4     |  Loads all models -> Streamlit dashboard
|  Dashboard       |  Archaeologist uploads image -> 3 predictions
|  (Weeks 7-8)     |
+------------------+
```

### Step-by-step what each code cell does

**Data pipeline steps:**
1. Install libraries — PyTorch, segmentation-models-pytorch, Ultralytics, pycocotools, albumentations
2. Mount Google Drive — all outputs saved here permanently
3. Define ZIP filenames — exact Roboflow download names
4. Extract ZIPs — unpack COCO segmentation and YOLOv5 datasets
5. Parse COCO annotations — read polygon outlines from JSON files
6. Exploratory data analysis — class distribution, image size charts, annotation counts
7. Preprocessing — letterbox resize to 640x640, normalise 0-255 to 0-1
8. Augmentation — flip, rotate, brightness, noise applied 3x to grow training set
9. Train/val/test split — 75/15/10 stratified by class
10. Save metadata.json and artifacts_data.yaml to Drive for downstream use

**Segmentation model steps:**
1. Build custom PyTorch Dataset — reads image + converts COCO polygons to pixel masks
2. Create DataLoaders — batches of 4 images, shuffle train, fixed val/test
3. Build U-Net — ResNet-34 encoder (ImageNet pretrained) + decoder with skip connections
4. Define loss — CrossEntropy (50%) + Dice (50%) combined
5. Train loop — forward pass, backprop, AdamW update, early stopping on val IoU
6. Load best checkpoint — PyTorch 2.6 compatible loading with safe globals fix
7. Evaluate test set — compute mean IoU and Dice Score
8. Visualise — side-by-side input / ground truth / prediction images

**Detection model steps:**
1. Verify data.yaml — YOLO config pointing to split/detection folders
2. Load YOLOv5s pretrained — transfer learning from COCO 80-class weights
3. Train YOLO — .train() call with AdamW, cosine LR, early stopping (patience=10)
4. Evaluate — mAP@0.5, Precision, Recall per class
5. Visualise — draw bounding boxes with confidence scores on sample images
6. Save backup — best_seg_model.pth and best.pt copied to Drive

**Erosion prediction steps:**
1. Collect all images from M1 split folders
2. Extract 13 terrain features per image — slope, NDVI proxy, texture, vegetation cover etc.
3. Generate erosion labels — domain-rule weighted formula produces score 0-1
4. EDA — class balance, feature correlations, scatter plots
5. Split and scale — stratified 75/15/10, StandardScaler fit on train only
6. Train Random Forest — 200 trees, OOB validation, feature importance
7. Train XGBoost — 300 rounds, early stopping on val RMSE, SHAP explainability
8. Compare models — RMSE, MAE, R2 side by side
9. Generate colour mosaic — patches tinted green/amber/red by risk score
10. Build Folium map — interactive HTML with clickable markers per patch
11. Save all pkl files to Drive

**Dashboard steps:**
1. Restore all M2 and M3 outputs from Drive to Colab runtime
2. Load all 3 models — U-Net, YOLO, XGBoost
3. Print full performance report — all scores from all milestones
4. Define predict_all() — runs all 3 models on one image in sequence
5. Test inference on sample images — visual proof all 3 predictions work
6. Write dashboard.py — complete Streamlit application code
7. Launch via ngrok — get public URL for browser access

---

## Algorithms Used and Why

| Algorithm | Task | Why chosen |
|-----------|------|-----------|
| U-Net | Pixel segmentation | Designed for small medical/satellite datasets. Encoder-decoder with skip connections preserves fine spatial detail. Works well with limited annotated data. |
| DeepLabV3+ | Pixel segmentation (alt) | Better for multi-scale features in complex terrain. Atrous convolutions capture wider context without losing resolution. |
| ResNet-34 | Encoder backbone | ImageNet pretrained — 1.2M image features available from day one. Transfer learning reduces training data and time needed dramatically. |
| YOLOv5 | Artifact detection | Single-pass detection processes the full image in one forward pass. Fast and accurate. Pretrained on COCO 80 classes — artifact fine-tuning needs far fewer epochs. |
| Random Forest | Erosion prediction | Builds 200 independent trees. Each votes on the score. Robust to noisy terrain features. Built-in OOB validation. Easy feature importance ranking. |
| XGBoost | Erosion prediction | Sequential boosting — each tree corrects previous errors. Generally outperforms RF on structured tabular data. Early stopping prevents overfitting automatically. |
| SHAP | Explainability | Shows exactly why each erosion prediction was made — which features pushed the score up or down. Critical for archaeologist trust in AI decisions. |
| Sobel filter | Feature extraction | Computes pixel gradient magnitude = slope steepness proxy. Fast, no model needed, based on image intensity differences. |
| NDVI proxy | Feature extraction | (Green-Red)/(Green+Red) measures vegetation density. Adapted from satellite remote sensing for standard RGB cameras. |
| Streamlit | Dashboard | Python-native web framework. No HTML/CSS/JS needed. Upload widget, columns, metrics, progress bars all built in. Easy to deploy on any machine. |
| Folium | Geographic map | Python wrapper for Leaflet.js. Produces interactive HTML maps with clickable markers, popups, and legends. Works inside both Colab and Streamlit. |
| Albumentations | Augmentation | Fastest augmentation library for Python. Handles image+mask synchronised transforms so segmentation polygons stay aligned after flips and rotations. |

---

## Evaluation Scores

### Segmentation (U-Net)

| Metric | What it measures | Formula | Target | Direction |
|--------|-----------------|---------|--------|-----------|
| IoU (Intersection over Union) | Overlap between predicted mask and real mask | Intersection area / Union area | > 0.70 | Higher is better |
| Dice Score | F1 equivalent for pixel masks | 2 x Overlap / (Predicted + Actual) | > 0.75 | Higher is better |

IoU of 0 means no overlap at all. IoU of 1 means the prediction exactly matches the ground truth. Dice is always slightly higher than IoU for the same prediction — both are reported together to confirm segmentation quality.

### Detection (YOLOv5)

| Metric | What it measures | Target | Direction |
|--------|-----------------|--------|-----------|
| mAP@0.5 | Mean average precision at 50% box overlap threshold | > 0.50 | Higher is better |
| Precision | Of all predicted boxes, what fraction were correct | > 0.70 | Higher is better |
| Recall | Of all real artifacts, what fraction did the model find | > 0.70 | Higher is better |

Precision and Recall trade off against each other. High precision = model is careful (few false positives). High recall = model finds everything (few missed artifacts). mAP balances both across all confidence thresholds.

### Erosion Prediction (XGBoost + Random Forest)

| Metric | What it measures | Formula | Target | Direction |
|--------|-----------------|---------|--------|-----------|
| RMSE (Root Mean Square Error) | Average prediction error, penalises large errors more | sqrt(mean((predicted - actual)^2)) | < 0.10 | Lower is better |
| R2 Score | What percentage of erosion score variance the model explains | 1 - (SS_res / SS_tot) | > 0.75 | Higher is better |
| MAE (Mean Absolute Error) | Simple average of all prediction errors | mean(abs(predicted - actual)) | < 0.08 | Lower is better |

R2 of 1.0 = perfect model. R2 of 0.0 = model is no better than predicting the mean every time. R2 > 0.75 means the model explains 75% or more of the variation in erosion risk.

---

## Tech Stack

```
Language        Python 3.10+

Deep Learning   PyTorch 2.0+
                segmentation-models-pytorch  (U-Net, DeepLabV3+)
                Ultralytics                  (YOLOv5)

Machine Learning scikit-learn               (Random Forest, preprocessing, metrics)
                 XGBoost                    (gradient boosted trees)
                 SHAP                       (model explainability)

Computer Vision  OpenCV                     (image processing, feature extraction)
                 Albumentations             (augmentation pipeline)
                 pycocotools                (COCO annotation parsing)

Geospatial       Folium                     (interactive HTML maps)
                 GeoPandas                  (spatial data - optional)
                 Rasterio                   (GeoTIFF support - optional)

Dashboard        Streamlit                  (web application)
                 streamlit-folium           (embed Folium in Streamlit)
                 pyngrok                    (public URL from Colab)

Data & Viz       Pandas, NumPy
                 Matplotlib, Seaborn
```

---

## Dataset Sources

| Dataset | Source | Format | Classes |
|---------|--------|--------|---------|
| vegetation_data set.v3i.coco-segmentation.zip | Roboflow | COCO JSON polygons | ruins-vegetation, vegetation |
| Artifacts Dataset.v2i.yolov5pytorch.zip | Roboflow | YOLOv5 .txt bounding boxes | artifact |
| Supplementary aerial imagery | OpenAerialMap | GeoTIFF / JPEG | — |
| Supplementary satellite imagery | Google Earth Pro | JPEG | — |

Annotation schema defined before data collection:
- Segmentation classes — ruins-vegetation (mixed ruin and vegetation areas), vegetation (pure vegetation), erosion zones
- Detection class — artifact (any structural artifact visible in aerial view)

---

## How to Run

### Requirements

- Google account with Google Drive (minimum 5 GB free space)
- Google Colab with GPU runtime (T4 or better — free tier works)
- Roboflow account (free) — to download the datasets
- ngrok account (free at ngrok.com) — to get a public URL for the dashboard

### Step 1 — Clone this repository

```bash
git clone https://github.com/your-username/archaeological-site-mapping.git
cd archaeological-site-mapping
```

### Step 2 — Upload datasets to Google Drive

1. Download both ZIP files from your Roboflow workspace
2. Go to drive.google.com
3. Drag both ZIPs directly into My Drive root (not inside any folder)

```
My Drive/
├── vegetation_data set.v3i.coco-segmentation.zip
└── Artifacts Dataset.v2i.yolov5pytorch.zip
```

### Step 3 — Run notebooks in order

Open each notebook in Google Colab. Set Runtime → Change runtime type → GPU (T4).

```
Milestone1_Complete.ipynb
  └── Run all cells
  └── Last cell must print all ✅ (Drive backup complete)
  └── Do not proceed until last cell shows success

Milestone2_Complete.ipynb
  └── Run all cells
  └── Last cell must print:
        ✅ best_seg_model.pth → Drive
        ✅ YOLO weights → Drive

Milestone3_Complete.ipynb
  └── Run all cells
  └── Last cell must print:
        ✅ random_forest.pkl → Drive
        ✅ xgboost.pkl → Drive
        ✅ scaler.pkl → Drive
        ✅ imputer.pkl → Drive

Milestone4_Dashboard.ipynb
  └── Run all cells
  └── Step 2 must show all ✅ (models found)
  └── Step 8 prints public ngrok URL
  └── Open URL in any browser
```

### Step 4 — Use the dashboard

Once the ngrok URL appears:
- Open it in any browser
- Upload a satellite or drone image (JPG or PNG)
- Adjust confidence threshold with the sidebar slider
- View segmentation mask, artifact boxes, and erosion score
- Explore the interactive Folium map

---

## Project Structure

```
archaeological-site-mapping/
│
├── notebooks/
│   ├── Milestone1_Complete.ipynb       Data collection and preparation
│   ├── Milestone2_Complete.ipynb       Segmentation and detection training
│   ├── Milestone3_Complete.ipynb       Erosion prediction
│   └── Milestone4_Dashboard.ipynb     Dashboard and final reporting
│
├── README.md                           This file
│
└── Google Drive outputs (auto-created during runtime)
    │
    ├── MyDrive/Archaeological_AI/project/
    │   ├── split/segmentation/train/   Prepared segmentation images
    │   ├── split/segmentation/val/
    │   ├── split/segmentation/test/
    │   ├── split/detection/train/      Prepared detection images
    │   ├── split/detection/val/
    │   ├── split/detection/test/
    │   ├── processed/
    │   │   ├── milestone1_metadata.json
    │   │   └── artifacts_data.yaml
    │   └── augmented/segmentation/     3x augmented training images
    │
    ├── MyDrive/Archaeological_AI/Milestone2_Output/
    │   ├── best_seg_model.pth          Trained U-Net weights
    │   ├── yolo_artifacts/
    │   │   └── weights/
    │   │       └── best.pt             Trained YOLOv5 weights
    │   └── milestone2_metrics.json     IoU, Dice, mAP, P, R scores
    │
    └── MyDrive/Archaeological_AI/Milestone3_Output/
        ├── random_forest.pkl           Trained Random Forest model
        ├── xgboost.pkl                 Trained XGBoost model
        ├── scaler.pkl                  Feature StandardScaler
        ├── imputer.pkl                 Missing value handler
        ├── erosion_risk_map.html       Interactive Folium map
        └── milestone3_metrics.json     RMSE, R2, MAE scores
```

---

## Dashboard Preview

The Streamlit dashboard provides three columns:

```
+-------------------+-------------------+-------------------+
|   Input image     | Segmentation mask | Artifact detection|
|                   |                   |                   |
|  [uploaded photo] | [colour-coded     | [bounding boxes   |
|                   |  pixel mask]      |  with labels]     |
|                   |                   |                   |
|                   | Legend:           | Artifacts found: N|
|                   |  Black=background |                   |
|                   |  Red=ruins        |                   |
|                   |  Green=vegetation |                   |
|                   |  Amber=erosion    |                   |
+-------------------+-------------------+-------------------+

+-------------------------------------------------+
|  Terrain Erosion Risk                           |
|  Score: 0.72   Risk class: High                 |
|  [============================--------] 72%     |
+-------------------------------------------------+

+-------------------------------------------------+
|  Geographic Map (Folium interactive)            |
|  Coloured markers: green=low  amber=medium      |
|                    red=high risk                |
+-------------------------------------------------+
```

---

## Results

Results are populated after running all four milestones. Update this section with your actual scores.

### Segmentation — U-Net + ResNet-34

| Metric | Score |
|--------|-------|
| Test IoU | — |
| Test Dice Score | — |

### Detection — YOLOv5s

| Metric | Score |
|--------|-------|
| mAP@0.5 | — |
| Precision | — |
| Recall | — |

### Erosion Prediction — XGBoost

| Metric | Score |
|--------|-------|
| Test RMSE | — |
| Test R2 | — |
| Test MAE | — |

---

## Acknowledgements

- [Roboflow](https://roboflow.com) — dataset annotation platform and hosting
- [OpenAerialMap](https://openaerialmap.org) — open aerial and satellite imagery
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) — U-Net and DeepLabV3+ implementations
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv5 implementation and training pipeline
- [SHAP](https://github.com/slundberg/shap) — model explainability library
- [Folium](https://python-visualization.github.io/folium/) — Python Leaflet.js wrapper for interactive maps
- [Streamlit](https://streamlit.io) — Python web application framework

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

---

*B.E. Artificial Intelligence & Machine Learning — Capstone Project*
*SSCE Bangalore*
