#  AI-Driven Archaeological Site Mapping & Erosion Prediction

## Project Overview

This project presents an AI-based system designed to detect archaeological features such as ruins, vegetation, and artifacts from satellite images and predict terrain erosion risk.

The system combines Computer Vision and Machine Learning techniques to support:

Archaeological site monitoring
Heritage preservation
Terrain risk analysis
* System Functionality

The system performs the following tasks:

Detects archaeological features from satellite images
Segments ruins and vegetation areas
Predicts terrain erosion risk
Visualizes outputs through an interactive dashboard

## Project Pipeline

The project follows a structured pipeline from data collection to final visualization, ensuring a smooth flow of data and model integration.

### End-to-End Pipeline
```
Data Collection & Annotation (Milestone 1)
             ↓
Data Preprocessing & Augmentation
                ↓
Dataset Splitting (Train / Validation / Test)
                ↓
Segmentation Model Training (U-Net) (Milestone 2)
                ↓
Object Detection Model Training (YOLO)
                ↓
Feature Extraction (Slope, Vegetation Index, Elevation)
                ↓
Erosion Prediction Model (Random Forest / XGBoost) (Milestone 3)
                ↓
Model Evaluation (IoU, mAP, RMSE, R²)
                ↓
Integration of All Models
                ↓
Streamlit Dashboard Development (Milestone 4)
                ↓
Final Visualization & User Interaction
                 ↓
 Final Visualization & User Interaction
```

# Step-by-Step Explanation
* Step 1: Data Collection & Annotation
Collect satellite and drone images
Annotate ruins, vegetation, and artifacts
Prepare labeled dataset
* Step 2: Data Preprocessing
Resize and normalize images
Apply augmentation (flip, rotate, brightness)
Split dataset into train, validation, and test sets
* Step 3: Segmentation (U-Net)
Input: Satellite image
Output: Pixel-wise segmentation
Ruins mask
Vegetation mask
* Step 4: Object Detection (YOLO)
Input: Original image
Output: Bounding boxes for:
Ruins
Vegetation
Artifacts
* Step 5: Feature Extraction
Extract terrain-related features:
Slope
Vegetation Index
Elevation
* Step 6: Erosion Prediction
Input: Extracted features
Model: Random Forest / XGBoost
Output:
Stable area
Erosion-prone area
* Step 7: Model Evaluation
Segmentation:
IoU Score
Dice Score
Detection:
mAP, Precision, Recall
Prediction:
RMSE, R² Score
* Step 8: Integration
Combine outputs from:
YOLO (detection)
U-Net (segmentation)
ML model (erosion prediction)
* Step 9: Dashboard Visualization
Upload satellite image
Display:
Detected objects
Segmented regions
Predict erosion risk
Provide interactive interface using Streamlit

# Key Features
> Archaeological ruin detection
> Vegetation detection
> Artifact detection
> Terrain segmentation
> Terrain erosion prediction
> Interactive visualization dashboard
 
##  Technologies Used
# Category	                               Tools Used
# Computer Vision	                         YOLO (Ultralytics)
# Segmentation	                            U-Net
# Machine Learning	                        Random Forest
# Dataset Platform	                        Roboflow
# Visualization	                           Streamlit
# Programming Language	                    Python
-

## Dataset

Satellite and drone images were used to train the models.

Dataset Preparation Steps:
 > Image collection from satellite sources
 > Annotation of ruins, vegetation, and artifacts
 > Data augmentation
 > Train / validation / test split


##  Project Milestones

### Milestone 1
Milestone 1: Dataset Collection & Preparation (Weeks 1–2)

This phase focused on building a high-quality dataset, which is the foundation of the entire system.

* Week 1: Data Acquisition & Planning
Collected satellite and drone imagery from:
Google Earth Pro
OpenAerialMap
Identified regions with visible archaeological patterns such as ruins and vegetation clusters
Designed an annotation schema, defining clear classes:
Ruins
Vegetation
Artifacts
Studied image characteristics like resolution, lighting, and terrain variations
* Week 2: Annotation & Preprocessing
Annotated images using tools such as:
Roboflow / Labelbox
Created bounding boxes for object detection and masks for segmentation
Performed data preprocessing:
Image resizing (uniform dimensions)
Normalization for model compatibility
Applied data augmentation techniques:
Rotation
Flipping
Brightness/contrast adjustment
Split dataset into:
Training set
Validation set
Test set

 Outcome: A clean, labeled, and structured dataset ready for training models.

### Milestone 2
Milestone 2: Segmentation & Object Detection (Weeks 3–4)

This phase focused on extracting meaningful features from images using deep learning.

* Week 3: Semantic Segmentation
Implemented U-Net / DeepLabV3+ for pixel-level classification
Trained the model to differentiate:
Ruins regions
Vegetation areas
- Key Steps:
Prepared segmentation masks from annotated data
Trained model using PyTorch/TensorFlow
Tuned hyperparameters (learning rate, epochs, batch size)
- Evaluation Metrics:
IoU (Intersection over Union) – measures overlap accuracy
Dice Score – evaluates similarity between predicted and actual masks

* Week 4: Object Detection
Implemented YOLOv5 / Faster R-CNN for object detection
Trained the model to detect:
Ruins
Vegetation
Artifacts
- Key Steps:
Converted annotations into YOLO format
Trained detection model using Ultralytics
Performed validation on unseen images
- Evaluation Metrics:
mAP (Mean Average Precision)
Precision – correctness of detections
Recall – completeness of detections

* Outcome: A robust object detection model capable of identifying archaeological features in satellite imagery.

* Outcome: Model capable of accurately segmenting terrain regions.

### Milestone 3
Milestone 3: Terrain Erosion Prediction (Weeks 5–6)

This phase focused on predicting environmental risk using machine learning.

* Week 5: Feature Engineering
Extracted terrain-related features such as:
Slope (steepness of terrain)
Vegetation Index (NDVI)
Elevation
Prepared labeled dataset:
Stable areas
Erosion-prone areas
- Key Steps:
Combined image-derived and external terrain data
Cleaned and normalized feature values
Structured dataset for machine learning models
* Week 6: Model Training & Evaluation
Implemented:
Random Forest Classifier
XGBoost (optional enhancement)
Trained models to classify terrain condition:
Stable
Erosion-prone
- Evaluation Metrics:
RMSE (Root Mean Square Error)
R² Score (model accuracy and fit)
- Example Prediction:
```
Slope: 28  
Vegetation Index: 0.35  
Elevation: 130  
```

Prediction: Erosion-Prone Area

- Outcome: A predictive model that identifies high-risk erosion zones.

### Milestone 4
Milestone 4: Visualization & Final Reporting (Weeks 7–8)

This phase focused on making the system usable and interpretable.

* Week 7: Dashboard Development
Developed an interactive dashboard using Streamlit
- Features Implemented:
Upload satellite images
Display:
Detection results
Segmentation outputs
Input terrain parameters manually
Predict erosion risk dynamically
Integrated:
YOLO detection outputs
U-Net segmentation results
ML prediction results
* Week 8: Integration & Documentation
Combined all modules into a unified system
Tested end-to-end workflow:
Input image → Output prediction
- Deliverables:
Final project report
Model performance evaluation
Visual outputs and screenshots
Project demonstration

- Outcome: A complete AI system with an interactive interface for real-world use.

---

## Installation
```
git clone https://github.com/krishnan4/archaeological-site-Mapping.git
cd archaeological-site-detection
pip install ultralytics
pip install segmentation-models-pytorch
pip install streamlit
pip install opencv-python
pip install scikit-learn
pip install matplotlib
```
##  Running the Project
```
streamlit run app.py
```

The dashboard will open automatically in your browser.

##  Example Outputs

Object Detection:
Ruins detected
Vegetation detected
Artifacts detected
Segmentation:
Ruins segmentation mask
Vegetation segmentation mask

Erosion Prediction:
Slope: 28
Vegetation Index: 0.35
Elevation: 130

Prediction: Erosion Prone Area

##  Applications
This system can be used in:
Archaeological site monitoring
Heritage conservation
Remote sensing analysis
Terrain risk assessment

# Future Improvements
Use higher-resolution satellite datasets
Improve segmentation accuracy
Integrate GIS mapping
Enable real-time satellite monitoring

# Author
Developed as part of an AI/ML milestone-based project on archaeological site detection and terrain analysis.
