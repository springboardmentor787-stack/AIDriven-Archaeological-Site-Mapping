# AIDriven-Archaeological-Site-Mapping

## Problem Statement

In Archaeological surveys over large regions are slow, expensive, and difficult to scale manually. So to overcome for this statement:
  * AI-driven Archaeological Site Mapping Project uses advanced techniques like deep learning and computer vision to automatically detect, segment, and analyze archaeological features from satellite or aerial imagery. 
  * By combining models such as U-Net for image segmentation and object detection algorithms, it helps identify hidden structures and terrain patterns with high accuracy through IoU and Dice Score. 
  * The system further evaluates performance using metrics like RMSE and R² score, making the mapping process faster, more precise, and less dependent on manual effort.
  * Then after predict the risk of the erosion prone and stable areas. 
Future enhancements may include real-time analysis, integration with geospatial data, improved model accuracy, and expansion to detect more complex archaeological patterns using advanced AI techniques.

## System Architecture

 ```text
  Input Image
       |
       v
Annotate / Preprocessed Image
       |
       v  
 U-Net Segmentation
       |
       v
 YOLOv8 Object Detection
       |
       v 
 Terrain Erosion Prediction
       |
       v  
 Combined Feature Mapping
       |
       v  
 Visualization (via Streamlit App)
 ```     

## Key Features
* Satellite/Drone imagery preprocessing.
* Semantic segmentation (ruins vs vegetation vs bounding box).
* Artifact detection and classification.
* Annotation to Segmentation mask conversion
* Terrain erosion prediction.
* Streamlit Dashboard visualization for archaeologists.
  
##  Project Workflow
* Acquire and annotate the imagery.
* Preprocess and split the dataset.
* Generate the segmentation masks from the JSON annotations.
* Train U-net segmentation and YOLOv8 based Object detection models.
* Predict erosion zones using XGBoost terrain features.
* Overlay and visualize results through a Streamlit Dashboard.

## Dataset Sources
* Google Earth Pro
* Open AerialMap
* Custom Annotated images (via Labelbox/QGIS)
* Images Segmentation (via U-Net/DeepLabv3+)
* Object Detection (via YOLOv8)
* Predicts Erosion Risk and Terrain analysis (via XGBoost)
* Using SHAP for AI explanation
* Streamlit App (For Visualization)

## Tech Stack
* Language - Python
* Libraries - Pandas, Numpy, OpenCV, Ultralytics YOLOv8, XGBoost, scikit-learn, GeoPandas, Matplotlib
* Frameworks - PyTorch/TensorFlow
* Interfaces and Data Processing - Streamlit

## Project Objectives
The overall objective of this project is to:
* To collect and preprocess satellite/aerial image data for analysis.
* Apply Deep learning models like U-Net for Image segmentation and YOLOv8 for Object detection to identify archaeological features.
* To extract terrain and structural patterns from the images using AI techniques.
* Evaluate model performance using metrics such as RMSE and R² score.
* To improve the accuracy and efficiency of detecting hidden archaeological sites.
* Build an interactive Streamlit interface for easy visualization and user interaction.
* To create a scalable system that can be extended for future improvements and advanced archaeological analysis.

## Installation

### 1. Clone and install

```bash
!git clone https://github.com/your-username/AIDriven-Archaeological-Site-Mapping.git
%cd AIDriven-Archaeological-Site-Mapping
!pip install streamlit pyngrok
pip install -r requirements.txt
!pip install ultralytics
!pip install matplotlib
!pip install scikit-learn
!pip install opencv-python
```
### 2. Running the project

```bash
!ngrok config add-authtoken YOUR_AUTH_TOKEN
!streamlit run app.py
public_url = ngrok.connect(8501)
print("Streamlit App URL:", public_url)
```
The dashboard will automatically opens in the browser.

## Demo Usage Flow

- **User Input Layer:** Streamlit interface through which users upload satellite/drone images and set parameters.  

- **Preprocessing:** Images are processed using OpenCV and converted for analysis (RGB + grayscale).  

- **Object Detection (YOLOv8):** Detects archaeological structures and draws bounding boxes on images.  

- **Vegetation Segmentation:** Uses green channel thresholding to separate vegetation and non-vegetation areas.  

- **Feature Extraction:** Extracts key features like vegetation %, texture, brightness, edge density, and contrast.  

- **Risk Prediction:** Calculates erosion risk score using a weighted feature-based model.  

- **Explainability (SHAP):** Explains how each feature contributes to the risk prediction.  

- **Visualization:** Displays results using Streamlit and Plotly (charts, maps, heatmaps).  

- **Geospatial Mapping:** Plots risk zones on interactive maps using latitude and longitude.  

- **Report Generation:** Generates downloadable CSV and PDF reports of analysis results.

## Project Strength
* This project integrates Computer Vision, Deep Learning, Explainable AI, and Geospatial Visualization for archaeological intelligence.
* This project also demonstrates how Artificial Intelligence can assist in cultural heritage preservation by automating archaeological site detection and risk    analysis from satellite imagery.


## Future Scope
### 1. Preserving Cultural Heritage
   * Archaeological sites are fragile and can be damaged by natural erosion, urbanization, or climate change.
   * Using AI to map and monitor these sites ensures digital preservation before they are lost.
### 2. Faster & More Accurate Discoveries
   * Traditionally, surveying sites is time-consuming and labor-intensive.
   * AI can analyze satellite images, drones, and other data quickly, helping archaeologists spot potential dig sites they might miss     manually.
### 3. Support for Conservation Policies
   * Governments and organizations can use AI-generated maps to prioritize site protection, plan urban expansion, or prevent illegal excavations.

## Author
PALAK AGRAWAL
