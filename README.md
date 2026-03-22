# AIDriven-Archaeological-Site-Mapping
This project builds an AI-based platform to analyze satellite and drone imagery for archaeological research. 
It assists archaeologists in:
- Segmenting ancient ruins and vegetation
- Detecting and classifying artifact structures
- Predicting erosion-prone zones

The system supports conservation planning and heritage preservation by combining deep learning models with geospatial analysis and interactive dashboards.

## Features
* Satellite/Drone imagery preprocessing.
* Semantic segmentation (ruins vs vegetation vs bounding box).
* Artifact detection and classification.
* Terrain erosion prediction.
* Streamlit Dashboard visualization for archaeologists.
  
## Workflow
* Acquire and annotate the imagery.
* Preprocess and split dataset.
* Train segmentation and detection models.
* Predict erosion zones using terrain features.
* Overlay and visualize results on a dashboard.

## Dataset Sources
* Google Earth Pro
* Open AerialMap
* Custom Annotated images (via Labelbox/QGIS)
* Images Segmentation (via U-Net/DeepLabv3+)
* Object Detection (via YOLOv8)

## Tech Stack
* Language - Python
* Libraries - Pandas, Numpy, OpenCV, scikit-learn, GeoPandas, Matplotlib
* Frameworks - PyTorch/TensorFlow
