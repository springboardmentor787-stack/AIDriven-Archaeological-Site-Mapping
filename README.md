# AI-Driven Archaeological Site Mapping

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![ComputerVision](https://img.shields.io/badge/Task-ComputerVision-green)

AI-Driven Archaeological Site Mapping is a computer vision-based system designed to analyze archaeological landscapes from aerial or satellite imagery. The project combines object detection and semantic segmentation to automatically identify terrain features such as ruins, vegetation, structures, and boulders.

The system aims to assist researchers and archaeologists by automating site mapping and landscape analysis, enabling faster discovery and interpretation of potential archaeological regions.

This repository represents the public development version of the project. It includes datasets, training scripts, inference pipelines, and a Streamlit-based visualization interface.

## System Architecture

```text
Input Image
	|
	v
YOLOv8 Object Detection
	|
	v
DeepLabV3+ Segmentation
	|
	v
Combined Feature Mapping
	|
	v
Visualization (Streamlit App)
```

## Project Structure

```text
archaeological-site-mapping
|
|-- app.py                         # Streamlit interface
|-- demo_pipeline.py               # Combined inference pipeline
|-- predict.py                     # YOLO detection inference
|-- train_seg.py                   # YOLO training script
|-- train_deeplab_seg.py           # DeepLab segmentation training
|
|-- seg_dataset/                   # Segmentation dataset
|-- runs/                          # YOLO training outputs
|
|-- terrain_model/
|   |-- extract_terrain_features.py
|
|-- docs/
|   |-- screenshots/
|
|-- requirements.txt
`-- README.md
```

## Quick Demo

Run the combined detection and segmentation pipeline:

```bash
python demo_pipeline.py
```

Launch the interactive interface:

```bash
streamlit run app.py
```

Upload an image and visualize:

- detected ruins
- segmented landscape
- combined archaeological mapping

## Project Objectives

The goal of this project is to:

- Detect archaeological structures from satellite imagery
- Segment landscape elements for better terrain understanding
- Assist archaeologists in site discovery and mapping
- Provide an interactive visualization interface
- Experiment with terrain-based feature analysis for potential risk detection

## Features

- YOLO-based object detection
- DeepLabV3+ semantic segmentation
- COCO annotation to segmentation mask conversion
- Interactive Streamlit web interface
- Combined detection + segmentation inference pipeline
- Terrain feature extraction experiment
- Visualization of archaeological mapping outputs

## Tech Stack

### Core Technologies

- Python
- PyTorch
- NumPy

### Computer Vision

- Ultralytics YOLOv8
- segmentation-models-pytorch
- OpenCV
- Pillow
- pycocotools
- TorchMetrics

### Data Processing and Interface

- Streamlit
- pandas

## Milestones

### Milestone 1 - Dataset Collection & Preparation

- Collected archaeological aerial imagery datasets
- Organized detection and segmentation datasets
- Configured dataset splits for training, validation, and testing
- Prepared annotation formats for YOLO and segmentation models

### Milestone 2 - Model Development & Pipeline

- Implemented YOLOv8 object detection pipeline
- Developed DeepLabV3+ segmentation model
- Created dataset loaders and preprocessing utilities
- Implemented inference pipeline combining detection and segmentation
- Built Streamlit interface for visualization
- Performed initial experiments on terrain feature extraction

### Upcoming Milestone (MS3) - Advanced Analysis & Improvements

Planned work includes:

- Improve model performance and training strategies
- Add advanced visualization layers to archaeological mapping
- Implement automated site detection analysis
- Optimize dataset balancing techniques
- Explore integration with geospatial mapping tools
- Improve UI/UX for the Streamlit application

## Public Repository Scope

Included in this repository:

- Training scripts
- Inference scripts
- Streamlit application
- Dataset configuration files
- Utility scripts
- Documentation
- Example datasets
- Core inference weights

Excluded from this repository:

- runs/
- large experimental outputs
- intermediate model checkpoints

Included inference weights:

- deeplab_model.pth
- runs/detect/yolov8s_archaeology2/weights/best.pt

These allow users to run inference without retraining the models.

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd archaeological-site-mapping
pip install -r requirements.txt
```

## Project Workflow

1. Prepare detection and segmentation datasets
2. Generate segmentation masks from COCO annotations
3. Train the YOLO detection model
4. Train the DeepLab segmentation model
5. Run inference pipeline
6. Visualize results through Streamlit interface

## Usage

### Generate Segmentation Masks

```bash
python generate_masks.py
```

Used to convert COCO annotations into segmentation masks.

### Train Detection Model

```bash
python train_seg.py
```

Notes:

- trains YOLOv8s detection model
- uses dataset configuration from data.yaml
- outputs saved to runs/detect/

### Train Segmentation Model

```bash
python train_deeplab_seg.py
```

Notes:

- trains DeepLabV3+ with ResNet34 encoder
- expects segmentation dataset in seg_dataset/
- saves weights to deeplab_model.pth

### Run Detection Inference

```bash
python predict.py
```

Uses trained YOLO weights:

- runs/detect/yolov8s_archaeology2/weights/best.pt

### Run Combined Pipeline

```bash
python demo_pipeline.py
```

Pipeline performs:

- object detection
- semantic segmentation
- combined archaeological visualization

### Launch Streamlit Application

```bash
streamlit run app.py
```

The interface supports:

- image upload
- YOLO detection visualization
- segmentation overlays
- combined archaeological mapping view
- class filtering
- confidence threshold control

### Extract Terrain Features

```bash
python terrain_model/extract_terrain_features.py
```

Generates terrain feature tables including:

- vegetation ratio
- simulated slope
- simulated elevation
- erosion-risk classification

## Screenshots

### Hampi Example

### Khajuraho Example

### Nalanda Example

## Key Files

| File | Description |
| --- | --- |
| app.py | Streamlit application |
| train_seg.py | YOLO detection training |
| train_deeplab_seg.py | DeepLab segmentation training |
| demo_pipeline.py | combined inference pipeline |
| predict.py | YOLO single image inference |
| generate_masks.py | COCO annotation to mask converter |
| dataset_loader.py | segmentation dataset loader |
| metrics.py | IoU and Dice metric computation |
| terrain_model/extract_terrain_features.py | terrain feature experiment |

## Dataset Summary

Total dataset size: 740 images

| Split | Images | Labels | Masks |
| --- | ---: | ---: | ---: |
| Train | 648 | 648 | 648 |
| Validation | 61 | 61 | 61 |
| Test | 31 | 31 | 31 |

## Detection Classes

Defined in data.yaml:

- 0: boulders
- 1: others
- 2: ruins
- 3: structures
- 4: vegetation

## Segmentation Classes

Used in DeepLab model:

- 0: background
- 1: boulders
- 2: others
- 3: ruins
- 4: structures
- 5: vegetation

## Detection Dataset Distribution

| Class | Instances |
| --- | ---: |
| Vegetation | 10620 |
| Boulders | 3876 |
| Ruins | 2386 |
| Others | 1346 |
| Structures | 230 |
| Total | 18458 |

The dataset shows class imbalance, with vegetation dominating the dataset.

## Dataset Locations

| Site | Images |
| --- | ---: |
| Hampi | 255 |
| Khajuraho | 238 |
| Pattadakal | 130 |
| Nalanda | 117 |

## Model Configuration

### Detection Model

- model: YOLOv8s
- epochs: 80
- image size: 640
- batch size: 8
- workers: 8

### Segmentation Model

- architecture: DeepLabV3+
- encoder: ResNet34
- encoder weights: ImageNet
- input resolution: 512x512
- optimizer: Adam
- learning rate: 1e-4
- loss: CrossEntropyLoss
- metrics: IoU, Dice

## Detection Results

Best metrics from YOLO training:

| Metric | Value |
| --- | ---: |
| mAP50-95 | 0.58673 |
| mAP50 | 0.83206 |
| Precision | 0.89978 |
| Recall | 0.72010 |

## Future Work

Future improvements may include:

- improving detection accuracy
- balancing dataset classes
- integrating geospatial mapping tools
- adding GIS-based visualization
- building a web deployment version
- extending terrain analysis models

## Important Notes

- The repository includes the two trained weights needed by the main app and demo pipeline, but other model files and run outputs remain excluded.
- `train_seg.py` is the YOLO detection training script despite its generic name.
