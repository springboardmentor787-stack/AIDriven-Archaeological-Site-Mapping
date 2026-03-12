# Archaeological Site Mapping AI V2

Archaeological Site Mapping AI V2 is a computer vision project for analyzing archaeological landscapes from aerial or satellite-style imagery. It combines object detection and semantic segmentation to identify classes such as vegetation, ruins, structures, boulders, and other scene elements relevant to site mapping.

This repository is published as a public, code-first version of the project. Large datasets, trained weights, and generated experiment outputs are intentionally excluded from Git.

## Features

- YOLO-based object detection training and inference
- DeepLabV3+ semantic segmentation training and inference
- COCO-to-mask conversion utilities for segmentation datasets
- Streamlit interface for interactive image upload and visualization
- Terrain-feature extraction experiment for erosion-risk style analysis

## Tech Stack

### Core

- Python
- PyTorch
- NumPy

### Computer Vision

- Ultralytics YOLO
- segmentation-models-pytorch
- OpenCV
- Pillow
- pycocotools
- TorchMetrics

### App and Analysis

- Streamlit
- pandas

## Public Repository Scope

Included in this repository:

- training scripts
- inference scripts
- Streamlit app code
- utility scripts
- dataset config files
- documentation

Excluded from this repository:

- `dataset/`
- `seg_dataset/`
- `runs/`
- `*.pt`
- `*.pth`

To run training or inference end to end, you must restore the expected datasets and model weights locally or retrain the models.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Project Workflow

1. Prepare the detection and segmentation datasets.
2. Generate segmentation masks from COCO annotations if needed.
3. Train the YOLO detection model.
4. Train the DeepLab segmentation model.
5. Run local inference or launch the Streamlit app.

## Usage

### Generate Segmentation Masks

Use this when segmentation masks need to be created from COCO annotations:

```bash
python generate_masks.py
```

### Train the Detection Model

```bash
python train_seg.py
```

Notes:

- trains a YOLOv8s detector
- uses `data.yaml` for dataset configuration
- writes outputs under `runs/detect/`

### Train the Segmentation Model

```bash
python train_deeplab_seg.py
```

Notes:

- trains DeepLabV3+ with a ResNet34 encoder
- expects images and masks under `seg_dataset/`
- saves weights to `deeplab_model.pth`

### Run Detection Inference

```bash
python predict.py
```

Expected local asset:

- `runs/detect/yolov8s_archaeology2/weights/best.pt`

### Run the Combined Demo Pipeline

```bash
python demo_pipeline.py
```

Expected local assets:

- `runs/detect/yolov8s_archaeology2/weights/best.pt`
- `deeplab_model.pth`

### Launch the Streamlit App

```bash
streamlit run app.py
```

The app provides:

- uploaded image preview
- YOLO detection output
- segmentation overlay
- combined archaeological mapping view
- class visibility toggles
- confidence threshold control

### Extract Terrain Features

```bash
python terrain_model/extract_terrain_features.py
```

This script derives a small terrain-style feature table using vegetation ratio, simulated slope, simulated elevation, and erosion-risk labels.

## Key Files

- `app.py`: Streamlit application for interactive inference
- `train_seg.py`: YOLO detection training
- `train_deeplab_seg.py`: DeepLab segmentation training and evaluation
- `demo_pipeline.py`: local combined detection and segmentation demo
- `predict.py`: single-image YOLO inference
- `generate_masks.py`: COCO-to-mask conversion for segmentation data
- `dataset_loader.py`: segmentation dataset loader
- `metrics.py`: IoU and Dice metric setup
- `terrain_model/extract_terrain_features.py`: terrain-feature extraction experiment

## Dataset Summary

The original local workspace used aligned detection and segmentation splits.

### Split Sizes

| Split | Images | Labels | Masks |
| --- | ---: | ---: | ---: |
| Train | 648 | 648 | 648 |
| Validation | 61 | 61 | 61 |
| Test | 31 | 31 | 31 |
| Total | 740 | 740 | 740 |

### Detection Classes

Defined in `data.yaml`:

- `0`: boulders
- `1`: others
- `2`: ruins
- `3`: structures
- `4`: vegetation

### Segmentation Classes

Used by the DeepLab model configuration:

- `0`: background
- `1`: boulders
- `2`: others
- `3`: ruins
- `4`: structures
- `5`: vegetation

### Detection Object Distribution

| Class | Count |
| --- | ---: |
| Vegetation | 10,620 |
| Boulders | 3,876 |
| Ruins | 2,386 |
| Others | 1,346 |
| Structures | 230 |
| Total | 18,458 |

The dataset is imbalanced, with vegetation strongly overrepresented and structures relatively scarce.

### Site Distribution by Filename Prefix

| Site | Images |
| --- | ---: |
| Hampi | 255 |
| Khajuraho | 238 |
| Pattadakal | 130 |
| Nalanda | 117 |

## Model Configuration Summary

### Detection

Saved local training configuration showed:

- model: `yolov8s.pt`
- task: detection
- epochs: 80
- image size: 640
- batch size: 8
- workers: 8

### Segmentation

The training script configures:

- architecture: DeepLabV3+
- encoder: ResNet34
- encoder weights: ImageNet
- input resolution: `512 x 512`
- batch size: 4
- optimizer: Adam
- learning rate: `1e-4`
- loss: CrossEntropyLoss
- metrics: IoU and Dice

## Recorded Detection Results

From the saved local YOLO training run:

- best `mAP50-95`: `0.58673`
- best `mAP50`: `0.83206`
- final precision: `0.89978`
- final recall: `0.72010`
- final `mAP50`: `0.83206`
- final `mAP50-95`: `0.58611`





