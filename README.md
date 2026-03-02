# AIDriven-Archaeological-Site-Mapping
Project: AI-Driven Archaeological Site Mapping
Milestone 1: Dataset Collection & Preparation (Weeks 1–2)
1. Data Collection
Objective:

Gather high-resolution satellite and drone imagery for archaeological analysis.

Sources Used:

Google Earth Pro

OpenAerialMap

Custom drone imagery (if available)

Process:

Identified archaeological regions of interest.

Downloaded high-resolution imagery.

Ensured:

Clear terrain visibility

Minimal cloud cover

Consistent resolution

Organized images into structured folders.

Example structure:

Dataset/
    Ruins/
    Vegetation/
    Artifacts/
    Background/

Outcome:
✔ Collected structured raw imagery dataset.

2. Annotation Schema Definition
Objective:

Define how images will be labeled.

Defined Classes:
Category	Description
Ruins	Ancient walls, foundations, structural remains
Vegetation	Grass, trees, shrubs covering site
Artifacts	Visible structures, pillars, carved remains
Background	Non-relevant terrain

For segmentation:

Pixel-level mask labeling (ruins vs vegetation vs soil)

For detection:

Bounding boxes around artifacts

Outcome:
✔ Clear labeling strategy for classification, segmentation, and detection tasks.

3. Image Annotation
Tools Used:

QGIS (for spatial annotation)

Labelbox (for bounding box labeling)

Folder-based labeling (for classification task)

What Was Done:

✔ Classification labeling:
Images organized into folders representing classes.

✔ Segmentation labeling:
Binary or multi-class masks created (if required).

✔ Object detection:
Bounding boxes defined around artifact structures.

Outcome:
✔ Images paired with labels / masks / annotations.

4. Dataset Preparation (What You Implemented in Code)

You performed:

✔ Automatic label extraction from folder names
✔ Created annotation DataFrame
✔ Train-Test split (80-20 stratified)
✔ Image resizing & normalization
✔ Generator creation for deep learning

This prepares dataset for:

CNN classification

Future segmentation models

Object detection models

5. Preprocessing Steps Applied

Image resizing (64x64 / 128x128)

Pixel normalization (rescale 1./255)

Train-test split

Class encoding (automatic via Keras)
