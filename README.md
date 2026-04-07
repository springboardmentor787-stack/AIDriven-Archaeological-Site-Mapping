# ArchAI — Archaeological Intelligence Platform

<div align="center">

![Screenshot_7-4-2026_221226_](https://github.com/user-attachments/assets/e15fa236-4008-45ae-a5b1-d97a07690bb6)


> **AI-powered satellite & drone image analysis for archaeological survey, mound detection, and hidden ruins discovery.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF?style=flat-square)](https://ultralytics.com)
[![Groq LLaMA](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F97316?style=flat-square)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-4.2-gold?style=flat-square)]()

</div>

---

## Overview

**ArchAI** is an end-to-end archaeological intelligence platform that brings AI, computer vision, and geospatial analysis into a single Streamlit dashboard. Upload satellite or drone imagery and get instant insights on artifact detection, vegetation analysis, erosion risk, digital deforestation, and potential hidden ruins — all powered by YOLOv11, VARI indexing, and Groq's LLaMA 3.3 70B for structured AI field reports.

---

## Features

| Module | Description |
|--------|-------------|
|  **Artifact Detection** | YOLOv11 inference on satellite / drone imagery with confidence filtering |
|  **VARI Vegetation Index** | Visible Atmospherically Resistant Index — RGB proxy for NDVI |
|  **Erosion Risk Assessment** | Composite score from slope, elevation, and vegetation; ML model or formula fallback |
|  **Object Detection & Classification** | Detects ALL objects — classifies each as Man-made, Natural, or Uncertain |
|  **Digital Deforestation AI** | Digitally removes vegetation and reveals ground anomalies + buried structure hotspots |
|  **AI Field Reports** | Groq LLaMA 3.3 70B generates structured 4–5 line archaeological assessments |
|  **Interactive Map** | Google Satellite basemap via Folium with artifact and object overlays |
|  **Export** | KMZ, plain-text reports, PNG processed images, CSV detection data |
|  **Theme Toggle** | Dark / Light themes with full CSS variable theming |
|  **Geocoding** | 4-engine fallback: Nominatim → Photon → Structured → India-scoped |

---

## Screenshots

<div align="center">

### Analysis Tab — VARI Index & Erosion Risk
<img width="1919" height="910" alt="image" src="https://github.com/user-attachments/assets/59b959ff-f052-4eb9-8fa7-5a5c3bdd41fd" />


### Object Detection — Classification Overlay
<img width="1912" height="910" alt="image" src="https://github.com/user-attachments/assets/b392a52c-8bda-4e3a-b8fe-0ef7e803feaf" />


### Digital Deforestation — Hidden Structure Heatmap
<img width="1810" height="908" alt="image" src="https://github.com/user-attachments/assets/4f83750b-6e74-42a5-b87d-264c5ff3c987" />




### Interactive Map View
<img width="1843" height="879" alt="image" src="https://github.com/user-attachments/assets/36eaf6ec-ed63-467d-bebc-27f5a8e2c72b" />


### AI Field Report Widget
<img width="1892" height="818" alt="image" src="https://github.com/user-attachments/assets/c509e003-126d-4a9c-9ebf-fb8c4f2b76e9" />


</div>

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/archai-dashboard.git
cd archai-dashboard
```

### 2. Install Dependencies

```bash
pip install streamlit ultralytics folium streamlit-folium opencv-python groq joblib numpy pandas matplotlib requests
```

### 3. (Optional) Add Model Weights

| File | Purpose | Required? |
|------|---------|-----------|
| `model/best.pt` | YOLOv11 weights for artifact detection | No — demo mode activates automatically |
| `erosion_model.pkl` | Trained erosion risk classifier (joblib bundle) | No — formula fallback is used |

```bash
mkdir model
# Place your YOLOv11 weights:
cp /path/to/best.pt model/best.pt
```

### 4. Run the Dashboard

```bash
streamlit run dashboard_app.py
```

Open your browser at `http://localhost:8501`

---

## Project Structure

```
archai-dashboard/
│
├── dashboard_app.py          # Main Streamlit application
│
├── model/
│   └── best.pt               # YOLOv11 weights (optional)
│
├── erosion_model.pkl          # Erosion ML model (optional)
│
├── assets/                    # README screenshots & banner
│   ├── banner.png
│   ├── dashboard_overview.png
│   ├── tab_analysis.png
│   ├── tab_mound_detection.png
│   ├── tab_deforestation.png
│   ├── tab_map.png
│   └── ai_report_widget.png
│
└── README.md
```

---

## Tabs & Workflow

### 1. Analysis
Upload a satellite or drone image to:
- Run YOLOv11 artifact detection
- Compute VARI vegetation index with segmentation (Very Dense → Bare Soil)
- Auto-detect terrain slope and elevation from image gradients
- Calculate composite erosion risk score

### 2. Object Detection
AI-assisted survey optimisation:
- Detects ALL visible objects in the image
- Classifies each as **Man-made**, **Natural**, or **Uncertain** using a 4-feature scoring model
- Estimates survey time and cost savings vs. traditional field methods
- Generates overlay with colour-coded bounding boxes + detection density heatmap
- AI-powered survey report via Groq LLaMA 3.3 70B

### 3. Deforestation AI
Digitally strips vegetation to reveal buried features:
- VARI-based vegetation masking with adjustable threshold
- Earth-tone channel suppression with configurable intensity
- CLAHE + Sobel edge enhancement for ground detail
- Composite anomaly heatmap (texture + edge density + linearity + ground exposure)
- Connected-component counting for hidden structure estimation

### 4. Map
Interactive Folium map with:
- Google Satellite basemap
- Site origin marker with risk-coloured icon
- Artifact and classified object overlays
- External links to Google Maps, Bing, OpenStreetMap, and Google Earth

### 5. Reports
Export everything:
- `.kmz` for Google Earth Pro
- `.txt` full site report
- `.png` processed images (deforestation view, heatmap, ground enhancement)
- `.csv` detection data

---

## Object Classification Logic

```
VARI > 0.35  ───────────────────────────→  Natural  (override)
Shape regularity < 0.45  ───────────────→  Natural  (override)

Otherwise score = 0.35 × shape_regularity
               + 0.25 × (1 - texture_variance)
               + 0.25 × (1 - VARI)
               + 0.15 × detection_confidence

Score ≥ 0.65  →  Man-made
Score ≤ 0.45  →  Natural
Else          →  Uncertain
```

### Overlay Legend (v4.2)

| Visual Style | Class |
|---|---|
| Thick red border + filled label tag | Man-made (priority) |
| Thin green border + `N xx%` label | Natural |
| Thin gold border + `U xx%` label | Uncertain |

---

## Demo Mode

If no YOLO model is found at `model/best.pt`, the platform runs in **demo mode**:
- Object Detection uses Laplacian blob detection on image texture to generate synthetic candidates
- The full classification pipeline (VARI, shape regularity, texture variance) still runs on detected blobs
- VARI, segmentation, erosion risk, and deforestation AI modules are **fully operational** in demo mode
- AI field reports via Groq work as normal

---

## Geocoding

Location search uses a 4-engine fallback chain:

```
1. Nominatim (OpenStreetMap) — free-form query
2. Nominatim — India-scoped query
3. Photon (Komoot) — free-form query
4. Photon — India-scoped query
5. Nominatim structured (village + state + country)
6. Token-by-token fallback for partial matches
```

---

## Bug Fixes — v4.2

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Object detection only showed mounds/ruins labels | Label whitelist filtered all other YOLO detections | Removed filter — ALL detections accepted; demo mode uses Laplacian blobs |
| Deforestation `NoneType` subscript error | `rgb` stored in session state went `None` on re-render | `rgb` always decoded fresh from live uploader; session state stores processed arrays only |
| Sidebar hidden on narrow viewports | Streamlit auto-collapses sidebar; collapse arrow invisible | CSS forces sidebar open; collapse arrow always visible with `z-index: 999999` |
| Natural / Uncertain boxes invisible on overlay | `cv2.addWeighted` alpha-blend produced invisible boxes | All boxes drawn directly: thin 1px border for Natural/Uncertain, thick 3px for Man-made |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard framework | [Streamlit](https://streamlit.io) |
| Object detection | [Ultralytics YOLOv11](https://ultralytics.com) |
| Computer vision | [OpenCV](https://opencv.org) |
| AI field reports | [Groq — LLaMA 3.3 70B](https://groq.com) |
| Geospatial map | [Folium](https://python-visualization.github.io/folium/) + [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) |
| Geocoding | [Nominatim](https://nominatim.org) + [Photon](https://photon.komoot.io) |
| ML model support | [joblib](https://joblib.readthedocs.io) |
| Fonts | Cormorant Garamond · JetBrains Mono · Archivo Narrow |

---

## Configuration

All configuration is done through the **sidebar** at runtime — no config files needed.

| Setting | Location | Default |
|---------|----------|---------|
| YOLO weights path | Sidebar → Detection Model | `model/best.pt` |
| Confidence threshold | Sidebar slider | 40% |
| Erosion model path | Sidebar → Erosion Model | `erosion_model.pkl` |
| Location name | Sidebar → Location Search | — |
| Latitude / Longitude | Sidebar → Coordinates | 20.5937 N, 78.9629 E |
| Vegetation threshold (VARI) | Deforestation tab slider | 0.18 |
| Removal intensity | Deforestation tab slider | 75% |

---

## Developer

**Hari Krishanan M**  
Archaeological Intelligence Platform — v4.2

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built for archaeology. Powered by AI.*

</div>
