<div align="center">

```
░█████╗░██████╗░░█████╗░██╗░░██╗░█████╗░██╗
██╔══██╗██╔══██╗██╔══██╗██║░░██║██╔══██╗██║
███████║██████╔╝██║░░╚═╝███████║███████║██║
██╔══██║██╔══██╗██║░░██╗██╔══██║██╔══██║██║
██║░░██║██║░░██║╚█████╔╝██║░░██║██║░░██║██║
╚═╝░░╚═╝╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝
```

### ᴀʀᴄʜᴀᴇᴏʟᴏɢɪᴄᴀʟ ɪɴᴛᴇʟʟɪɢᴇɴᴄᴇ ᴘʟᴀᴛꜰᴏʀᴍ

*Satellite · Drone · AI · Hidden Ruins · v4.2*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00d4ff?style=for-the-badge)](https://ultralytics.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F97316?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Release-v4.2-gold?style=for-the-badge)]()

<br/>

> *"What has been hidden beneath forest and stone for millennia —*
> *revealed in seconds."*

<br/>

![ArchAI Banner](https://github.com/user-attachments/assets/e15fa236-4008-45ae-a5b1-d97a07690bb6)

</div>

---

<br/>

## ◈ What is ArchAI?

**ArchAI** is an end-to-end archaeological intelligence platform that fuses **computer vision**, **geospatial AI**, and **large language models** into a single unified dashboard. Upload satellite or drone imagery and extract instant, structured insights — from buried mound detection to AI-generated field reports — without touching a single command line.

Built for field archaeologists, remote sensing analysts, and heritage researchers who need results *fast.*

<br/>

---

<br/>

## ◈ Feature Matrix

<br/>

```
 MODULE                    CAPABILITY                                 STATUS
 ─────────────────────────────────────────────────────────────────────────────
 Artifact Detection       YOLOv11 inference · confidence filtering    ██████  LIVE
 VARI Vegetation Index    RGB proxy for NDVI · 5-class segmentation   ██████  LIVE
 Erosion Risk Engine      Slope + elevation + vegetation composite    ██████  LIVE
 Object Classification    Man-made / Natural / Uncertain scoring      ██████  LIVE
 Digital Deforestation    AI vegetation removal · anomaly heatmap     ██████  LIVE
 AI Field Reports         Groq LLaMA 3.3 70B structured assessments   ██████  LIVE
 Interactive Map          Google Satellite · Folium overlays          ██████  LIVE
 Export Suite             KMZ · TXT · PNG · CSV                       ██████  LIVE
 Geocoding Engine         4-engine fallback · India-scoped search     ██████  LIVE
 Theme Toggle             Dark / Light · full CSS variable theming    ██████  LIVE
```

<br/>

---

<br/>

## ◈ Screenshots

<br/>

<div align="center">

**— Analysis Tab · VARI Index & Erosion Risk —**

<img width="1919" alt="Analysis Tab" src="https://github.com/user-attachments/assets/59b959ff-f052-4eb9-8fa7-5a5c3bdd41fd" />

<br/><br/>

**— Object Detection · Classification Overlay —**

<img width="1912" alt="Object Detection" src="https://github.com/user-attachments/assets/b392a52c-8bda-4e3a-b8fe-0ef7e803feaf" />

<br/><br/>

**— Digital Deforestation · Hidden Structure Heatmap —**

<img width="1810" alt="Deforestation AI" src="https://github.com/user-attachments/assets/4f83750b-6e74-42a5-b87d-264c5ff3c987" />

<br/><br/>

**— Interactive Map View —**

<img width="1843" alt="Map View" src="https://github.com/user-attachments/assets/36eaf6ec-ed63-467d-bebc-27f5a8e2c72b" />

<br/><br/>

**— AI Field Report Widget —**

<img width="1892" alt="AI Field Report" src="https://github.com/user-attachments/assets/c509e003-126d-4a9c-9ebf-fb8c4f2b76e9" />

</div>

<br/>

---

<br/>

## ◈ Installation

<br/>

### Ⅰ — Prerequisites

```
Python ≥ 3.9        pip (latest)
```

<br/>

### Ⅱ — Clone

```bash
git clone https://github.com/YOUR_USERNAME/archai-dashboard.git
cd archai-dashboard
```

<br/>

### Ⅲ — Install Dependencies

```bash
pip install streamlit ultralytics folium streamlit-folium \
            opencv-python groq joblib numpy pandas \
            matplotlib requests
```

<br/>

### Ⅳ — Model Weights *(Optional)*

```bash
mkdir model
cp /path/to/best.pt model/best.pt          # YOLOv11 artifact weights
cp /path/to/erosion_model.pkl .            # Trained erosion classifier
```

> **No weights? No problem.** ArchAI runs in **Demo Mode** automatically — all AI modules remain fully operational.

<br/>

### Ⅴ — Launch

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

<br/>

---

<br/>

## ◈ Project Structure

<br/>

```
archai/
│
├── config/
│   ├── settings.py          ← Groq key · themes · session defaults
│   └── styles.py            ← Global CSS
│
├── modules/
│   ├── image_processing.py  ← VARI · segmentation · erosion · terrain
│   ├── mound_detection.py   ← Candidate detection · classification · heatmap
│   └── deforestation.py     ← Vegetation mask · removal · anomaly detection
│
├── utils/
│   ├── geocoding.py         ← 4-engine geocoder with India fallback
│   ├── model_loaders.py     ← YOLO + erosion model loaders
│   └── export.py            ← KML / KMZ builder
│
├── widgets/
│   ├── ai_report.py         ← Erosion AI report widget
│   ├── mound_report.py      ← Mound survey AI report widget
│   └── deforest_report.py   ← Deforestation AI report widget
│
├── tabs/
│   ├── sidebar.py
│   ├── tab_analysis.py
│   ├── tab_mound.py
│   ├── tab_deforestation.py
│   ├── tab_map.py
│   ├── tab_reports.py
│   └── tab_about.py
│
└── app.py                   ← Entry point
```

<br/>

---

<br/>

## ◈ Module Walkthrough

<br/>

### ① Analysis
Upload satellite or drone imagery to:
- Run **YOLOv11** artifact detection with confidence filtering
- Compute **VARI** vegetation index across 5 segmentation classes *(Very Dense → Bare Soil)*
- Auto-detect terrain **slope & elevation** from image gradients
- Calculate composite **erosion risk score**

<br/>

### ② Object Detection
AI-assisted survey optimisation:
- Detects **all** visible objects in the image — not just ruins
- Classifies each detection as **Man-made**, **Natural**, or **Uncertain** via 4-feature scoring
- Estimates **survey time & cost savings** vs. traditional field methods
- Generates colour-coded bounding box overlay + **detection density heatmap**
- AI-powered survey report via Groq **LLaMA 3.3 70B**

<br/>

### ③ Deforestation AI
Digitally strips vegetation to reveal buried features:
- **VARI-based** vegetation masking with adjustable threshold
- Earth-tone channel suppression with configurable intensity
- **CLAHE + Sobel** edge enhancement for ground detail
- Composite anomaly heatmap — texture · edge density · linearity · ground exposure
- Connected-component counting for **hidden structure estimation**

<br/>

### ④ Map
Interactive Folium map with:
- **Google Satellite** basemap
- Site origin marker with risk-coloured icon
- Artifact and classified object overlays
- External links → Google Maps · Bing · OpenStreetMap · Google Earth

<br/>

### ⑤ Reports
Export everything:

```
  .kmz   →  Google Earth Pro
  .txt   →  Full site report
  .png   →  Processed images (deforestation · heatmap · ground enhancement)
  .csv   →  All detection data
```

<br/>

---

<br/>

## ◈ Classification Logic

<br/>

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OBJECT CLASSIFICATION ENGINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   VARI > 0.35          ────────────────────────▶   NATURAL          │
│   Shape regularity < 0.45  ────────────────────▶   NATURAL          │
│                                                                     │
│   Otherwise:                                                        │
│                                                                     │
│   Score  =  0.35 × shape_regularity                                 │
│          +  0.25 × (1 − texture_variance)                           │
│          +  0.25 × (1 − VARI)                                       │
│          +  0.15 × detection_confidence                             │
│                                                                     │
│   Score ≥ 0.65  ──▶  MAN-MADE                                       │
│   Score ≤ 0.45  ──▶  NATURAL                                        │
│   Otherwise     ──▶  UNCERTAIN                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

<br/>

### Overlay Legend

| Visual Style | Class |
|:---|:---|
| `▐███▌` Thick red border · filled label | **Man-made** *(priority)* |
| `▐─ ─▌` Thin green border · `N xx%` | **Natural** |
| `▐─ ─▌` Thin gold border · `U xx%` | **Uncertain** |

<br/>

---

<br/>

## ◈ Geocoding Architecture

<br/>

```
  Query Input
      │
      ▼
  ① Nominatim        (OpenStreetMap · free-form)
      │ fail
      ▼
  ② Nominatim        (India-scoped)
      │ fail
      ▼
  ③ Photon           (Komoot · free-form)
      │ fail
      ▼
  ④ Photon           (India-scoped)
      │ fail
      ▼
  ⑤ Nominatim        (structured: village + state + country)
      │ fail
      ▼
  ⑥ Token-by-token   (partial match fallback)
      │
      ▼
  Coordinates Resolved ✓
```

<br/>

---

<br/>

## ◈ Configuration

<br/>

All settings live in the **sidebar** — no config files needed.

| Setting | Location | Default |
|:---|:---|:---|
| YOLO weights path | Sidebar → Detection Model | `model/best.pt` |
| Confidence threshold | Sidebar slider | `40%` |
| Erosion model path | Sidebar → Erosion Model | `erosion_model.pkl` |
| Location name | Sidebar → Location Search | — |
| Latitude / Longitude | Sidebar → Coordinates | `20.5937 N · 78.9629 E` |
| Vegetation threshold (VARI) | Deforestation tab | `0.18` |
| Removal intensity | Deforestation tab | `75%` |

<br/>

---

<br/>

## ◈ Demo Mode

When no YOLO weights are present at `model/best.pt`, ArchAI activates **Demo Mode** automatically:

```
  ✓  Laplacian blob detection replaces YOLO inference
  ✓  Full classification pipeline runs on detected blobs
  ✓  VARI, segmentation, and erosion risk — fully operational
  ✓  Digital Deforestation AI — fully operational
  ✓  Groq LLaMA field reports — fully operational
```

<br/>

---

<br/>

## ◈ Changelog · v4.2

<br/>

| # | Bug | Root Cause | Fix |
|:---|:---|:---|:---|
| 1 | Object detection only showed mounds/ruins | Label whitelist filtered all other YOLO detections | Removed filter — all detections accepted; demo mode uses Laplacian blobs |
| 2 | Deforestation `NoneType` subscript error | `rgb` in session state went `None` on re-render | `rgb` always decoded fresh from live uploader |
| 3 | Sidebar hidden on narrow viewports | Streamlit auto-collapse; arrow invisible | CSS forces sidebar open; collapse arrow z-indexed to `999999` |
| 4 | Natural/Uncertain boxes invisible | `cv2.addWeighted` alpha-blend failure | All boxes drawn directly — `1px` Natural/Uncertain, `3px` Man-made |

<br/>

---

<br/>

## ◈ Tech Stack

<br/>

```
  Dashboard Framework    Streamlit
  Object Detection       Ultralytics YOLOv11
  Computer Vision        OpenCV
  AI Field Reports       Groq — LLaMA 3.3 70B
  Geospatial Map         Folium + streamlit-folium
  Geocoding              Nominatim + Photon (Komoot)
  ML Model Support       joblib
  Typography             Cormorant Garamond · JetBrains Mono · Archivo Narrow
```

<br/>

---

<br/>

<div align="center">

```
                               ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**ᴅᴇᴠᴇʟᴏᴘᴇᴅ ʙʏ**

### Hari Krishnan M

*Archaeological Intelligence Platform · v4.2*

```
                             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

*"The past is never truly buried — only waiting to be seen."*

</div>
