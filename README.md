# Archaeological Site Mapping AI

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![License](https://img.shields.io/badge/License-TBD-lightgrey)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2026--03--29-informational)
![Project Status](https://img.shields.io/badge/Status-Active-success)

Production-grade geo-AI system for archaeological landscape intelligence using computer vision, geospatial feature engineering, explainable ML, and LLM-assisted interpretation.

## Demo / Preview

### Node.js UI (Next.js)

<p align="center">
	<img src="docs/Next%20JS%20UI%20Samples/1.png" width="48%" alt="Next.js UI sample 1" />
	<img src="docs/Next%20JS%20UI%20Samples/2.png" width="48%" alt="Next.js UI sample 2" />
</p>
<p align="center">
	<img src="docs/Next%20JS%20UI%20Samples/3.png" width="48%" alt="Next.js UI sample 3" />
	<img src="docs/Next%20JS%20UI%20Samples/4.png" width="48%" alt="Next.js UI sample 4" />
</p>

### Streamlit UI

<p align="center">
	<img src="docs/Streamlit%20UI%20Samples/1.png" width="48%" alt="Streamlit UI sample 1" />
	<img src="docs/Streamlit%20UI%20Samples/2.png" width="48%" alt="Streamlit UI sample 2" />
</p>
<p align="center">
	<img src="docs/Streamlit%20UI%20Samples/3.png" width="48%" alt="Streamlit UI sample 3" />
	<img src="docs/Streamlit%20UI%20Samples/4.png" width="48%" alt="Streamlit UI sample 4" />
</p>

## Problem Statement

Archaeological surveys over large regions are slow, expensive, and difficult to scale manually. Teams need a system that can:

- detect ruins and terrain classes from imagery,
- estimate erosion risk at candidate locations,
- explain *why* risk is predicted,
- provide outputs usable by both field and technical teams.

## Solution Overview

This repository combines:

1. Object detection for structures/ruins/vegetation cues (YOLO).
2. Semantic segmentation for dense surface understanding (DeepLabV3+).
3. Terrain and geo-context feature engineering (slope, elevation, rainfall, soil, class ratios).
4. Erosion risk scoring with XGBoost.
5. SHAP-based explainability plus optional LLM narrative generation.
6. Dual interfaces: Streamlit and a Next.js + Python worker architecture.

This project is built with a focus on interpretability, combining SHAP and LLMs to ensure predictions are transparent and actionable.

What makes it unique is the end-to-end bridge from image evidence to interpretable, location-aware risk output.

## Key Features

- YOLO-based archaeological object detection
- DeepLabV3+ terrain segmentation
- XGBoost erosion risk classification
- SHAP feature attribution for local explainability
- LLM insight generation for human-readable analysis
- Streamlit and Next.js interfaces
- API fallback strategies for geospatial inputs
- Production-style model metrics artifact pack

## System Architecture

High-level pipeline:

1. Input image + coordinates
2. CV branch A: YOLO detection boxes/classes
3. CV branch B: DeepLab segmentation mask
4. Feature Engineering: ratios + geospatial API features
5. ML Inference: XGBoost erosion prediction
6. Explainability: SHAP top factors
7. LLM Layer: narrative interpretation (optional)
8. UI Layer: overlays, metrics, report-ready summaries

Interaction by layer:

- Computer Vision: extracts visual evidence from aerial imagery.
- Feature Engineering: converts raw vision + geo signals into model features.
- ML Model: predicts erosion risk probability and class.
- LLM: translates numeric explanations into readable insights.

Detailed design is documented in `ARCHITECTURE.md`.

## Tech Stack

### ML / CV

- PyTorch
- Ultralytics YOLOv8
- segmentation-models-pytorch (DeepLabV3+)
- scikit-learn
- xgboost
- SHAP

### Backend

- Python
- Streamlit
- joblib
- NumPy, pandas

### Frontend

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS

### APIs

- Open-Elevation API
- Open-Meteo API
- SoilGrids API

## Quick Start Guide

### 1. Clone and install

```bash
git clone <your-repo-url>
cd Archaeological-Site-Mapping-AI-V2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run backend (Streamlit)

```bash
python -m streamlit run app.py
```

### 3. Run frontend (Next.js)

```bash
cd geo-ai-ui
npm install
npm run dev
```

### 4. Core scripts

```bash
python scripts/data_prep/generate_masks.py
python scripts/training/train_seg.py
python scripts/training/train_deeplab_seg.py
python scripts/inference/predict.py --source dataset/test/images
python scripts/inference/demo_pipeline.py
```

## Project Structure

```text
Archaeological-Site-Mapping-AI-V2/
|-- app.py
|-- config/
|   `-- data.yaml
|-- scripts/
|   |-- data_prep/
|   |-- training/
|   |-- inference/
|   |-- reporting/
|   `-- utils/
|-- models/
|-- dataset/
|-- seg_dataset/
|-- terrain_model/
|-- runs/
|-- artifacts/
|-- geo-ai-ui/
|-- docs/
|   |-- screenshots/
|   `-- reports/
|-- requirements.txt
|-- README.md
|-- METRICS.md
|-- ARCHITECTURE.md
|-- FEATURES.md
|-- MODELS.md
|-- EXPLAINABILITY.md
|-- API.md
|-- UI_UX.md
|-- FUTURE_WORK.md
|-- CONTRIBUTING.md
`-- CHANGELOG.md
```

## Results Summary

For full benchmark tables, confusion matrices, cross-validation, and interpretation, see `METRICS.md`.

## Demo Usage Flow

1. Start Streamlit (`app.py`) or Next.js UI (`geo-ai-ui`).
2. Upload image and provide/select coordinates.
3. View YOLO detections and DeepLab segmentation overlays.
4. Trigger erosion prediction from engineered geo-visual features.
5. Inspect SHAP top contributors and optional LLM explanation.
6. Export report for downstream analysis.

## Documentation

- [Architecture](./ARCHITECTURE.md)
- [Models](./MODELS.md)
- [Features](./FEATURES.md)
- [Explainability](./EXPLAINABILITY.md)
- [Metrics](./METRICS.md)
- [API](./API.md)
- [UI/UX](./UI_UX.md)
- [Future Work](./FUTURE_WORK.md)

Additional project docs:

- [Contributing](./CONTRIBUTING.md)
- [Changelog](./CHANGELOG.md)

## Why This Project Matters

- Reduces dependency on manual surveys
- Enables faster conservation decisions
- Combines AI + domain knowledge
- Provides explainable predictions (not black-box)

## Future Scope

- DEM-aware slope extraction
- Better soil and geological priors
- Multi-region transfer and domain adaptation
- Model serving and SaaS deployment
- Active learning loop with expert feedback

See `FUTURE_WORK.md` for details.

## License

Add your license file and update this section (for example MIT/Apache-2.0) before public release.

## Author

Harshil Somisetty

If you are using this project for research or deployment, please cite the repository and include the model/artifact version from `artifacts/model_metrics_*/index.json`.
