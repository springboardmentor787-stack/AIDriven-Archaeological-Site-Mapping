# Model Metrics Artifact Index

Generated at: 2026-03-29T14:15:45.739492
Git commit: c63af9ccaab2c0dc63c80b372dddba7b4cc66e38

## Models Included
- deeplabv3plus_segmentation: see deeplabv3plus_segmentation/metrics_report.md and deeplabv3plus_segmentation/metrics_summary.json
- xgboost_erosion: see xgboost_erosion/metrics_report.md and xgboost_erosion/metrics_summary.json
- yolo11n_base: see yolo11n_base/metrics_report.md and yolo11n_base/metrics_summary.json
- yolov8n_base: see yolov8n_base/metrics_report.md and yolov8n_base/metrics_summary.json
- yolov8n_train2: see yolov8n_train2/metrics_report.md and yolov8n_train2/metrics_summary.json
- yolov8s_archaeology2: see yolov8s_archaeology2/metrics_report.md and yolov8s_archaeology2/metrics_summary.json
- yolov8s_base: see yolov8s_base/metrics_report.md and yolov8s_base/metrics_summary.json

## Dataset Inventory
{
  "detection": {
    "train": {
      "images": 648,
      "labels": 648
    },
    "valid": {
      "images": 61,
      "labels": 61
    },
    "test": {
      "images": 31,
      "labels": 31
    }
  },
  "segmentation": {
    "train": {
      "images": 648,
      "masks": 648
    },
    "valid": {
      "images": 61,
      "masks": 61
    },
    "test": {
      "images": 31,
      "masks": 31
    }
  },
  "terrain_rows": 1000
}

## Notes
- Detection uses mAP metrics (standard for object detection).
- Segmentation includes IoU and Dice computed on validation split.
- XGBoost includes both classification and regression-style probability metrics.
- Base checkpoints without local training runs include operational metrics and reproducibility metadata.