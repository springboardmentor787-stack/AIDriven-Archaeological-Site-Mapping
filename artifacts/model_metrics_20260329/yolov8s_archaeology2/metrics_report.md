# yolov8s_archaeology2 Production Metrics Report

## Snapshot
- Model type: YOLO detection
- Run directory: runs/detect/yolov8s_archaeology2
- Epochs: 80
- Dataset config: data.yaml

## Key Metrics
- Final Precision: 0.899780
- Final Recall: 0.720100
- Final F1 (estimated): 0.799975
- Final mAP50: 0.832060
- Final mAP50-95: 0.586110
- Best mAP50-95: 0.586730 (epoch 78)

## Training/Loss Metrics
- Train Box Loss: 0.864320
- Train Cls Loss: 0.559350
- Train DFL Loss: 1.042260
- Val Box Loss: 1.434700
- Val Cls Loss: 1.242220
- Val DFL Loss: 1.557650

## Operational Metrics
- Training time (sec): 1509.27
- Avg epoch time (sec): 18.866
- Model size (MB): 21.4823
- Mean latency CPU (ms): 206.248
- Throughput CPU (FPS): 4.849

## Produced Artifacts
- metrics_summary.json
- training_curves_custom.png
- raw_artifacts/results.png
- raw_artifacts/confusion_matrix.png
- raw_artifacts/confusion_matrix_normalized.png
- raw_artifacts/PR_curve.png
- raw_artifacts/P_curve.png
- raw_artifacts/R_curve.png
- raw_artifacts/F1_curve.png
