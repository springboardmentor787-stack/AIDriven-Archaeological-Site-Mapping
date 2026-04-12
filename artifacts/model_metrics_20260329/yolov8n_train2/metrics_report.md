# yolov8n_train2 Production Metrics Report

## Snapshot
- Model type: YOLO detection
- Run directory: runs/detect/train2
- Epochs: 50
- Dataset config: data.yaml

## Key Metrics
- Final Precision: 0.695120
- Final Recall: 0.548980
- Final F1 (estimated): 0.613467
- Final mAP50: 0.621800
- Final mAP50-95: 0.386060
- Best mAP50-95: 0.396110 (epoch 49)

## Training/Loss Metrics
- Train Box Loss: 1.274410
- Train Cls Loss: 1.155760
- Train DFL Loss: 1.278770
- Val Box Loss: 1.591480
- Val Cls Loss: 1.486590
- Val DFL Loss: 1.513710

## Operational Metrics
- Training time (sec): 559.67
- Avg epoch time (sec): 11.193
- Model size (MB): 5.9618
- Mean latency CPU (ms): 80.138
- Throughput CPU (FPS): 12.478

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
