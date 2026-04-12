# XGBoost Erosion Production Report

## Evaluation Scope
- Dataset: terrain_model/erosion_dataset.csv
- Target: erosion
- Features: slope, vegetation, elevation, rainfall, soil, boulders, ruins, structures
- Train/Test rows: 800/200

## Classification Metrics
- Accuracy: 0.905000
- Precision: 0.909091
- Recall: 0.900000
- F1: 0.904523
- ROC-AUC: 0.966
- Average Precision: 0.9691711745275607

## Regression-style Metrics on Probabilities
- MSE: 0.074041
- MAE: 0.136555
- RMSE: 0.272105
- R2: 0.703835

## Operational
- Model size (MB): 0.4968
- Mean latency CPU (ms): 2.0813
- Throughput CPU (FPS): 480.469

## Produced Artifacts
- metrics_summary.json
- classification_report_eval.txt
- confusion_matrix_eval.png
- roc_curve_eval.png
- pr_curve_eval.png
