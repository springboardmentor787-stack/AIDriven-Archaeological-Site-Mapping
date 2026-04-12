import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
)

print("Loading dataset...")

df = pd.read_csv("terrain_model/erosion_dataset.csv")
artifact_dir = "terrain_model/artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# Features & target
expected_features = [
    "slope", "vegetation", "elevation", "rainfall", "soil",
    "boulders", "ruins", "structures"
]
X = df[expected_features]
y = df["erosion"]

print("\nClass Distribution:")
print(y.value_counts())

# Train-test split (VERY IMPORTANT: stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("\nEvaluating model...")

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

rmse = float(np.sqrt(mean_squared_error(y_test, probs)))
r2 = float(r2_score(y_test, probs))

print("\nAccuracy:", accuracy_score(y_test, preds))
print("RMSE (probability vs label):", rmse)
print("R2 (probability vs label):", r2)

print("\nClassification Report:\n")
report_text = classification_report(y_test, preds)
print(report_text)

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, preds)
print(cm)

report_path = os.path.join(artifact_dir, "xgboost_classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, preds):.6f}\n\n")
    f.write(f"RMSE (probability vs label): {rmse:.6f}\n")
    f.write(f"R2 (probability vs label): {r2:.6f}\n\n")
    f.write(report_text)

cm_path = os.path.join(artifact_dir, "xgboost_confusion_matrix.png")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", colorbar=False)
plt.title("XGBoost Erosion Confusion Matrix")
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()

print(f"Saved classification report: {report_path}")
print(f"Saved confusion matrix image: {cm_path}")

# Save model
joblib.dump(model, "terrain_model/erosion_model.pkl")

print("\n✅ Model saved as terrain_model/erosion_model.pkl")

# Feature importance analysis and plot
importances = model.feature_importances_

print("\n🔍 Feature Importance Scores:")
for name, score in zip(X.columns, importances):
    print(f"  {name}: {score:.3f}")

plt.figure(figsize=(8, 5))
plt.bar(X.columns, importances)
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("terrain_model/feature_importance.png", dpi=150)
plt.show()

print("✅ Feature importance plot saved as terrain_model/feature_importance.png")

# Reload model and run a sample prediction with explicit 8-feature vector.
loaded_model = joblib.load("terrain_model/erosion_model.pkl")

if all(col in X.columns for col in expected_features):
    features = X[expected_features].iloc[[0]].to_numpy(dtype=np.float32)
else:
    print("⚠️ Expected 8-feature schema not found; using first row from dataset columns.")
    features = X.iloc[[0]].to_numpy(dtype=np.float32)

sample_prediction = loaded_model.predict(features)[0]
print(f"✅ Sample prediction from reloaded model: {sample_prediction}")