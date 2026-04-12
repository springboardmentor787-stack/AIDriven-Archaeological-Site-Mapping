import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ------------------------
# Load Dataset
# ------------------------

data = pd.read_csv("terrain_model/erosion_dataset.csv")
artifact_dir = "terrain_model/artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# Features and labels
X = data.drop("erosion_risk", axis=1)
y = data["erosion_risk"]

# ------------------------
# Train-Test Split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ------------------------
# Train Model
# ------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ------------------------
# Evaluate Model
# ------------------------

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
report_text = classification_report(y_test, preds)
print(report_text)
print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, preds)
print(cm)

report_path = os.path.join(artifact_dir, "erosion_classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.6f}\n\n")
    f.write(report_text)

cm_path = os.path.join(artifact_dir, "erosion_confusion_matrix.png")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Erosion Model Confusion Matrix")
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()

print(f"Saved classification report: {report_path}")
print(f"Saved confusion matrix image: {cm_path}")

# ------------------------
# Save Model
# ------------------------

joblib.dump(model, "erosion_model.pkl")

print("\n✅ Model saved as erosion_model.pkl")