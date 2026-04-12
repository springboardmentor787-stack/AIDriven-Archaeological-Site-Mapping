import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

df = pd.read_csv("terrain_model/erosion_dataset.csv")

X = df.drop("erosion", axis=1)
y = df["erosion"]

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05
)

scores = cross_val_score(model, X, y, cv=5)
rmse_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring=make_scorer(rmse_score),
)
r2_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring=make_scorer(r2_score),
)

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("CV RMSE:", rmse_scores)
print("Mean RMSE:", float(np.mean(rmse_scores)))
print("CV R2:", r2_scores)
print("Mean R2:", float(np.mean(r2_scores)))