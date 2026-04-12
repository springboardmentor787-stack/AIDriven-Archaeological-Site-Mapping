import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv("Milestone 3/erosion_dataset (1).csv")
    
    X = df[["slope", "ndvi", "elevation"]]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nan_indices_train = y_train.isna()
    X_train_cleaned = X_train[~nan_indices_train]
    y_train_cleaned = y_train[~nan_indices_train]

    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    rf.fit(X_train_cleaned, y_train_cleaned)
    joblib.dump(rf, 'rf_erosion.pkl')
    print("Saved rf_erosion.pkl")

    print("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=50, random_state=42)
    xgb.fit(X_train_cleaned, y_train_cleaned)
    joblib.dump(xgb, 'xgb_erosion.pkl')
    print("Saved xgb_erosion.pkl")

    print("Done!")

if __name__ == "__main__":
    main()
