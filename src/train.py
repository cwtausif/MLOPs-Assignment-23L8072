"""
train.py
========
Load the Housing data from data/, train a RandomForest model, and save it
(serialised with joblib) into the model/ directory.

    python src/train.py

Data source:
  * If data/Housing_processed.csv exists (created by src/housing_eda.ipynb) it is
    used - it has the extra engineered features and gives better accuracy.
  * Otherwise the script falls back to the raw data/Housing.csv and does its own
    minimal encoding, so it always runs on a fresh clone.

Every run saves a new model file with a timestamp in its name, so previous
models are never overwritten.
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "Housing.csv"
PROCESSED_PATH = DATA_DIR / "Housing_processed.csv"
MODEL_DIR = BASE_DIR / "model"
TARGET = "price"

BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
FURNISHING_MAP = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}

# columns derived from the target - dropped so the model can't cheat
LEAKAGE_COLS = ["price_per_sqft", "log_price", "price_segment"]


def encode_raw(df):
    """Minimal encoding for the raw Housing.csv (yes/no -> 1/0, furnishing -> 0/1/2)."""
    for col in BINARY_COLS:
        df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})
    df["furnishingstatus"] = df["furnishingstatus"].str.strip().str.lower().map(FURNISHING_MAP)
    return df


def load_data():
    """Return (dataframe, source_label). Prefer the processed CSV, fall back to raw."""
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH), PROCESSED_PATH.name
    if RAW_PATH.exists():
        return encode_raw(pd.read_csv(RAW_PATH)), RAW_PATH.name + " (raw fallback)"
    raise SystemExit(f"No dataset found in {DATA_DIR} (expected Housing.csv).")


def main():
    # 1. load
    df, source = load_data()
    X = df.select_dtypes("number").drop(columns=[TARGET] + LEAKAGE_COLS, errors="ignore")
    y = df[TARGET]

    # 2. split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. train
    model = RandomForestRegressor(n_estimators=2000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. evaluate
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # 5. serialise + save with a timestamp
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"housing_model_{stamp}.joblib"
    joblib.dump(model, model_path)

    # 6. summary
    print("\n" + "=" * 52)
    print("TRAINING SUMMARY".center(52))
    print("=" * 52)
    print(f"Data source       : {source}")
    print(f"Rows              : {len(df)}")
    print(f"Features          : {X.shape[1]}")
    print(f"Train / test split: {len(X_train)} / {len(X_test)}")
    print(f"Model             : RandomForestRegressor(n_estimators=2000)")
    print("-" * 52)
    print(f"Accuracy (R2)     : {r2:.3f}  ({r2 * 100:.1f}%)")
    print(f"MAE               : {mae:,.0f}")
    print(f"RMSE              : {rmse:,.0f}")
    print("-" * 52)
    print(f"Saved model       : {model_path}")
    print("=" * 52)


if __name__ == "__main__":
    main()