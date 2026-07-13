"""Per-station XGBoost training on the historical backfill.

Trains one model per station that can forecast at *any* horizon from 1 to
MAX_HORIZON_HOURS hours ahead, by including the horizon itself as a feature
(`hours_ahead`). At prediction time, the same current-conditions row gets
queried once per hour with hours_ahead=1, 2, 3, ... to build an hourly
timeline, rather than needing a separate model per horizon.

Run as a script: `python -m src.modeling.train`
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils import compute_sample_weight

from src.config import STATIONS
from src.modeling.features import (
    CONDITION_ORDER,
    HOURS_AHEAD_COLUMN,
    encode_condition,
    get_feature_columns,
    prepare_features,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Time-based split: everything before this date trains the model, everything
# on/after it is held out for evaluation. Not a random split -- this is time
# series data with strong autocorrelation, so a random split would leak
# nearby timestamps between train and test and overstate performance.
TEST_START = "2025-08-01"

MAX_HORIZON_HOURS = 48
PERIODS_PER_HOUR = 4  # 15-min USGS grid
SPOT_CHECK_HORIZONS = [1, 6, 12, 24, 48]


def build_horizon_dataset(df: pd.DataFrame, base_feature_columns: list[str]):
    """One training row per (original timestamp, horizon) pair: the features
    are always the *current* reading, but the label comes from `h` hours
    later, and `h` itself becomes a feature (`hours_ahead`) so a single model
    learns how the answer changes as the horizon grows.
    """
    y_encoded = encode_condition(df["condition"])
    frames, targets = [], []
    for h in range(1, MAX_HORIZON_HOURS + 1):
        periods = h * PERIODS_PER_HOUR
        y_h = y_encoded.shift(-periods)
        keep = y_h.notna()
        X_h = df.loc[keep, base_feature_columns].copy()
        X_h[HOURS_AHEAD_COLUMN] = h
        frames.append(X_h)
        targets.append(y_h[keep])
    X_raw = pd.concat(frames)
    y = pd.concat(targets).astype(int)
    return X_raw, y


def train_station(station_key: str) -> dict:
    df = pd.read_parquet(DATA_DIR / "processed" / f"{station_key}_processed.parquet")

    base_feature_columns = get_feature_columns(df)
    feature_columns = base_feature_columns + [HOURS_AHEAD_COLUMN]

    t0 = time.time()
    X_raw, y = build_horizon_dataset(df, base_feature_columns)
    X = prepare_features(X_raw, feature_columns)
    print(f"[{station_key}] built {len(X):,} horizon-expanded training rows in {time.time()-t0:.0f}s", flush=True)

    is_test = X.index >= pd.Timestamp(TEST_START, tz="UTC")
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]

    sample_weight = compute_sample_weight("balanced", y_train)

    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(CONDITION_ORDER),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        enable_categorical=True,
        eval_metric="mlogloss",
        random_state=0,
        tree_method="hist",
        n_jobs=-1,  # XGBoost doesn't always default to using all cores
    )
    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    print(f"[{station_key}] trained in {time.time()-t0:.0f}s", flush=True)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CONDITION_ORDER))))

    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(
        ascending=False
    )

    print(f"\n=== {station_key} (horizon-aware, 1-{MAX_HORIZON_HOURS}h) ===")
    print(f"train rows: {len(X_train):,}  test rows: {len(X_test):,}")
    print(f"overall accuracy: {acc:.3f}  macro F1: {f1_macro:.3f}")
    print(f"confusion matrix (rows=actual, cols=predicted), labels={CONDITION_ORDER}:")
    print(pd.DataFrame(cm, index=CONDITION_ORDER, columns=CONDITION_ORDER))
    print("top feature importances:")
    print(importances.head(10).to_string())

    print("\naccuracy by horizon (spot check -- should degrade as horizon grows):")
    horizon_col = X_test[HOURS_AHEAD_COLUMN]
    horizon_accuracies = {}
    for h in SPOT_CHECK_HORIZONS:
        mask = horizon_col == h
        if mask.sum() == 0:
            continue
        h_acc = accuracy_score(y_test[mask], y_pred[mask])
        horizon_accuracies[h] = h_acc
        print(f"  {h:>2}h ahead: accuracy={h_acc:.3f}  (n={mask.sum():,})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{station_key}_xgb.json"
    model.save_model(model_path)

    feature_spec = {
        "feature_columns": feature_columns,
        "categorical_categories": {
            col: X[col].cat.categories.tolist()
            for col in ["season", "time_of_day"]
            if col in X.columns
        },
        "condition_order": CONDITION_ORDER,
        "test_start": TEST_START,
        "max_horizon_hours": MAX_HORIZON_HOURS,
        "hours_ahead_column": HOURS_AHEAD_COLUMN,
    }
    spec_path = MODELS_DIR / f"{station_key}_feature_spec.json"
    spec_path.write_text(json.dumps(feature_spec, indent=2))

    return {
        "station": station_key,
        "accuracy": acc,
        "macro_f1": f1_macro,
        "confusion_matrix": cm.tolist(),
        "top_features": importances.head(10).to_dict(),
        "horizon_accuracies": horizon_accuracies,
    }


def main():
    results = {}
    for station_key in STATIONS:
        results[station_key] = train_station(station_key)
    return results


if __name__ == "__main__":
    main()
