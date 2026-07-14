"""Live prediction pipeline.

Pulls current river/weather data, builds an hourly forecast timeline (1-48h
ahead) per station using the trained horizon-aware models, rolls that up
into a single daily badge, and computes a "right now" conditions summary --
then writes everything to one JSON artifact for the Streamlit app to read.
This script recomputes nothing the app itself should recompute.

Run as a script: `python -m src.pipeline.predict`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import STATIONS
from src.features.engineering import LOCAL_TZ, engineer_features
from src.ingestion.usgs import fetch_and_pivot
from src.ingestion.weather import fetch_weather_forecast, join_weather_to_usgs
from src.modeling.features import CONDITION_ORDER, HOURS_AHEAD_COLUMN

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Daily badge only averages over these local-clock hours -- 3am conditions
# shouldn't drag down "today" when nobody's out fishing at 3am.
DAYLIGHT_START_HOUR = 5
DAYLIGHT_END_HOUR = 21

# Station-specific water-quality reading shown in the "right now" summary
# (see src/config.py -- Buford has DO, no turbidity; Medlock is the reverse).
EXTRA_SUMMARY_COLUMNS = {
    "dissolved_oxygen_mgl": "dissolved_oxygen_mgl",
    "turbidity_fnu": "turbidity_fnu",
}


def load_model_and_spec(station_key: str):
    model = xgb.XGBClassifier()
    model.load_model(MODELS_DIR / f"{station_key}_xgb.json")
    spec = json.loads((MODELS_DIR / f"{station_key}_feature_spec.json").read_text())
    return model, spec


def build_live_features(station_key: str) -> pd.DataFrame:
    """Recent USGS readings + Open-Meteo forecast, joined and feature-engineered
    -- the exact same steps used in training, applied to live data."""
    station = STATIONS[station_key]
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(hours=60)).strftime("%Y-%m-%d")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    usgs_df = fetch_and_pivot(station_key, start, end)
    weather_df = fetch_weather_forecast(station["lat"], station["lon"], past_days=2, forecast_days=3)
    joined = join_weather_to_usgs(usgs_df, weather_df)
    return engineer_features(joined)


def predict_timeline(model, spec: dict, anchor: pd.DataFrame) -> list[dict]:
    """Query the model once per hour (1..max_horizon_hours) from the anchor
    row's current conditions, using the exact category sets fixed at train
    time so encodings can't drift between training and live prediction."""
    feature_columns = spec["feature_columns"]
    base_cols = [c for c in feature_columns if c != HOURS_AHEAD_COLUMN]
    max_h = spec["max_horizon_hours"]

    X = pd.concat([anchor[base_cols]] * max_h, ignore_index=True)
    X[HOURS_AHEAD_COLUMN] = range(1, max_h + 1)
    for col, categories in spec["categorical_categories"].items():
        X[col] = pd.Categorical(X[col].astype(str), categories=categories)

    proba = model.predict_proba(X)
    anchor_time = anchor.index[0]

    timeline = []
    for h, p in zip(range(1, max_h + 1), proba):
        ts = anchor_time + pd.Timedelta(hours=h)
        timeline.append(
            {
                "timestamp_utc": ts.isoformat(),
                "condition": CONDITION_ORDER[int(np.argmax(p))],
                "probabilities": {c: round(float(pi), 3) for c, pi in zip(CONDITION_ORDER, p)},
            }
        )
    return timeline


def compute_daily_badge(timeline: list[dict]) -> str | None:
    """Average predicted probabilities across daylight hours, then classify
    the average -- smoother than majority-voting the labels directly, and
    avoids one bad hour dominating the headline verdict."""
    daylight_probs = []
    for row in timeline:
        ts_local = pd.Timestamp(row["timestamp_utc"]).tz_convert(LOCAL_TZ)
        if DAYLIGHT_START_HOUR <= ts_local.hour <= DAYLIGHT_END_HOUR:
            daylight_probs.append([row["probabilities"][c] for c in CONDITION_ORDER])
    if not daylight_probs:
        return None
    avg = np.mean(daylight_probs, axis=0)
    return CONDITION_ORDER[int(np.argmax(avg))]


def compute_current_summary(anchor: pd.DataFrame) -> dict:
    """'Right now' snapshot: the latest known reading for each field (the
    anchor row is already forward-filled, so this is the most recent real
    value even for slower-reporting sensors) -- not an average, not a
    prediction."""
    row = anchor.iloc[0]
    summary = {
        "water_temp_c": round(row["water_temp_c"], 1),
        "discharge_cfs": round(row["discharge_cfs"]),
        "cloud_cover_pct": round(row["cloud_cover"]),
        "precipitation_in": round(row["precipitation"], 2),
        "conductance_uscm": round(row["conductance_uscm"]),
    }
    for col in EXTRA_SUMMARY_COLUMNS:
        if col in anchor.columns:
            summary[col] = round(row[col], 1)
    return summary


def predict_station(station_key: str) -> dict:
    print(f"[{station_key}] pulling live USGS + forecast weather...")
    featured = build_live_features(station_key)
    # Forward-fill before picking the anchor: slower-reporting sensors
    # (water temp, conductance, turbidity, DO) commonly lag discharge by
    # 15min-1hr in near-real-time telemetry, so the single freshest row is
    # exactly where these are most likely to still be NaN. The model has no
    # "neutral" handling for that the way composite.py's rule-based scoring
    # does -- it just learned some default split for missing values in
    # training, which isn't reliably neutral. Carrying forward the last
    # known real reading is a better anchor than a row with holes in it.
    anchor = featured.ffill().iloc[[-1]]
    print(f"[{station_key}] anchor time (UTC): {anchor.index[0]}")

    model, spec = load_model_and_spec(station_key)
    timeline = predict_timeline(model, spec, anchor)
    daily_badge = compute_daily_badge(timeline)
    current_summary = compute_current_summary(anchor)

    return {
        "station": station_key,
        "anchor_time_utc": anchor.index[0].isoformat(),
        "current_summary": current_summary,
        "daily_badge": daily_badge,
        "hourly_timeline": timeline,
    }


def main():
    result = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "stations": {},
    }
    for station_key in STATIONS:
        result["stations"][station_key] = predict_station(station_key)

    out_dir = DATA_DIR / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latest_forecast.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
