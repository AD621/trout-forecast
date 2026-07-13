"""Historical backfill: USGS + weather -> features -> composite label,
saved to data/processed/ for later model training.

Run as a script: `python -m src.pipeline.backfill`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import STATIONS
from src.features.engineering import engineer_features
from src.ingestion.usgs import fetch_and_pivot
from src.ingestion.weather import fetch_weather_historical, join_weather_to_usgs
from src.scoring.composite import score_and_classify

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def build_station_dataset(station_key: str, start: str, end: str) -> pd.DataFrame:
    station = STATIONS[station_key]
    print(f"[{station_key}] pulling USGS {station['usgs_id']} {start}..{end}")
    usgs_df = fetch_and_pivot(station_key, start, end)
    print(f"[{station_key}] {len(usgs_df)} USGS rows")

    print(f"[{station_key}] pulling Open-Meteo history for ({station['lat']}, {station['lon']})")
    weather_df = fetch_weather_historical(station["lat"], station["lon"], start, end)
    joined = join_weather_to_usgs(usgs_df, weather_df)

    featured = engineer_features(joined)
    labeled = score_and_classify(featured, station_key)
    return labeled


def run_backfill(start: str = "2023-01-01", end: str = "2026-01-01") -> dict[str, pd.DataFrame]:
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    results = {}
    for station_key in STATIONS:
        df = build_station_dataset(station_key, start, end)
        out_path = DATA_DIR / "processed" / f"{station_key}_processed.parquet"
        df.to_parquet(out_path)
        print(f"[{station_key}] wrote {out_path} ({len(df)} rows)")

        balance = df["condition"].value_counts(normalize=True).round(3) * 100
        print(f"[{station_key}] class balance:\n{balance}\n")
        results[station_key] = df
    return results


if __name__ == "__main__":
    run_backfill()
