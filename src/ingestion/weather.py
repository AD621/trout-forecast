"""Open-Meteo weather ingestion and joining onto the USGS timestamp grid.

No API key required. Historical pulls use the archive endpoint (for training
data); live pulls use the forecast endpoint (which also backfills a short
"past_days" window so rolling features have enough history at predict time).
"""

from __future__ import annotations

import pandas as pd
import requests

from src.config import (
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_HISTORICAL_URL,
    OPEN_METEO_HOURLY_VARS,
)


def _hourly_json_to_df(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def fetch_weather_historical(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Hourly historical weather for [start, end] (YYYY-MM-DD, UTC).

    Open-Meteo's archive endpoint is backed by ERA5 reanalysis, which has
    ~5 days of processing latency. Requesting an `end` date within that
    window returns forecast-model output for the gap instead of observed
    data, including for hours that haven't happened yet. Use
    `fetch_weather_forecast(past_days=...)` for anything near the current
    date.
    """
    r = requests.get(
        OPEN_METEO_HISTORICAL_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
            "timezone": "UTC",
        },
        timeout=30,
    )
    r.raise_for_status()
    return _hourly_json_to_df(r.json())


def fetch_weather_forecast(lat: float, lon: float, past_days: int = 2, forecast_days: int = 3) -> pd.DataFrame:
    """Hourly forecast weather, including a short trailing window of recent
    actuals (`past_days`) so rolling features can be computed at predict time.
    """
    r = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
            "timezone": "UTC",
            "past_days": past_days,
            "forecast_days": forecast_days,
        },
        timeout=30,
    )
    r.raise_for_status()
    return _hourly_json_to_df(r.json())


def join_weather_to_usgs(usgs_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill hourly weather onto the finer (15-min) USGS timestamp grid.

    An hourly reading applies to every USGS timestamp until the next hourly
    reading arrives -- deliberately not interpolated, since precipitation and
    cloud cover are not smooth continuous signals.
    """
    weather_reindexed = weather_df.reindex(
        weather_df.index.union(usgs_df.index)
    ).ffill()
    weather_on_grid = weather_reindexed.reindex(usgs_df.index)
    return usgs_df.join(weather_on_grid)
