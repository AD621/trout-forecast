"""USGS instantaneous-values ingestion.

Pulls long-format continuous time series from the USGS Water Data API (via
`dataretrieval.waterdata.get_continuous`) and pivots to one row per timestamp,
one column per parameter. Used for both historical backfill and daily live
pulls, so the date range is always a caller-supplied argument.
"""

from __future__ import annotations

import pandas as pd
from dataretrieval import waterdata

from src.config import STATIONS


def fetch_and_pivot(station_key: str, start: str, end: str) -> pd.DataFrame:
    """Fetch USGS instantaneous values for one station and pivot to wide format.

    Parameters
    ----------
    station_key : one of the keys in ``src.config.STATIONS`` (e.g. "buford_dam")
    start, end : "YYYY-MM-DD" date strings (UTC), passed straight to the API

    Returns
    -------
    DataFrame indexed by UTC timestamp, one column per parameter (named per
    STATIONS[station_key]["parameters"]), plus a ``station`` column. Missing
    readings (sensor offline, not-yet-installed, etc.) are left as NaN --
    scoring downstream treats them as neutral rather than penalized.
    """
    station = STATIONS[station_key]
    param_codes = list(station["parameters"].keys())

    long_df, _meta = waterdata.get_continuous(
        monitoring_location_id=f"USGS-{station['usgs_id']}",
        parameter_code=param_codes,
        time=f"{start}/{end}",
    )

    if long_df.empty:
        wide = pd.DataFrame(columns=list(station["parameters"].values()))
        wide.index = pd.DatetimeIndex([], name="datetime", tz="UTC")
        wide["station"] = station_key
        return wide

    wide = long_df.pivot_table(
        index="time", columns="parameter_code", values="value", aggfunc="first"
    )
    wide = wide.rename(columns=station["parameters"])
    wide.index.name = "datetime"
    wide = wide.sort_index()
    wide["station"] = station_key
    return wide
