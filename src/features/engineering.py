"""Feature engineering on the pivoted, weather-joined USGS DataFrame.

All functions assume a DataFrame indexed by a monotonic, tz-aware (UTC)
DatetimeIndex, as produced by `src.ingestion.usgs.fetch_and_pivot` joined with
`src.ingestion.weather.join_weather_to_usgs`.
"""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "America/New_York"

SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


def _shift_by_time(series: pd.Series, offset: str) -> pd.Series:
    """Value of `series` as of `offset` before each timestamp.

    Shifts the index forward by `offset` and reindexes onto the original
    index, so exact grid matches line up. Timestamps with no exact match
    `offset` earlier (grid gaps) come back as NaN rather than a stale value.
    """
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(offset)
    return shifted.reindex(series.index)


def add_discharge_features(df: pd.DataFrame, stability_threshold_cfs_hr: float = 200.0) -> pd.DataFrame:
    """1hr/3hr discharge rate of change (CFS/hr) and a stability flag.

    Buford is a peaking hydro facility -- the rate of change often matters
    more than the absolute discharge value.
    """
    df = df.copy()
    prev_1hr = _shift_by_time(df["discharge_cfs"], "1h")
    prev_3hr = _shift_by_time(df["discharge_cfs"], "3h")

    df["discharge_delta_1hr_cfs"] = (df["discharge_cfs"] - prev_1hr) / 1.0
    df["discharge_delta_3hr_cfs"] = (df["discharge_cfs"] - prev_3hr) / 3.0
    df["discharge_stable_1hr"] = (
        df["discharge_delta_1hr_cfs"].abs() < stability_threshold_cfs_hr
    ).astype("Int64")
    df.loc[df["discharge_delta_1hr_cfs"].isna(), "discharge_stable_1hr"] = pd.NA
    return df


def add_water_temp_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """3hr / 12hr / 24hr rolling means of water temperature."""
    df = df.copy()
    for label, window in [("3hr", "3h"), ("12hr", "12h"), ("24hr", "24h")]:
        df[f"water_temp_roll_{label}"] = (
            df["water_temp_c"].rolling(window, min_periods=1).mean()
        )
    return df


def add_season_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season"] = df.index.month.map(SEASON_BY_MONTH)
    return df


def add_time_of_day_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Night/Morning/Afternoon/Evening, bucketed on local (US Eastern) time."""
    df = df.copy()
    local_hour = df.index.tz_convert(LOCAL_TZ).hour
    bins = pd.cut(
        local_hour,
        bins=[-1, 4, 10, 16, 21, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Night"],
        ordered=False,
    )
    df["time_of_day"] = bins
    return df


def generate_stocking_weeks(year: int) -> set[pd.Timestamp]:
    """Monday-anchored calendar weeks (as the Monday's date) containing a
    GA DNR trout stocking on this tailwater, for the given year.

    - Weekly from April 1 through July 31.
    - Twice more in the two weeks immediately before Labor Day.
    - Once in the fall (Sept/Oct) -- placeholder week; refine against the
      real GA DNR stocking calendar later.
    """
    weeks: set[pd.Timestamp] = set()

    def week_start(ts: pd.Timestamp) -> pd.Timestamp:
        return (ts - pd.Timedelta(days=ts.weekday())).normalize()

    # Weekly April 1 - July 31
    start = pd.Timestamp(year=year, month=4, day=1)
    end = pd.Timestamp(year=year, month=7, day=31)
    d = week_start(start)
    while d <= end:
        weeks.add(d)
        d += pd.Timedelta(weeks=1)

    # Labor Day = first Monday in September; two weeks before it
    sept1 = pd.Timestamp(year=year, month=9, day=1)
    labor_day = sept1 + pd.Timedelta(days=(7 - sept1.weekday()) % 7)
    weeks.add(week_start(labor_day - pd.Timedelta(weeks=1)))
    weeks.add(week_start(labor_day - pd.Timedelta(weeks=2)))

    # Fall stocking placeholder: week containing Oct 1
    weeks.add(week_start(pd.Timestamp(year=year, month=10, day=1)))

    return weeks


def add_stocked_week_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the timestamp's Monday-anchored week contains a
    stocking, per `generate_stocking_weeks`. Week-level granularity only --
    the day of week within a stocking week isn't known.
    """
    df = df.copy()
    years = df.index.year.unique()
    all_weeks: set[pd.Timestamp] = set()
    for y in years:
        for yy in (int(y) - 1, int(y), int(y) + 1):
            all_weeks |= generate_stocking_weeks(yy)

    local_dates = df.index.tz_convert(LOCAL_TZ)
    week_starts = (local_dates - pd.to_timedelta(local_dates.weekday, unit="D")).normalize()
    week_starts = week_starts.tz_localize(None)
    df["stocked_week_flag"] = pd.Index(week_starts).isin(all_weeks).astype(int)
    return df


def to_local(df: pd.DataFrame, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Return a copy of `df` for *viewing* in local (Eastern) time, 12-hour clock.

    Everything else in this pipeline computes in UTC on purpose -- Eastern
    time repeats an hour at every fall-back DST transition, which can
    silently break time-based rolling windows and forward-fill joins if used
    as the working index. This is a display-only conversion: call it right
    before viewing a DataFrame, not as part of the ingestion/feature/scoring
    pipeline.

    Adds a `local_time` column (12-hour clock, e.g. "2026-07-11 06:00 PM")
    and converts the index itself to Eastern, so the DataFrame still
    sorts/slices correctly if you keep working with it afterward.
    """
    out = df.copy()
    out.index = out.index.tz_convert(tz)
    out.index.name = "datetime_local"
    out.insert(0, "local_time", out.index.strftime("%Y-%m-%d %I:%M %p"))
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature set to a pivoted, weather-joined station DataFrame."""
    df = add_discharge_features(df)
    df = add_water_temp_rolling(df)
    df = add_season_flag(df)
    df = add_time_of_day_flag(df)
    df = add_stocked_week_flag(df)
    return df
