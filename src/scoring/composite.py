"""Composite fishing-*conditions* score -> Good/Fair/Poor label (the training target).

This label measures whether the water is safe and comfortable to fish and
wade in, and within trout's physiological comfort zone (temperature, flow,
oxygen, clarity) -- not how likely fish are to bite. There's no catch-rate,
creel, or angler-report data anywhere in this pipeline, so bite quality
isn't something this project predicts. Keep that distinction in any
downstream UI copy.

Scoring is station-specific: each station has different instrumentation and
hydrological character, so scores are built from station-specific component
scales rather than one shared formula. Every component comes from a USGS
water-quality/hydrology reading -- weather (including cloud cover, which
affects fish behavior rather than water conditions) is left out of the
score and only used as a model feature during training.

Missing readings get a neutral score equal to half of that component's max
points (not zero, not full credit), applied consistently across every
component on both stations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Component scores (each returns a float Series on the component's 0..max scale)
# ---------------------------------------------------------------------------

def score_water_temp(temp_c: pd.Series) -> pd.Series:
    """0-3 pts. Optimal (3 pts) 10-18C, tapering off toward the extremes."""
    conditions = [
        (temp_c >= 10) & (temp_c <= 18),
        ((temp_c >= 7) & (temp_c < 10)) | ((temp_c > 18) & (temp_c <= 21)),
        ((temp_c >= 4) & (temp_c < 7)) | ((temp_c > 21) & (temp_c <= 24)),
    ]
    choices = [3, 2, 1]
    return pd.Series(
        np.select(conditions, choices, default=0), index=temp_c.index, dtype=float
    ).where(temp_c.notna(), 1.5)


def score_buford_discharge(discharge_cfs: pd.Series) -> pd.Series:
    """0-3 pts. Wadeable range 600-1800 CFS scores highest."""
    conditions = [
        (discharge_cfs >= 600) & (discharge_cfs <= 1800),
        ((discharge_cfs >= 300) & (discharge_cfs < 600))
        | ((discharge_cfs > 1800) & (discharge_cfs <= 2500)),
        ((discharge_cfs >= 100) & (discharge_cfs < 300))
        | ((discharge_cfs > 2500) & (discharge_cfs <= 4000)),
    ]
    choices = [3, 2, 1]
    return pd.Series(
        np.select(conditions, choices, default=0), index=discharge_cfs.index, dtype=float
    ).where(discharge_cfs.notna(), 1.5)


def score_buford_stability(discharge_stable_1hr: pd.Series) -> pd.Series:
    """0-1 pt. 1 if the 1hr discharge delta was under the stability threshold."""
    stable = discharge_stable_1hr.astype("float64")  # pd.NA -> np.nan
    return stable.fillna(0.5)


def score_buford_do(do_mgl: pd.Series) -> pd.Series:
    """0-2 pts. Ideal >= 8 mg/L."""
    conditions = [do_mgl >= 8, (do_mgl >= 6) & (do_mgl < 8)]
    choices = [2, 1]
    return pd.Series(
        np.select(conditions, choices, default=0), index=do_mgl.index, dtype=float
    ).where(do_mgl.notna(), 1.0)


def score_medlock_discharge(discharge_cfs: pd.Series) -> pd.Series:
    """0-2 pts. Wadeable-range gradient, same idea as Buford's discharge score.

    Thresholds come from this station's actual 2023-2026 daily-discharge
    distribution: 10th pct ~1100 CFS, 25th ~1300, median ~1720, 75th ~2440,
    90th ~3280 CFS.
    """
    conditions = [
        (discharge_cfs >= 800) & (discharge_cfs <= 2000),
        ((discharge_cfs >= 400) & (discharge_cfs < 800))
        | ((discharge_cfs > 2000) & (discharge_cfs <= 3500)),
    ]
    choices = [2, 1]
    return pd.Series(
        np.select(conditions, choices, default=0), index=discharge_cfs.index, dtype=float
    ).where(discharge_cfs.notna(), 1.0)


def score_medlock_turbidity(turbidity_fnu: pd.Series) -> pd.Series:
    """0-2 pts. Ideal <= 5 NTU."""
    conditions = [turbidity_fnu <= 5, (turbidity_fnu > 5) & (turbidity_fnu <= 15)]
    choices = [2, 1]
    return pd.Series(
        np.select(conditions, choices, default=0), index=turbidity_fnu.index, dtype=float
    ).where(turbidity_fnu.notna(), 1.0)


# ---------------------------------------------------------------------------
# Station composite scores + classification
# ---------------------------------------------------------------------------

BUFORD_THRESHOLDS = {"Good": (7, 9), "Fair": (4, 6), "Poor": (0, 3)}
MEDLOCK_THRESHOLDS = {"Good": (5, 7), "Fair": (3, 4), "Poor": (0, 2)}


def _classify(score: pd.Series, thresholds: dict) -> pd.Series:
    conditions = [
        (score >= lo) & (score <= hi) for lo, hi in thresholds.values()
    ]
    return pd.Series(
        np.select(conditions, list(thresholds.keys()), default="Poor"),
        index=score.index,
    )


def score_and_classify_buford(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_water_temp"] = score_water_temp(df["water_temp_c"])
    df["score_discharge"] = score_buford_discharge(df["discharge_cfs"])
    df["score_stability"] = score_buford_stability(df["discharge_stable_1hr"])
    df["score_dissolved_oxygen"] = score_buford_do(df["dissolved_oxygen_mgl"])
    df["composite_score"] = (
        df["score_water_temp"]
        + df["score_discharge"]
        + df["score_stability"]
        + df["score_dissolved_oxygen"]
    )
    df["condition"] = _classify(df["composite_score"], BUFORD_THRESHOLDS)
    return df


def score_and_classify_medlock(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_water_temp"] = score_water_temp(df["water_temp_c"])
    df["score_discharge"] = score_medlock_discharge(df["discharge_cfs"])
    df["score_turbidity"] = score_medlock_turbidity(df["turbidity_fnu"])
    df["composite_score"] = (
        df["score_water_temp"] + df["score_discharge"] + df["score_turbidity"]
    )
    df["condition"] = _classify(df["composite_score"], MEDLOCK_THRESHOLDS)
    return df


SCORERS = {
    "buford_dam": score_and_classify_buford,
    "medlock_bridge": score_and_classify_medlock,
}


def score_and_classify(df: pd.DataFrame, station_key: str) -> pd.DataFrame:
    return SCORERS[station_key](df)
