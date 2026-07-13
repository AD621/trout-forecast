"""Shared feature/label definitions for training and live prediction -- kept
in one place so both stages build the exact same inputs.
"""

from __future__ import annotations

import pandas as pd

# Columns that are functions of the label itself (see src/scoring/composite.py)
# -- these must never be used as model inputs, or the model just learns to
# reverse our own scoring formula instead of predicting from real conditions.
LABEL_DERIVED_COLUMNS = {
    "score_water_temp",
    "score_discharge",
    "score_stability",
    "score_dissolved_oxygen",
    "score_turbidity",
    "composite_score",
}

# Not a feature: the target itself, and a column that's constant within any
# single station's file (so it carries no information for a per-station model).
NON_FEATURE_COLUMNS = {"condition", "station"}

CATEGORICAL_COLUMNS = ["season", "time_of_day"]

CONDITION_ORDER = ["Poor", "Fair", "Good"]

# Synthetic feature added at training time (see src/modeling/train.py) so one
# model can answer "what will conditions be at any of these horizons" instead
# of predicting only one fixed number of hours ahead.
HOURS_AHEAD_COLUMN = "hours_ahead"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns in `df` usable as model input for this station."""
    excluded = LABEL_DERIVED_COLUMNS | NON_FEATURE_COLUMNS
    return [c for c in df.columns if c not in excluded]


def prepare_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Return the feature matrix, with season/time_of_day cast to pandas
    `category` dtype so XGBoost's native categorical support handles them
    without manual one-hot encoding.

    `discharge_stable_1hr` comes in as a nullable Int64 (0/1/<NA>) -- cast to
    float64 so NaN survives as an ordinary float NaN, which XGBoost's
    missing-value handling already understands (consistent with the
    "missing is neutral, not penalized" rule used everywhere else in this
    project).
    """
    feature_columns = feature_columns or get_feature_columns(df)
    X = df[feature_columns].copy()
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    if "discharge_stable_1hr" in X.columns:
        X["discharge_stable_1hr"] = X["discharge_stable_1hr"].astype("float64")
    return X


def encode_condition(condition: pd.Series) -> pd.Series:
    """Good/Fair/Poor -> 2/1/0, per CONDITION_ORDER."""
    mapping = {label: i for i, label in enumerate(CONDITION_ORDER)}
    return condition.map(mapping)
