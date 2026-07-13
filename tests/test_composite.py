import numpy as np
import pandas as pd

from src.scoring.composite import score_and_classify_buford, score_and_classify_medlock


def _buford_frame(**overrides):
    base = {
        "water_temp_c": [14.0],
        "discharge_cfs": [1000.0],
        "dissolved_oxygen_mgl": [9.0],
        "discharge_stable_1hr": pd.array([1], dtype="Int64"),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_buford_good_conditions_score_high():
    df = score_and_classify_buford(_buford_frame())
    assert df.loc[0, "composite_score"] == 9.0
    assert df.loc[0, "condition"] == "Good"


def test_buford_missing_readings_are_neutral_not_penalized():
    df = _buford_frame(
        water_temp_c=[np.nan],
        dissolved_oxygen_mgl=[np.nan],
        discharge_stable_1hr=pd.array([pd.NA], dtype="Int64"),
    )
    out = score_and_classify_buford(df)
    assert out.loc[0, "score_water_temp"] == 1.5
    assert out.loc[0, "score_dissolved_oxygen"] == 1.0
    assert out.loc[0, "score_stability"] == 0.5


def test_buford_poor_conditions():
    df = _buford_frame(
        water_temp_c=[28.0],
        discharge_cfs=[9000.0],
        dissolved_oxygen_mgl=[3.0],
        discharge_stable_1hr=pd.array([0], dtype="Int64"),
    )
    out = score_and_classify_buford(df)
    assert out.loc[0, "condition"] == "Poor"


def test_medlock_flood_is_poor():
    df = pd.DataFrame(
        {
            "water_temp_c": [28.0],
            "discharge_cfs": [4500.0],
            "turbidity_fnu": [20.0],
        }
    )
    out = score_and_classify_medlock(df)
    assert out.loc[0, "score_discharge"] == 0
    assert out.loc[0, "condition"] == "Poor"


def test_medlock_normal_flow_is_good():
    df = pd.DataFrame(
        {
            "water_temp_c": [14.0],
            "discharge_cfs": [1200.0],
            "turbidity_fnu": [3.0],
        }
    )
    out = score_and_classify_medlock(df)
    assert out.loc[0, "composite_score"] == 7.0
    assert out.loc[0, "condition"] == "Good"
