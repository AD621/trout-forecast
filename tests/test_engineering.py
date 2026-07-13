import pandas as pd

from src.features.engineering import generate_stocking_weeks, add_stocked_week_flag, to_local


def test_generate_stocking_weeks_covers_april_through_july():
    weeks = generate_stocking_weeks(2024)
    d = pd.Timestamp("2024-04-01")
    while d <= pd.Timestamp("2024-07-31"):
        week_start = d - pd.Timedelta(days=d.weekday())
        assert week_start.normalize() in weeks
        d += pd.Timedelta(days=1)


def test_generate_stocking_weeks_labor_day_and_fall():
    weeks = generate_stocking_weeks(2024)
    # Labor Day 2024 is Sept 2 (Monday)
    labor_day = pd.Timestamp("2024-09-02")
    assert (labor_day - pd.Timedelta(weeks=1)) in weeks
    assert (labor_day - pd.Timedelta(weeks=2)) in weeks
    # fall placeholder week containing Oct 1, 2024 (a Tuesday) -> Monday Sept 30
    assert pd.Timestamp("2024-09-30") in weeks


def test_stocked_week_flag_off_season():
    idx = pd.DatetimeIndex(
        ["2024-01-15 12:00:00", "2024-05-15 12:00:00"], tz="UTC"
    )
    df = pd.DataFrame(index=idx)
    out = add_stocked_week_flag(df)
    assert out["stocked_week_flag"].tolist() == [0, 1]


def test_to_local_converts_and_formats_12hr():
    # 2026-07-11 22:00 UTC -> 2026-07-11 18:00 EDT (summer, UTC-4)
    idx = pd.DatetimeIndex(["2026-07-11 22:00:00"], tz="UTC")
    df = pd.DataFrame({"discharge_cfs": [636.0]}, index=idx)
    out = to_local(df)
    assert out["local_time"].iloc[0] == "2026-07-11 06:00 PM"
    assert out.index.tz.key == "America/New_York"
    # original frame is untouched (still UTC)
    assert str(df.index.tz) == "UTC"
