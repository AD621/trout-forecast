# Code Documentation — trout-forecast

Technical reference for every module in this codebase: what each function does, its
signature, and how the pieces connect. For *why* the project is built this way, see
[`Project_Report.md`](Project_Report.md); for setup/usage instructions, see
[`README.md`](README.md). This document covers *what the code does*, function by function.

## Contents

- [Repository layout](#repository-layout)
- [Data flow overview](#data-flow-overview)
- [`src/config.py`](#srcconfigpy)
- [`src/ingestion/usgs.py`](#srcingestionusgspy)
- [`src/ingestion/weather.py`](#srcingestionweatherpy)
- [`src/features/engineering.py`](#srcfeaturesengineeringpy)
- [`src/scoring/composite.py`](#srcscoringcompositepy)
- [`src/modeling/features.py`](#srcmodelingfeaturespy)
- [`src/modeling/train.py`](#srcmodelingtrainpy)
- [`src/pipeline/backfill.py`](#srcpipelinebackfillpy)
- [`src/pipeline/predict.py`](#srcpipelinepredictpy)
- [`app/app.py`](#appapppy)
- [`tests/`](#tests)
- [Data schemas](#data-schemas)

---

## Repository layout

```
trout-forecast/
  data/
    raw/            gitignored -- not currently used to persist raw pulls
    processed/      gitignored -- backfill.py output, one parquet per station
    samples/        small committed CSV samples for quick inspection
    live/
      latest_forecast.json   predict.py output, read by app.py
  models/
    {station}_xgb.json            train.py output -- saved XGBoost booster
    {station}_feature_spec.json   train.py output -- feature list + category encodings
  src/
    config.py
    ingestion/{usgs.py, weather.py}
    features/engineering.py
    scoring/composite.py
    modeling/{features.py, train.py}
    pipeline/{backfill.py, predict.py}
  app/app.py
  tests/{test_engineering.py, test_composite.py}
  .github/workflows/daily_forecast.yml
```

## Data flow overview

Two pipelines share the same ingestion, feature-engineering, and (for backfill/training)
scoring code:

```
Historical:  fetch_and_pivot -> join_weather_to_usgs -> engineer_features
             -> score_and_classify -> data/processed/*.parquet -> train_station -> models/*

Live:        fetch_and_pivot -> join_weather_to_usgs -> engineer_features
             -> (forward-fill, take last row = anchor) -> predict_timeline
             -> compute_daily_badge + compute_current_summary
             -> data/live/latest_forecast.json -> app.py (display only)
```

The historical path additionally computes `score_and_classify` to build the training
label; the live path does not, since there's nothing to score yet at prediction time —
that's the model's job.

---

## `src/config.py`

Static configuration, no functions. All station metadata (USGS ID, coordinates,
parameter-code-to-column mapping, and reference URLs) lives here as the single source of
truth other modules import from.

| Name | Type | Description |
|---|---|---|
| `STATIONS` | `dict` | Keyed by `"buford_dam"` / `"medlock_bridge"`. Each value has `usgs_id`, `usgs_site_name`, `lat`, `lon`, `parameters` (USGS parameter code → output column name), and `usgs_url` (link to the station's live monitoring page). |
| `DAM_SCHEDULE_URL` | `str` | USACE hydropower generation/release schedule, shown in the app under Buford Dam. |
| `OPEN_METEO_HISTORICAL_URL` / `OPEN_METEO_FORECAST_URL` | `str` | Base URLs for the two Open-Meteo endpoints. |
| `OPEN_METEO_HOURLY_VARS` | `list[str]` | `["temperature_2m", "surface_pressure", "precipitation", "cloud_cover"]` — requested from both endpoints. |

---

## `src/ingestion/usgs.py`

### `fetch_and_pivot(station_key: str, start: str, end: str) -> pd.DataFrame`

Pulls instantaneous values from the USGS Water Data API via
`dataretrieval.waterdata.get_continuous` for every parameter listed in
`STATIONS[station_key]["parameters"]`, over `[start, end]` (`"YYYY-MM-DD"` strings).

The API returns long format (one row per parameter per timestamp); this function pivots to
wide format (one row per timestamp, one column per parameter, columns renamed per the
station's `parameters` mapping) and adds a `station` column. Returns a DataFrame indexed by
a UTC `DatetimeIndex` named `datetime`. Missing readings (sensor offline, not yet
installed, etc.) are left as `NaN` — nothing in this function fills or drops them; that's
handled downstream by scoring (Section on `composite.py`) or by `predict.py`'s
forward-fill. If the API returns no rows for the requested window, returns an empty
DataFrame with the correct columns and an empty UTC `DatetimeIndex` rather than raising.

---

## `src/ingestion/weather.py`

### `fetch_weather_historical(lat, lon, start, end) -> pd.DataFrame`

Hourly historical weather from Open-Meteo's archive endpoint (ERA5 reanalysis). **Caveat**:
this endpoint has ~5 days of processing latency; requesting an `end` date within that
window returns forecast-model output for the gap, not observed data. Only used by
`backfill.py`, where `end` is always well in the past.

### `fetch_weather_forecast(lat, lon, past_days=2, forecast_days=3) -> pd.DataFrame`

Hourly weather from Open-Meteo's forecast endpoint: `past_days` days of recent actuals plus
`forecast_days` days of forward forecast. Used by `predict.py` for anything near "now,"
since the archive endpoint can't be trusted there.

Both functions return a DataFrame indexed by a UTC `DatetimeIndex` (column `time`), with one
column per `OPEN_METEO_HOURLY_VARS` entry.

### `join_weather_to_usgs(usgs_df, weather_df) -> pd.DataFrame`

Forward-fills the hourly weather DataFrame onto the finer 15-minute USGS grid (an hourly
reading applies to every USGS timestamp until the next hourly reading arrives) and joins
the result onto `usgs_df`. Deliberately not interpolated — precipitation and cloud cover
are not smooth continuous signals, so linear interpolation between hourly readings would
invent physically meaningless intermediate values.

---

## `src/features/engineering.py`

All functions here assume a DataFrame indexed by a monotonic, UTC-aware `DatetimeIndex`
(the output of `fetch_and_pivot` joined via `join_weather_to_usgs`). Each `add_*` function
takes and returns a DataFrame, adding one or more columns, so they can be chained.

### `add_discharge_features(df, stability_threshold_cfs_hr=200.0) -> pd.DataFrame`

Adds `discharge_delta_1hr_cfs`, `discharge_delta_3hr_cfs` (rate of change, computed via the
private `_shift_by_time` helper — see below), and `discharge_stable_1hr` (nullable `Int64`:
1 if the 1-hour delta's magnitude is under the threshold, `pd.NA` if the delta itself is
`NaN`).

### `_shift_by_time(series, offset) -> pd.Series` (private)

Returns `series`'s value as of `offset` before each timestamp, by shifting the index
forward by `offset` and reindexing onto the original index. Timestamps with no exact grid
match `offset` earlier (i.e., a gap in the 15-minute grid) come back `NaN` rather than a
stale/incorrect value — this is why discharge deltas use this instead of `.diff(n)`, which
would silently assume a fixed row-to-row spacing.

### `add_water_temp_rolling(df) -> pd.DataFrame`

Adds `water_temp_roll_3hr`, `water_temp_roll_12hr`, `water_temp_roll_24hr` — time-based
rolling means (`min_periods=1`, so a single valid reading is enough to produce a value).

### `add_season_flag(df) -> pd.DataFrame`

Adds `season` (`Winter`/`Spring`/`Summer`/`Fall`) from the calendar month, via the module
constant `SEASON_BY_MONTH`.

### `add_time_of_day_flag(df) -> pd.DataFrame`

Adds `time_of_day` (`Night`/`Morning`/`Afternoon`/`Evening`). Converts the index to
`LOCAL_TZ` (`"America/New_York"`) before bucketing by hour — bins are
`[-1, 4, 10, 16, 21, 23]` with labels `["Night", "Morning", "Afternoon", "Evening",
"Night"]` (Night appears at both ends so the 10pm-4:59am wraparound is handled by one bin
list rather than needing modular arithmetic).

### `generate_stocking_weeks(year: int) -> set[pd.Timestamp]`

Returns the set of Monday-anchored week-start dates containing a GA DNR trout stocking for
the given year: every week from April 1 through July 31, the two weeks immediately before
Labor Day (computed as the first Monday in September), and one placeholder week containing
October 1 for the undocumented fall stocking.

### `add_stocked_week_flag(df) -> pd.DataFrame`

Adds `stocked_week_flag` (0/1). Builds the union of `generate_stocking_weeks(y)` for every
year present in `df` plus one year on either side (so a week spanning a year boundary,
e.g. late December, isn't missed), converts the index to local time, computes each row's
Monday, and checks membership.

### `to_local(df, tz=LOCAL_TZ) -> pd.DataFrame`

**Display-only** — not part of the ingestion/feature/scoring pipeline, and not used by
`predict.py`'s scoring path. Returns a copy of `df` with the index converted to `tz` and a
new `local_time` column (12-hour clock string, e.g. `"2026-07-11 06:00 PM"`) inserted as
the first column. Everything else in this codebase stays in UTC because Eastern time
repeats an hour at every fall-back DST transition, which can silently corrupt time-based
rolling/join operations if used as the working index — this function exists purely so a
human can read a DataFrame in local time without the rest of the pipeline paying that cost.

### `engineer_features(df) -> pd.DataFrame`

Orchestrator: applies `add_discharge_features`, `add_water_temp_rolling`,
`add_season_flag`, `add_time_of_day_flag`, `add_stocked_week_flag` in sequence. This is the
one function both `backfill.py` and `predict.py` call — training and live features are
guaranteed to be built identically because they go through this same function.

---

## `src/scoring/composite.py`

Builds the `condition` (Good/Fair/Poor) training label from a feature-engineered
DataFrame. Not used at live-prediction time — only for historical backfill, where a label
is needed to train against.

### Component score functions

Each takes one `pd.Series` (a raw reading) and returns a `pd.Series` of the same length on
a fixed `0..max` point scale, with `NaN` inputs mapped to a neutral score (fixed at exactly
half the component's max, applied via `.where(series.notna(), neutral_value)`):

| Function | Scale | Best case | Neutral (NaN) value |
|---|---|---|---|
| `score_water_temp(temp_c)` | 0–3 | 10–18°C | 1.5 |
| `score_buford_discharge(discharge_cfs)` | 0–3 | 600–1800 cfs | 1.5 |
| `score_buford_stability(discharge_stable_1hr)` | 0–1 | stable flag = 1 | 0.5 |
| `score_buford_do(do_mgl)` | 0–2 | ≥ 8 mg/L | 1.0 |
| `score_medlock_discharge(discharge_cfs)` | 0–2 | 800–2000 cfs | 1.0 |
| `score_medlock_turbidity(turbidity_fnu)` | 0–2 | ≤ 5 NTU | 1.0 |

`score_water_temp` is shared between both stations. The discharge-scoring functions read
their wadeable range from the module constants `BUFORD_WADEABLE_CFS = (600, 1800)` and
`MEDLOCK_WADEABLE_CFS = (800, 2000)` rather than hardcoding the bounds twice, and those same
constants back `WADEABLE_MAX_CFS` (a `{station_key: upper_bound}` dict), which `app.py`
imports directly to drive the dashboard's "High water level" warning — so the warning
threshold and the scoring threshold can never drift apart.

### `_classify(score, thresholds) -> pd.Series` (private)

Maps a numeric `composite_score` Series to a Good/Fair/Poor label given a `thresholds` dict
of `{label: (lo, hi)}` ranges (see `BUFORD_THRESHOLDS` / `MEDLOCK_THRESHOLDS`), via
`np.select`. Defaults to `"Poor"` if no range matches.

### `score_and_classify_buford(df) -> pd.DataFrame` / `score_and_classify_medlock(df) -> pd.DataFrame`

Each computes its station's component scores, sums them into `composite_score`, and adds
`condition` via `_classify` against `BUFORD_THRESHOLDS` (`{"Good": (7,9), "Fair": (4,6),
"Poor": (0,3)}`) or `MEDLOCK_THRESHOLDS` (`{"Good": (5,7), "Fair": (3,4), "Poor": (0,2)}`)
respectively. Returns a copy of `df` with the new `score_*`, `composite_score`, and
`condition` columns added.

### `score_and_classify(df, station_key) -> pd.DataFrame`

Dispatcher — looks up `station_key` in the module-level `SCORERS` dict and calls the
matching function above. This is the one entry point `backfill.py` actually calls.

---

## `src/modeling/features.py`

Shared feature/label definitions used by both `train.py` and `predict.py`, so the two
stages can never build inconsistent inputs.

| Name | Description |
|---|---|
| `LABEL_DERIVED_COLUMNS` | The `score_*`/`composite_score` columns — functions of the label itself, must never be model inputs (see the label-leakage discussion in `Project_Report.md` Section 5.1). |
| `NON_FEATURE_COLUMNS` | `{"condition", "station"}` — the target itself, and a column that's constant per station file. |
| `CATEGORICAL_COLUMNS` | `["season", "time_of_day"]`. |
| `CONDITION_ORDER` | `["Poor", "Fair", "Good"]` — fixes the integer encoding used everywhere (Poor=0, Fair=1, Good=2). |
| `HOURS_AHEAD_COLUMN` | `"hours_ahead"` — the synthetic horizon feature added at training time. |

### `get_feature_columns(df) -> list[str]`

Returns every column in `df` except `LABEL_DERIVED_COLUMNS | NON_FEATURE_COLUMNS`.

### `prepare_features(df, feature_columns=None) -> pd.DataFrame`

Selects `feature_columns` (or `get_feature_columns(df)` if not given), casts
`CATEGORICAL_COLUMNS` to pandas `category` dtype (so XGBoost's native categorical support
applies without manual one-hot encoding), and casts `discharge_stable_1hr` from nullable
`Int64` to `float64` (so `pd.NA` becomes ordinary `np.nan`, which XGBoost's missing-value
handling understands).

### `encode_condition(condition: pd.Series) -> pd.Series`

Maps `Good`/`Fair`/`Poor` strings to `2`/`1`/`0` per `CONDITION_ORDER`.

---

## `src/modeling/train.py`

Run as `python -m src.modeling.train`. Trains and saves one XGBoost model per station.

### `build_horizon_dataset(df, base_feature_columns) -> (pd.DataFrame, pd.Series)`

For each horizon `h` in `1..MAX_HORIZON_HOURS` (48): shifts the encoded `condition` label
backward by `h * PERIODS_PER_HOUR` (4, the 15-minute grid) rows, drops rows with no future
label to pair with (the tail of the historical record), takes a copy of `base_feature_columns`
for the surviving rows, adds a constant `hours_ahead = h` column, and appends the frame to a
list. Concatenates all 48 horizon-frames together. Returns the combined feature DataFrame
and the combined (still-encoded) label Series — approximately 5 million rows per station
(48 × ~105K, minus the dropped tail rows).

### `train_station(station_key: str) -> dict`

1. Loads `data/processed/{station_key}_processed.parquet`.
2. Builds the horizon-expanded dataset via `build_horizon_dataset`, then
   `prepare_features`.
3. Time-based split on `TEST_START` (`"2025-08-01"`): everything before trains, everything
   on/after tests. Not a random split — this is autocorrelated time-series data.
4. Computes `sample_weight` via `sklearn.utils.compute_sample_weight("balanced", y_train)`
   to counter class imbalance.
5. Fits an `xgb.XGBClassifier` (`multi:softmax`, 300 estimators, max depth 6, learning rate
   0.1, `tree_method="hist"`, `n_jobs=-1` — see the note below).
6. Evaluates on the test split: overall accuracy, macro F1, confusion matrix, feature
   importances, and accuracy at each of `SPOT_CHECK_HORIZONS` (`[1, 6, 12, 24, 48]`).
7. Saves `models/{station_key}_xgb.json` (the booster) and
   `models/{station_key}_feature_spec.json` (feature column list, the exact categorical
   levels seen at train time, `CONDITION_ORDER`, `TEST_START`, `MAX_HORIZON_HOURS`, and
   `HOURS_AHEAD_COLUMN` — everything `predict.py` needs to reconstruct an identical feature
   matrix at inference time).

**`n_jobs=-1` is not cosmetic**: leaving it at XGBoost's default resulted in near
single-threaded training on a 12-core machine (~35 minutes per run); setting it explicitly
brought training down to ~2.5 minutes per station.

### `main()`

Calls `train_station` for every key in `STATIONS`.

---

## `src/pipeline/backfill.py`

Run as `python -m src.pipeline.backfill`. Builds the historical labeled dataset used for
training.

### `build_station_dataset(station_key, start, end) -> pd.DataFrame`

`fetch_and_pivot` → `fetch_weather_historical` → `join_weather_to_usgs` →
`engineer_features` → `score_and_classify`. Returns the fully labeled, feature-engineered
DataFrame for one station.

### `run_backfill(start="2023-01-01", end="2026-01-01") -> dict[str, pd.DataFrame]`

Calls `build_station_dataset` for every station, writes each result to
`data/processed/{station_key}_processed.parquet`, prints the resulting `condition` class
balance, and returns `{station_key: DataFrame}`.

---

## `src/pipeline/predict.py`

Run as `python -m src.pipeline.predict`. Produces the artifact the Streamlit app reads.

### `load_model_and_spec(station_key) -> (xgb.XGBClassifier, dict)`

Loads `models/{station_key}_xgb.json` into a fresh `XGBClassifier` and parses
`models/{station_key}_feature_spec.json`.

### `build_live_features(station_key) -> pd.DataFrame`

Pulls the last ~60 hours of USGS data (`fetch_and_pivot`) and Open-Meteo forecast data
(`fetch_weather_forecast(past_days=2, forecast_days=3)`), joins and feature-engineers them
via the same `join_weather_to_usgs` / `engineer_features` used in training.

### `predict_timeline(model, spec, anchor) -> list[dict]`

Builds `spec["max_horizon_hours"]` (48) copies of the anchor row's feature columns
(everything in `spec["feature_columns"]` except `HOURS_AHEAD_COLUMN`), sets `hours_ahead`
to `1, 2, ..., 48` across those copies, re-casts `season`/`time_of_day` to the *exact*
category sets recorded in `spec["categorical_categories"]` (so a live prediction can't be
encoded differently than training), and calls `model.predict_proba` once on the whole
batch. Returns one dict per hour: `timestamp_utc` (the anchor's timestamp plus `h` hours),
`condition` (argmax label), and `probabilities` (all three class probabilities, rounded to
3 decimals).

### `compute_daily_badge(timeline) -> str | None`

Filters `timeline` to rows whose local hour falls between `DAYLIGHT_START_HOUR` (5) and
`DAYLIGHT_END_HOUR` (21) inclusive, averages those rows' probability vectors elementwise,
and returns the argmax label of the average. Returns `None` if no rows fall in the window.

### `compute_current_summary(anchor) -> dict`

Reads the single anchor row directly (not an average) for `water_temp_c`, `discharge_cfs`,
`cloud_cover_pct` (from the `cloud_cover` column), `precipitation_in` (from `precipitation`),
`conductance_uscm`, and whichever of `dissolved_oxygen_mgl` / `turbidity_fnu` the station
has (`EXTRA_SUMMARY_COLUMNS`). Values are rounded for display (1 decimal for
temperature/DO/turbidity, whole numbers for discharge/cloud-cover/conductance, 2 decimals
for precipitation).

### `predict_station(station_key) -> dict`

Orchestrator for one station: builds live features, **forward-fills the entire DataFrame
before selecting the last row as the anchor** (slower-reporting sensors — water temp,
conductance, turbidity, DO — commonly lag discharge by 15min–1hr, so the single freshest
row is exactly where gaps are most likely; the trained model has no built-in neutral
handling for missing values the way the rule-based composite score does, so an unfilled
anchor can flip a prediction based on nothing but sensor lag rather than a real change in
conditions), then calls `predict_timeline`, `compute_daily_badge`, and
`compute_current_summary`. Returns a dict with `station`, `anchor_time_utc`,
`current_summary`, `daily_badge`, and `hourly_timeline`.

### `main()`

Calls `predict_station` for every station, assembles `{"generated_at_utc": ..., "stations":
{...}}`, and writes it to `data/live/latest_forecast.json`.

---

## `app/app.py`

Run as `streamlit run app/app.py`. Reads `data/live/latest_forecast.json` and displays it
— performs no computation of its own beyond formatting. Imports `STATIONS` and
`DAM_SCHEDULE_URL` from `src.config` and `WADEABLE_MAX_CFS` from `src.scoring.composite`;
since the app lives in `app/` rather than the project root, it inserts the project root
onto `sys.path` before those imports so they resolve regardless of the working directory
Streamlit was launched from.

| Function | Purpose |
|---|---|
| `fmt_ampm(ts)` | 12-hour time without a leading zero (`"6:05 PM"`), cross-platform (avoids the Unix-only `%-I` strftime flag, which raises on Windows). |
| `load_forecast()` | Reads and parses `data/live/latest_forecast.json`; returns `None` if the file doesn't exist yet. |
| `render_badge(condition)` | The colored "Good/Fair/Poor today" pill, styled via `BADGE_STYLE`. |
| `render_summary(summary, station_key)` | The "Right now" 3-column stat grid (`st.metric` per field, in `SUMMARY_FIELDS` order, skipping any field the station doesn't have). Adds a "⚠ High water level" caption under the discharge metric if its value exceeds `WADEABLE_MAX_CFS[station_key]`. |
| `render_timeline(timeline)` | The 48-hour colored strip. Plain HTML/CSS `<div>` blocks in a flex row (colored via `CONDITION_COLORS`), not a chart library — see the docstring in the code and `Project_Report.md` Section 7.1 for why. |
| `render_links(station_key)` | Markdown links: the station's `usgs_url` (always), plus `DAM_SCHEDULE_URL` (Buford Dam only). |
| `render_station(station_key, station_data)` | Orchestrates one station's whole section: subheader, badge, summary, links, timeline. |
| `main()` | Page config, title, "Refreshes daily at 6:00 AM EST" / "Updated {time}" captions, loops `render_station` over every station in the loaded JSON, and the closing "Good means..." scope-reminder caption. |

---

## `tests/`

- **`test_engineering.py`** — `generate_stocking_weeks` (April–July coverage, Labor Day
  math, fall placeholder), `add_stocked_week_flag` (on/off weeks), and `to_local`
  (UTC→Eastern conversion and 12-hour formatting, confirming the original DataFrame is left
  untouched).
- **`test_composite.py`** — Buford scoring (good conditions, missing-data neutrality,
  poor conditions) and Medlock scoring (flood-level discharge, normal flow), using small
  hand-built DataFrames rather than the full pipeline.

Run with `python -m pytest tests/ -q` from the project root.

---

## Data schemas

### `data/processed/{station}_processed.parquet` (backfill output / training input)

One row per 15-minute timestamp (UTC `DatetimeIndex`), columns: the raw station-specific
readings (`water_temp_c`, `discharge_cfs`, `conductance_uscm`, plus `dissolved_oxygen_mgl`
for Buford or `turbidity_fnu` for Medlock), `station`, the four Open-Meteo columns
(`temperature_2m`, `surface_pressure`, `precipitation`, `cloud_cover`), the engineered
columns from `engineer_features` (`discharge_delta_1hr_cfs`, `discharge_delta_3hr_cfs`,
`discharge_stable_1hr`, `water_temp_roll_3hr/12hr/24hr`, `season`, `time_of_day`,
`stocked_week_flag`), and the scoring columns from `score_and_classify`
(`score_water_temp`, `score_discharge`, `score_stability` or `score_turbidity` depending on
station, `score_dissolved_oxygen` for Buford only, `composite_score`, `condition`).

### `models/{station}_feature_spec.json`

```json
{
  "feature_columns": ["water_temp_c", "discharge_cfs", ..., "hours_ahead"],
  "categorical_categories": {
    "season": ["Fall", "Spring", "Summer", "Winter"],
    "time_of_day": ["Afternoon", "Evening", "Morning", "Night"]
  },
  "condition_order": ["Poor", "Fair", "Good"],
  "test_start": "2025-08-01",
  "max_horizon_hours": 48,
  "hours_ahead_column": "hours_ahead"
}
```

### `data/live/latest_forecast.json` (predict.py output / app.py input)

```json
{
  "generated_at_utc": "2026-07-13T23:45:00+00:00",
  "stations": {
    "buford_dam": {
      "station": "buford_dam",
      "anchor_time_utc": "2026-07-13T23:45:00+00:00",
      "current_summary": {
        "water_temp_c": 9.7, "discharge_cfs": 636, "cloud_cover_pct": 62,
        "precipitation_in": 0.0, "conductance_uscm": 76, "dissolved_oxygen_mgl": 5.7
      },
      "daily_badge": "Good",
      "hourly_timeline": [
        {
          "timestamp_utc": "2026-07-14T00:45:00+00:00",
          "condition": "Good",
          "probabilities": {"Poor": 0.002, "Fair": 0.365, "Good": 0.634}
        }
      ]
    },
    "medlock_bridge": { "...": "same shape, turbidity_fnu instead of dissolved_oxygen_mgl" }
  }
}
```
