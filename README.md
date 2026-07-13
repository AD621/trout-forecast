# Chattahoochee River Trout Fishing Forecast

ML pipeline forecasting trout fishing *conditions* (Good/Fair/Poor) on the
Chattahoochee River tailwater, per station, using USGS hydrological sensors +
Open-Meteo weather.

## Scope: conditions, not bite quality

This predicts whether the water is safe and comfortable to fish/wade in, and
within trout's physiological comfort zone (temperature, flow, oxygen,
clarity) -- not how likely fish are to actually bite. There's no catch-rate,
creel-survey, or angler-report data anywhere in this pipeline, so bite
quality isn't something it can predict. "Good" means good conditions to go
fish in, not a guaranteed catch. Keep this distinction in any UI copy (see
`src/scoring/composite.py`'s module docstring for the same note in code).

Weather variables (including cloud cover, despite its effect on fish feeding
behavior) are joined into the dataset as model features only and excluded
from the composite score itself -- the score is built entirely from USGS
water-quality/hydrology readings, which directly measure conditions; weather
doesn't.

## Project structure

```
trout-forecast/
  data/
    raw/            # gitignored -- raw USGS pulls
    processed/      # gitignored -- labeled, feature-engineered parquet
    samples/        # small CSV samples, committed, for quick inspection
    live/
      latest_forecast.json  # written by predict.py, read by the Streamlit app
  models/
    {station}_xgb.json           # trained XGBoost model, per station
    {station}_feature_spec.json  # feature columns + category encodings fixed at train time
  src/
    config.py       # station IDs, verified coords, USGS parameter codes
    ingestion/
      usgs.py        # fetch_and_pivot(station_key, start, end)
      weather.py      # Open-Meteo historical/forecast pulls + forward-fill join
    features/
      engineering.py  # discharge deltas, rolling temp, season, time-of-day, stocking weeks
    scoring/
      composite.py    # station-specific point scoring -> Good/Fair/Poor label
    modeling/
      features.py     # shared feature/label definitions for train + predict
      train.py         # per-station XGBoost training (horizon-aware, 1-48h)
    pipeline/
      backfill.py     # historical backfill script
      predict.py       # live prediction pipeline
  app/
    app.py             # Streamlit dashboard -- reads data/live/latest_forecast.json only
  .github/workflows/
    daily_forecast.yml  # scheduled predict.py run + commit
  tests/
    test_engineering.py
    test_composite.py
```

## Setup and usage

```
pip install -r requirements.txt
```

**1. Historical backfill** -- pulls USGS 15-min data + Open-Meteo hourly
weather for both stations, 2023-01-01 through 2026-01-01, engineers
features, scores/classifies, and writes
`data/processed/{station}_processed.parquet`. Takes ~5-8 minutes (dominated
by the USGS API pull, ~135s/station for the full 3-year range).

```
python -m src.pipeline.backfill
```

**2. Model training** -- reads the backfilled parquet, trains one XGBoost
classifier per station that can forecast at any horizon from 1-48 hours
ahead (horizon itself is a feature, `hours_ahead`), and saves
`models/{station}_xgb.json` + `models/{station}_feature_spec.json`. Expands
each station's ~105K rows into ~5M horizon-labeled training rows.

```
python -m src.modeling.train
```

**3. Live prediction** -- pulls the last ~60h of real USGS readings plus
Open-Meteo *forecast* data (not the historical archive -- see notes below),
builds an hourly timeline 1-48h out per station, rolls it into a daily
badge, and writes `data/live/latest_forecast.json`.

```
python -m src.pipeline.predict
```

**4. The Streamlit app** -- reads that JSON file only; never recomputes
anything itself.

```
streamlit run app/app.py
```

## Station notes

Station `02334578` ("Level Creek at Suwanee Dam Road") is a small tributary,
not a Chattahoochee mainstem gauge. The correct station near Medlock Bridge
Rd is **`02335000`** ("Chattahoochee River near Norcross") -- it reports
turbidity and sits at a discharge scale consistent with the tailwater
(roughly 1100-3300 CFS across the 10th-90th percentile band, 2023-2026).
`src/config.py` points at `02335000`; `src/scoring/composite.py`'s Medlock
discharge score is calibrated to this station's real distribution. Buford
Dam's ID, name, and coordinates are correct as originally specified.

## Validated class balance (historical backfill)

| Station | Good | Fair | Poor |
|---|---|---|---|
| Buford Dam | 77.6% | 18.8% | 3.6% |
| Medlock Bridge (02335000) | 83.1% | 15.7% | 1.2% |

The composite scoring thresholds are graduated point scales chosen within
the point ranges the assignment specified (e.g. 0-3 for water temperature),
since exact sub-breakpoints weren't given -- see `src/scoring/composite.py`
for the exact thresholds and rationale.

## Model results

Each station's model was evaluated on a held-out time-based test split
(everything on/after `2025-08-01`), across all horizons 1-48h combined, plus
a spot-check at a few individual horizons:

| Station | Overall accuracy | Macro F1 | 1h / 24h / 48h accuracy |
|---|---|---|---|
| Buford Dam | 0.593 | 0.358 | 0.798 / 0.690 / 0.665 |
| Medlock Bridge | 0.832 | 0.408 | 0.962 / 0.855 / 0.771 |

Medlock's accuracy degrades roughly as expected as the horizon grows.
Buford's does not follow that pattern -- it dips at 12h (0.529, worse than
either 6h or 24h), which doesn't have a clear explanation yet. Buford's top
features are `season` and `dissolved_oxygen_mgl` (discharge-related features
collectively matter but don't dominate outright); Medlock's top feature is
`turbidity_fnu`, with `precipitation`/`cloud_cover` ranking high too --
weather acting as a leading indicator of future river state.

**Horizon-aware vs. fixed-horizon tradeoff**: a simpler model fixed at
exactly 24h ahead (no horizon feature) scored notably higher for Buford
specifically (0.776 vs. 0.690 at the 24h mark) -- training one model across
all 48 horizons dilutes how sharply it can learn any single one, especially
for Buford where hydro-release timing may make different horizons genuinely
different problems. Medlock showed no such penalty (0.858 vs. 0.855,
negligible). The horizon-aware version is what makes the hourly timeline in
the app possible; worth revisiting if per-horizon accuracy at Buford matters
more than the full timeline.

## Design notes

- **Missing water-quality data is neutral, not penalized**: every scoring
  component (`src/scoring/composite.py`) falls back to half its max points
  when the underlying reading is NaN (e.g. Buford's DO/conductance sensors
  didn't come online until 2023-07-26).
- **Time-of-day and stocking weeks are computed in local time**
  (`America/New_York`), not UTC -- USGS/Open-Meteo timestamps come back UTC.
  Every timestamp in this pipeline is stored in UTC by design; use
  `src.features.engineering.to_local()` to view a DataFrame in Eastern time
  (adds a 12-hour `local_time` column) -- display-only, since Eastern time
  repeats an hour at every fall-back DST transition, which can silently
  break time-based rolling windows and forward-fill joins if used as the
  working index.
- **Discharge rate-of-change** uses time-based shifting (exact grid match
  `offset` earlier), not row-count shifting, so gaps in the 15-min USGS grid
  correctly produce NaN deltas instead of silently wrong values.
- `generate_stocking_weeks(year)`'s fall-stocking week is a placeholder
  (week containing Oct 1) pending the real GA DNR stocking calendar.
- **Open-Meteo's "historical" endpoint isn't strictly historical for very
  recent dates.** It's backed by ERA5 reanalysis (~5 days of processing
  latency), so requesting `end` near "today" returns forecast-model values
  for hours that haven't happened yet. Doesn't affect the 2023-2026
  backfill (well outside that window); use
  `fetch_weather_forecast(past_days=...)` for anything near "now."
- **The live prediction anchor row is forward-filled before use**
  (`src/pipeline/predict.py`): the single freshest USGS reading commonly has
  NaN for slower-reporting sensors (water temp, conductance, turbidity, DO
  routinely lag discharge by 15min-1hr in near-real-time telemetry). Unlike
  the rule-based composite score, the ML model has no explicit neutral
  handling for missing values -- it learns some default split from training
  data, which isn't reliably neutral. This matters in practice: an
  unfilled anchor row previously flipped a near-term Medlock forecast from a
  high-confidence "Good" to a high-confidence "Poor" purely from a few NaN
  readings.
- **The daily badge averages predicted probabilities, not labels**, across
  daylight hours (5am-9pm local, a placeholder worth reconsidering) --
  averaging Good/Fair/Poor as raw labels doesn't mean anything numerically;
  averaging each hour's class-probability vector and taking the argmax does.
- **Not yet built**: "key driving factors" plain-language text and a raw
  discharge/temperature chart in the app (both mentioned in the assignment
  as nice-to-haves).

## Deploying (GitHub Actions + Streamlit)

`.github/workflows/daily_forecast.yml` runs `predict.py` daily at 11:00 UTC
(or on demand via the Actions tab) and commits the refreshed
`data/live/latest_forecast.json`. To make it live:

1. `git init`, commit, push to a new GitHub repository
2. Repo Settings -> Actions -> General -> Workflow permissions: select "Read
   and write permissions" (needed for the workflow's `git push` step)
3. Optional, for a public always-current page: deploy on Streamlit Community
   Cloud pointed at this repo's `app/app.py` -- it reads
   `data/live/latest_forecast.json` straight from the repo, so each commit
   the workflow makes is what a deployed app shows next time it's viewed.
