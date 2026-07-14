# A Machine Learning Pipeline for Trout Fishing Condition Forecasting on the Chattahoochee River Tailwater

### Design, Implementation, and Evaluation of an End-to-End Hydrological Forecasting and Decision-Support System

Adam Chabaane · INFD 651 · Mercer University

---

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Data Sources](#2-data-sources)
- [3. Feature Engineering](#3-feature-engineering)
- [4. Composite Scoring Methodology](#4-composite-scoring-methodology)
- [5. Predictive Modeling](#5-predictive-modeling)
- [6. Live Prediction Pipeline](#6-live-prediction-pipeline)
- [7. Application Layer](#7-application-layer)
- [8. Limitations and Future Work](#8-limitations-and-future-work)
- [9. Conclusion](#9-conclusion)

---

## Abstract

This paper documents the design and implementation of an end-to-end machine learning system that forecasts trout fishing conditions on the Chattahoochee River tailwater between Buford Dam and Morgan Falls Dam. The system ingests real-time hydrological sensor data from two USGS gauging stations and weather data from the Open-Meteo API, engineers a set of physically motivated features, constructs a rule-based composite condition label, and trains a horizon-aware gradient-boosted tree classifier capable of forecasting conditions between one and forty-eight hours ahead. The trained models drive a live prediction pipeline and a public-facing Streamlit dashboard, refreshed daily via a scheduled continuous-integration workflow. Beyond describing the final architecture, this paper documents the verification and debugging process that shaped it, including a station-identity correction discovered by comparing pipeline output against a live data source, a label-leakage failure mode uncovered when an initial model achieved 99.9% accuracy, and a data-serialization defect that caused a charting library to silently drop valid input. These findings are reported alongside the final results because they materially shaped the system's design and are broadly relevant to similar forecasting projects.

---

## 1. Introduction

### 1.1 Motivation and Problem Statement

Recreational trout anglers fishing the Chattahoochee River tailwater face conditions that vary significantly on sub-daily timescales. Buford Dam operates as a peaking hydroelectric facility, meaning discharge can change by thousands of cubic feet per second within a single hour as generation schedules turn on and off. Water temperature, dissolved oxygen, and turbidity all vary with discharge, season, and recent weather. An angler deciding whether — and when — to fish this stretch of river currently has to manually reason about several independent data sources (USGS gauge readings, weather forecasts, and informal knowledge of dam operations) to estimate whether conditions will be favorable. This project automates that reasoning process.

### 1.2 Scope: Fishing Conditions, Not Bite Quality

A foundational design decision, made explicit early in the project and enforced throughout the codebase, is that the system predicts river *conditions* — whether the water is safe and physiologically comfortable for trout and safe to wade in — and not whether fish are likely to bite. No catch-rate, creel-survey, or angler-report data exists anywhere in the pipeline, so bite propensity is not a claim this system can honestly make. This distinction matters for two reasons. First, it constrains which variables belong in the composite label used to train the model: variables that plausibly affect fish behavior without affecting water safety (for example, cloud cover, which anecdotally affects feeding activity through light penetration) are retained as model features but deliberately excluded from the composite label itself. Second, it constrains the language used in the user-facing application: the dashboard explicitly states that a "Good" rating means good conditions to fish in, not a guaranteed catch.

### 1.3 System Overview

The system consists of five stages, implemented as a modular Python package: (1) data ingestion from USGS and Open-Meteo, (2) feature engineering on the joined hydrological and meteorological time series, (3) construction of a rule-based composite condition label used as the training target, (4) training of a per-station gradient-boosted tree classifier, and (5) a live prediction pipeline that feeds a Streamlit dashboard, refreshed daily by a scheduled GitHub Actions workflow. Each stage is described in turn below.

---

## 2. Data Sources

### 2.1 USGS Hydrological Monitoring Stations

Two USGS stations on the Chattahoochee tailwater provide the hydrological input data, pulled at 15-minute resolution via the `dataretrieval` Python package's `waterdata.get_continuous` interface.

| Station | USGS ID | Latitude | Longitude | Parameters |
|---|---|---|---|---|
| Buford Dam | 02334430 | 34.156667 | -84.078417 | Discharge, water temp, DO, conductance |
| Medlock Bridge | 02335000 | 33.997222 | -84.201944 | Discharge, water temp, turbidity, conductance |

### 2.2 Open-Meteo Weather Data

Weather data is drawn from Open-Meteo, a free, keyless weather API, using two distinct endpoints for two distinct purposes. The archive endpoint, backed by ERA5 reanalysis, supplies historical hourly air temperature, surface pressure, precipitation, and cloud cover for model training. The forecast endpoint supplies the same variables for live prediction, including a configurable trailing window of recent-past data (needed to compute rolling features at prediction time) and a multi-day forward forecast. Hourly weather values are forward-filled onto the finer 15-minute USGS grid rather than interpolated, since precipitation and cloud cover are not smooth continuous signals.

A data-quality finding is documented here because it materially affects how the live pipeline is implemented: the archive endpoint is not strictly historical for dates within approximately five days of the present, since ERA5 reanalysis has that much processing latency. Requesting recent dates from the archive endpoint was empirically confirmed to silently return forecast-model output for hours that had not yet occurred. The live prediction pipeline therefore always uses the forecast endpoint, never the archive endpoint, regardless of how recent the requested window is.

### 2.3 Station Identity Verification

The initial project specification identified the downstream station as "Medlock Bridge," station ID 02334578, at approximately 34.0234, -84.2402. Verifying this against USGS site metadata revealed a discrepancy: station 02334578 is actually named "Level Creek at Suwanee Dam Road," a small tributary creek, not a Chattahoochee mainstem gauge. Its discharge magnitude (median approximately 5 cubic feet per second) happened to closely match the specification's stated statistics for the downstream station, which was initially treated as sufficient confirmation that the station ID was correct despite the name and coordinate mismatch. This was a methodological error: a coincidental statistical resemblance is not equivalent to a confirmed identity match.

The error was caught by directly comparing live pipeline output against the USGS current-conditions website rather than relying solely on metadata lookups. This comparison identified station 02335000, "Chattahoochee River near Norcross," as the correct downstream station: it reports turbidity (required for the downstream scoring rubric), and its discharge scale (approximately 1,100 to 3,300 cubic feet per second across the 10th-90th percentile band, 2023-2026) is consistent with a tailwater gauge rather than a tributary creek. All downstream configuration, including the discharge-scoring thresholds described in Section 4.1, was subsequently rebuilt around this corrected station's real distribution.

| | Specification | USGS Metadata (Verified) |
|---|---|---|
| Station ID | 02334578 | 02335000 |
| Name | "Medlock Bridge" | Chattahoochee River near Norcross |
| Character | Small tributary creek | Chattahoochee mainstem tailwater |
| Median discharge | ~3.7 cfs (creek-scale) | ~1,720 cfs (river-scale) |

*For clarity, this document and the codebase retain the label "Medlock Bridge" as the internal identifier for this station, since that label is used consistently throughout the project's configuration, scoring, and application code — only the underlying USGS station ID and coordinates were corrected.*

---

## 3. Feature Engineering

Five categories of derived features are computed on the pivoted, weather-joined time series for each station.

### 3.1 Discharge Dynamics

One-hour and three-hour discharge rate-of-change features are computed using time-based shifting rather than row-count shifting, so that gaps in the 15-minute USGS grid produce an explicit missing value rather than a silently incorrect delta computed across an irregular time interval. A binary stability flag records whether the one-hour discharge change was below 200 cubic feet per second in magnitude. This feature category exists because Buford Dam is a peaking hydroelectric facility: the rate at which discharge is changing is often more informationally relevant than the absolute discharge level.

### 3.2 Rolling Water Temperature

Three-, twelve-, and twenty-four-hour rolling means of water temperature are computed to capture trend rather than a single potentially noisy instantaneous reading.

### 3.3 Season and Time-of-Day

A meteorological season flag (Winter, Spring, Summer, Fall) is derived from the calendar month. A time-of-day flag (Night, Morning, Afternoon, Evening) is derived from the local hour. Both USGS and Open-Meteo timestamps are stored in UTC throughout the pipeline; the time-of-day computation explicitly converts to America/New_York local time before bucketing, since "which part of the day it is" is inherently a local-time question. All other computation, including the rolling-window and forward-fill logic described elsewhere in this paper, deliberately remains in UTC: Eastern time repeats an hour at every autumn daylight-saving transition, which can silently corrupt time-based rolling and join operations if used as the underlying working index.

### 3.4 Stocking Recency

The Georgia Department of Natural Resources stocks this tailwater on a weekly schedule from April through July, with two additional stockings in the two weeks preceding Labor Day and one further stocking in the fall. Because the exact day of week is not publicly documented, a binary `stocked_week_flag` is computed at week-level granularity: a Monday-anchored calendar week is flagged if it falls within the known April-July weekly range, is one of the two weeks immediately before Labor Day, or is the week containing October 1 (an explicit placeholder for the undocumented fall stocking, pending the actual GA DNR calendar).

---

## 4. Composite Scoring Methodology

The training label — a Good, Fair, or Poor condition rating — is constructed as a deterministic, rule-based composite score rather than collected from any external ground truth, since no such ground truth (e.g., angler-reported outcomes) exists. Scoring is station-specific, since the two stations have different instrumentation and hydrological character.

### 4.1 Station-Specific Rubrics

**Buford Dam scoring rubric (maximum 9 points)**

| Factor | Points | Best-case criterion |
|---|---|---|
| Water temperature | 0-3 | 10-18°C |
| Discharge | 0-3 | 600-1800 cfs (wadeable) |
| Discharge stability | 0-1 | <200 cfs/hr change |
| Dissolved oxygen | 0-2 | ≥ 8 mg/L |

**Medlock Bridge (station 02335000) scoring rubric (maximum 7 points)**

| Factor | Points | Best-case criterion |
|---|---|---|
| Water temperature | 0-3 | 10-18°C |
| Discharge | 0-2 | 800-2000 cfs (wadeable) |
| Turbidity | 0-2 | ≤ 5 NTU |

*Discharge thresholds are calibrated to this station's actual 2023-2026 daily-discharge distribution (10th/25th/50th/75th/90th percentile approximately 1,100/1,300/1,720/2,440/3,280 cfs).*

### 4.2 Handling Missing Data

Sensor gaps are common and expected — for example, Buford's dissolved-oxygen and conductance sensors did not report data until July 2023. Rather than penalizing a missing reading or dropping the affected row, every scoring component returns a neutral score equal to exactly half of that component's maximum points when its underlying reading is NaN. This rule is applied uniformly across every component of both stations' rubrics.

### 4.3 Classification Thresholds

| Condition | Buford Dam (of 9) | Medlock Bridge (of 7) |
|---|---|---|
| Good | 7-9 | 5-7 |
| Fair | 4-6 | 3-4 |
| Poor | 0-3 | 0-2 |

### 4.4 Validated Class Balance

The composite scoring rules were applied across the full 2023-01-01 through 2026-01-01 historical backfill to validate the resulting label distribution.

| Station | Good | Fair | Poor |
|---|---|---|---|
| Buford Dam | 77.6% | 18.8% | 3.6% |
| Medlock Bridge (02335000) | 83.1% | 15.7% | 1.2% |

*Both stations show the expected Good-majority skew for a well-regulated tailwater fishery. No independent, pre-existing exploratory data analysis exists for the corrected Medlock Bridge station to validate this distribution against, since the specification's original class-balance expectations were computed against the incorrect tributary station discussed in Section 2.3.*

---

## 5. Predictive Modeling

### 5.1 Initial Approach and the Label Leakage Problem

An initial XGBoost classifier was trained per station using the raw sensor readings (water temperature, discharge, dissolved oxygen or turbidity, discharge stability) as features, having excluded only the intermediate scoring columns (`score_water_temp`, `score_discharge`, `composite_score`, and so on) that make the composite label's arithmetic explicit. This model achieved **99.9% accuracy** on a held-out time-based test split — a result immediately recognized as a red flag rather than a success.

The underlying problem is that the composite label is a deterministic function of exactly those raw readings. Excluding the intermediate score columns removed the arithmetic's visible output but not its inputs: a sufficiently expressive tree model can trivially reconstruct a deterministic rule (e.g., "temperature between 10 and 18, discharge between 600 and 1800, therefore Good") from the same raw inputs the rule consumes. The model was not learning any real-world predictive relationship; it was recovering the project's own scoring formula. This is not useful for the system's actual goal, which is to forecast conditions that have not yet occurred, using only information available before they occur.

### 5.2 Horizon-Aware Reformulation

The fix was not to remove additional feature columns — the raw sensor readings are legitimate and necessary predictors — but to change *which row's label* is paired with each row's features during training. Rather than pairing a timestamp's features with that same timestamp's label (a pairing that is circular by construction), each row's features are paired with the label from a specified number of hours later, and that lead time itself, `hours_ahead`, is included as a model feature ranging from 1 to 48. At prediction time, a single current-conditions row is queried once per hour with `hours_ahead = 1, 2, ..., 48` to construct a full 48-hour hourly forecast timeline from one model, rather than requiring 48 separately trained models.

Constructing this horizon-expanded training set multiplies the original approximately 105,000 rows per station by 48 (one copy per horizon, with rows near the end of the historical record dropped where no future label exists to pair with), yielding approximately 5 million training rows per station.

### 5.3 Training Configuration

Each station's classifier is an XGBoost multi-class model (objective `multi:softmax`, three classes) with 300 estimators, maximum depth 6, and a learning rate of 0.1. Class imbalance (see Section 4.4) is addressed via balanced sample weighting (`sklearn.utils.compute_sample_weight`). The evaluation split is time-based rather than random — all data from August 1, 2025 onward is held out for testing — since this is autocorrelated time-series data, and a random split would leak information between temporally adjacent training and test rows and overstate performance. Categorical features (season, time-of-day) are handled natively by XGBoost rather than one-hot encoded.

*A practical implementation finding is reported here for completeness: training initially took approximately 35 minutes per run on a 12-core machine because the XGBoost classifier's `n_jobs` parameter was left at its default rather than explicitly set to -1, resulting in near single-threaded execution. Setting `n_jobs=-1` explicitly reduced this to approximately 2.5 minutes per station.*

### 5.4 Results

| Station | Overall Acc. | Macro F1 | 1h | 6h | 12h | 24h | 48h |
|---|---|---|---|---|---|---|---|
| Buford Dam | 0.593 | 0.358 | 0.798 | 0.684 | 0.529 | 0.690 | 0.665 |
| Medlock Bridge | 0.832 | 0.408 | 0.962 | 0.864 | 0.850 | 0.855 | 0.771 |

Medlock Bridge's per-horizon accuracy degrades approximately monotonically as the forecast horizon lengthens, as would be expected of a well-behaved forecasting model. Buford Dam's does not follow this pattern: accuracy dips at the 12-hour horizon (0.529) to a value lower than either the 6-hour (0.684) or 24-hour (0.690) horizon. This anomaly does not have a confirmed explanation and is reported as an open question rather than resolved; a plausible but unverified hypothesis is that Buford's hydroelectric release schedule introduces operational patterns tied to specific times of day that interact unevenly with a 12-hour-ahead query, since a 12-hour offset from an arbitrary anchor time systematically lands at a different local time of day than a 24-hour offset from the same anchor.

### 5.5 Feature Importance Analysis

For Buford Dam, the two highest-importance features are season and dissolved oxygen; discharge-related features (`discharge_cfs`, `discharge_delta_3hr_cfs`, `discharge_stable_1hr`) are collectively substantial but do not individually dominate. For Medlock Bridge, turbidity is the single highest-importance feature, followed closely by precipitation and cloud cover — weather variables ranking above the discharge reading itself. This is a physically sensible result for a genuine forecasting task: precipitation today is a leading indicator of a turbidity or discharge change tomorrow, so a model forecasting hours ahead should be expected to lean on current weather as an early-warning signal for future river state, rather than relying solely on the river's current state.

### 5.6 Horizon-Aware versus Fixed-Horizon Tradeoff

A simpler model fixed at exactly 24 hours ahead (trained without the `hours_ahead` feature or the horizon-expanded dataset) was evaluated for comparison and found to score notably higher for Buford Dam specifically at the 24-hour mark (accuracy 0.776 versus 0.690 for the horizon-aware model). Medlock Bridge showed no comparable penalty (0.858 versus 0.855, a negligible difference). This indicates that training a single model across all 48 horizons simultaneously dilutes how sharply that model can learn any one horizon, and that this cost is asymmetric between the two stations — plausibly because Buford's hydroelectric operational patterns make different horizons genuinely different learning problems, a distinction Medlock Bridge does not share. The horizon-aware model was retained in the deployed system because it is the only architecture that supports the application's full 48-hour timeline rather than a single fixed-horizon point estimate; this remains a documented tradeoff rather than a resolved question.

---

## 6. Live Prediction Pipeline

### 6.1 Architecture

The live prediction pipeline pulls approximately 60 hours of recent USGS readings and Open-Meteo forecast data (both historical actuals for the trailing window and forward forecast), applies the identical feature-engineering steps used during training, and selects the most recent row as the current-conditions anchor. The saved model and its accompanying feature specification (the exact feature-column list and categorical encodings fixed at training time) are loaded, and the anchor row is queried once per hour for 48 hours to construct the hourly forecast timeline described in Section 5.2. This timeline is further condensed into a single daily summary rating by averaging each hour's predicted class-probability vector across a fixed daylight window (5:00 a.m. to 9:00 p.m. local time) and classifying the resulting average — a method chosen over majority-voting the discrete hourly labels because it is not distorted by a single anomalous hour, and over naively averaging the labels' numeric encodings because it operates on the underlying probability distribution the model actually produces rather than an arbitrary post-hoc encoding of its output.

### 6.2 The Anchor-Row Missing-Data Issue

During verification, the current-conditions anchor row was found to routinely contain missing values for water temperature, conductance, turbidity, and dissolved oxygen, since these slower-reporting sensors commonly lag discharge telemetry by fifteen minutes to one hour in near-real-time USGS reporting — meaning the single freshest available row is precisely where these gaps are most likely to occur. This matters because the rule-based composite score's neutral missing-data handling (Section 4.2) is a property of that scoring code specifically; it does not extend to the trained model, which has no explicit neutral-value convention and instead learns some default split direction for missing values during training. This was confirmed to produce materially incorrect output in practice: prior to forward-filling the anchor row (carrying the last known valid reading forward to fill any gap at the current timestamp), a near-term Medlock Bridge forecast changed from **98.5% confidence in a Good rating to 73% confidence in a Poor rating** purely as an artifact of a few missing readings in the input row, with no underlying change in actual river conditions.

---

## 7. Application Layer

### 7.1 Dashboard Design

The user-facing application is a Streamlit dashboard that reads a single JSON artifact produced by the live prediction pipeline and performs no computation of its own, per the project's design principle that the application layer should only ever display precomputed results. For each station, the dashboard displays: a daily condition badge; a "right now" panel of the latest known sensor readings (water temperature, discharge, cloud cover, precipitation, conductance, and the station-specific dissolved-oxygen or turbidity reading); a 48-hour color-coded hourly timeline; and reference links to the station's live USGS monitoring page and, for Buford Dam, the U.S. Army Corps of Engineers hydropower generation schedule. A discharge-specific warning label appears when the current reading exceeds that station's wadeable-range upper bound (1,800 cfs at Buford Dam, 2,000 cfs at Medlock Bridge), reusing the identical thresholds already defined for composite scoring rather than duplicating the figures.

The 48-hour timeline was initially implemented using the Altair/Vega-Lite charting library, but this implementation encountered a reproducible defect: the browser console reported an "Infinite extent" error for the chart's temporal axis field, and the chart rendered as a single uniform color block rather than the expected color-coded hourly bands, despite the underlying data being confirmed correct at the Python level. The root cause was not conclusively isolated — candidate explanations investigated included a nested-dictionary column corrupting Streamlit's Arrow-based chart serialization and a timezone-aware datetime dtype incompatibility, and both were addressed without resolving the underlying symptom. The timeline was ultimately reimplemented as a row of plain HTML and CSS colored elements, which sidesteps the charting library's data-serialization path entirely and was verified, by direct inspection of the rendered page's DOM, to display the correct per-hour colors matching the underlying forecast data.

### 7.2 Daily Aggregation Methodology

See Section 6.1 for the probability-averaging method used to condense the 48-hour timeline into the dashboard's single daily badge.

### 7.3 Continuous Deployment and Automation

A GitHub Actions workflow, scheduled via cron at 11:00 UTC daily (approximately 6:00-7:00 a.m. Eastern time depending on daylight-saving status) and additionally triggerable on-demand, checks out the repository, installs dependencies, runs the live prediction pipeline, and commits the refreshed forecast artifact back to the repository. The deployed Streamlit Community Cloud application automatically redeploys when it detects a new commit on the tracked branch, so the public dashboard reflects each day's automated refresh without further manual action. Two implementation errors were identified and corrected during initial deployment: the workflow's file paths initially assumed the project lived in a `trout-forecast/` subdirectory within the repository, when in fact the repository root is the project root; and a repository-level GitHub setting restricting workflow permissions to read-only initially prevented the workflow's commit-and-push step from succeeding, requiring an explicit grant of read-and-write workflow permissions.

---

## 8. Limitations and Future Work

- The system predicts water conditions, not fish behavior or catch likelihood, as discussed in Section 1.2. Extending it to a genuine bite-quality prediction would require angler-reported catch or creel-survey data that does not currently exist for this fishery.
- The Buford Dam model's anomalous accuracy dip at the 12-hour horizon (Section 5.4) is unresolved and warrants targeted investigation, potentially by examining whether hydroelectric release scheduling introduces a systematic time-of-day interaction at that specific offset.
- Live forecasts beyond the immediate future are generated using the current weather snapshot rather than the actual forecasted weather trajectory for each future hour, since the training data's weather features are drawn from actuals rather than a historical archive of point-in-time forecasts. This is a reasonable simplification for a first system but represents a train/serve mismatch relative to how the live pipeline could ideally use the forecast data it already retrieves.
- The fall stocking week used in the stocking-recency feature (Section 3.4) is an explicit placeholder pending the real Georgia Department of Natural Resources stocking calendar.
- The daily aggregation window (5:00 a.m. to 9:00 p.m. local time, Section 6.1) is a design choice rather than a value derived from data and may warrant reconsideration.
- The dashboard's "key driving factors" plain-language explanation and a supplementary raw discharge/temperature chart, both suggested in the original project brief, remain unimplemented as deliberately deferred future work.

---

## 9. Conclusion

This project delivers a complete, deployed forecasting system spanning data ingestion, feature engineering, rule-based label construction, horizon-aware machine learning, live inference, and a continuously updated public dashboard. Beyond the system itself, this paper has documented several methodological findings that generalize beyond this particular application: that a coincidental statistical resemblance is not sufficient confirmation of a data source's identity; that a suspiciously high accuracy score is reason for scrutiny rather than celebration, since it frequently indicates that a model is recovering a known deterministic rule rather than learning a genuine predictive relationship; that a machine learning model does not automatically inherit a hand-engineered rule system's careful handling of missing data; and that a data visualization library failing silently can produce a plausible-looking but incorrect result that requires direct verification, not just the absence of a visible error, to catch. The resulting system, and the process that produced it, are documented here in full.
