# EV Range (Distance-to-Empty) Predictor

> Work in progress: Phase 5 (deployment, up to but not including the Android app) done; draft findings in
> [FINDINGS.md](FINDINGS.md); app handoff spec in [android/README.md](android/README.md).

Predicts an electric vehicle's **energy consumption rate over the next stretch of road**
from real driving telemetry, derives **distance-to-empty (DTE)** from it, and compares
running the model **on-device (TFLite on Android)** with running it **on a server (FastAPI)**.

## Data

**Battery and Heating Data in Real Driving Cycles** (IEEE DataPort, open access):
https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles

70 real BMW i3 trips (series A: 32 summer trips, series B: 38 winter trips), sampled at 10 Hz.
The data is not committed to this repo. To obtain it:

1. Sign in to IEEE DataPort (a free account works) and download the dataset.
2. Unzip it so that the trip CSVs sit directly in `data/raw/bmw_i3/`
   (`TripA01.csv` … `TripB38.csv`, plus `Overview.xlsx`).

> **Scope note:** these are four-wheeler EVs, while the author's work domain is
> connected two-wheelers. What transfers and what doesn't will be discussed in the final write-up.

## Setup

Requires Python 3.11 (developed on 3.11.9, macOS arm64).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt     # or requirements-lock.txt for the exact full set
brew install libomp                           # macOS only: LightGBM's OpenMP runtime
.venv/bin/python -m src.check_env             # prints library versions
.venv/bin/python -m pytest                    # runs the tests
```

## Reproduce

```bash
.venv/bin/python -m src.data.build_dataset    # raw CSVs -> data/processed/trips_clean.parquet (+ cleaning_report.csv)
.venv/bin/python -m src.data.data_card        # -> data/DATA_CARD.md (all numbers computed)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # -> figures/eda/*.png
.venv/bin/python -m src.data.splits           # -> experiments/configs/splits.csv (seeded)
.venv/bin/python -m src.models.run_baselines  # baseline CV results, appended to experiments/log.csv
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_baselines.ipynb
.venv/bin/python -m src.models.run_cv         # Phase 3 tuning on CV folds (~10 min) -> experiments/phase3_*.csv
.venv/bin/python -m src.models.run_test       # one-time test evaluation of the selected models
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_models.ipynb
.venv/bin/python -m src.models.run_oracle     # Phase 4 "knowing the road" upper-bound experiment
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/04_error_analysis.ipynb
.venv/bin/python -m src.export_tflite         # Phase 5: TFLite exports, verified on the test set
.venv/bin/python -m src.benchmark             # -> experiments/phase5_benchmark.csv
.venv/bin/python -m src.android_handoff       # -> models/handoff/ (models, replay trip, golden vectors)
.venv/bin/uvicorn server.app:app --port 8000  # inference API; docs at http://127.0.0.1:8000/docs
```

## Problem formulation (Phase 2)

- **Target:** consumption rate over the next **N = 5 km**, in %SoC/km, predicted every 100 m after the first 1 km of a trip.
  Distance-to-empty is derived as `DTE = (SoC − 8 %) / rate` (8 % = the BMS's lower SoC limit).
  No trip reaches empty, so "actual DTE" means *range if the next 5 km's consumption continued*.
- **Split:** by **day** (same-day trips share weather and sometimes run back-to-back), stratified by **season**.
  7 days held out as the test set; the other days form 5 cross-validation folds used for all tuning.

See [data/DATA_CARD.md](data/DATA_CARD.md) for every column, the cleaning steps and the sanity checks.

## Repo layout

```
data/          raw/ and processed/ (gitignored)
notebooks/     01_eda, 02_baselines, 03_models, 04_error_analysis
src/           data pipeline, features, models, evaluation, TFLite export
server/        FastAPI inference API
android/       Android app (Kotlin + Compose + TFLite)
experiments/   log.csv (one row per run) and configs
tests/         unit tests for the data pipeline and metrics
```
