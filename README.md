# Air Quality Prediction — Serbia

Spatio-temporal forecasting of PM2.5 / PM10 levels and AQI category across 28 Serbian cities, using 10 years of hourly air quality data. Built for the Computational Intelligence course.

## Overview

The project predicts PM2.5 concentration 24 hours ahead using four regression approaches — **XGBoost**, **LightGBM**, **KNN**, and **LSTM** — trained and evaluated on the same chronological data split for a fair comparison. Predicted PM2.5 values are mapped to an AQI danger category following SEPA methodology.

**Data:** ~87,600 hourly records per city (10 years), sourced from the [Open-Meteo Historical Air Quality API](https://open-meteo.com/en/docs/air-quality-api).

## Project structure

```
air-quality-prediction/
├── data/
│   ├── raw/               # raw per-city CSV files (not tracked in git)
│   └── processed/         # cleaned, feature-engineered data
├── src/
│   ├── data_prep.py       # cleaning, interpolation, lag & temporal features
│   ├── split.py            # chronological time-series split (70/20/10)
│   ├── evaluate.py         # RMSE / MAE / R² evaluation
│   ├── aqi_utils.py        # PM2.5 → AQI category mapping (SEPA)
│   └── models/
│       ├── xgboost_model.py
│       ├── lightgbm_model.py
│       ├── knn_model.py
│       └── lstm_model.py
├── notebooks/               # exploratory analysis & visualization
├── results/                  # saved models, plots, metrics
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place raw per-city CSV files in `data/raw/` (not included in this repository due to size).

## Usage

```python
from src.data_prep import prepare_city_data, get_feature_columns
from src.split import time_series_split, get_X_y
from src.evaluate import evaluate_predictions

df = prepare_city_data("data/raw/beograd.csv", city_name="Beograd")
train_df, val_df, test_df = time_series_split(df)

feature_cols = get_feature_columns(df)
X_train, y_train = get_X_y(train_df, feature_cols)
X_test, y_test = get_X_y(test_df, feature_cols)

# ... train a model ...

result = evaluate_predictions(y_test, y_pred, model_name="XGBoost")
```

## Methodology

- **Preprocessing:** time-based interpolation, MinMax scaling, lag features (1–48h) for PM2.5 and PM10, temporal features (hour, month, day of week)
- **Split:** chronological, no shuffling — train (2016–2023), validation (2023–2025), test (2025–2026)
- **Evaluation:** RMSE, MAE, R² on the same held-out test set across all models

## Authors

Teodora Aleksić & Lenka Nikolić
