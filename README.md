# Air Quality Prediction Serbia

Spatio-temporal forecasting of PM2.5 / PM10 levels and AQI category across 28 Serbian cities, using 10 years of hourly air quality data. Built for the Computational Intelligence course.

## Overview

The project predicts PM2.5 concentration 24 hours ahead using four regression approaches — **XGBoost**, **LightGBM**, **KNN**, and **LSTM** — trained and evaluated on the same chronological data split for a fair comparison. Predicted PM2.5 values are mapped to an AQI danger category following SEPA methodology.

**Data:** ~87,600 hourly records per city (10 years), sourced from the [Open-Meteo Historical Air Quality API](https://open-meteo.com/en/docs/air-quality-api). Models are trained on 8 key cities (Beograd, Novi Sad, Niš, Bor, Valjevo, Kostolac, Smederevo, Kopaonik) as a representative subset; full training across all 28 cities is supported but not required.

## Project structure

```
air-quality-prediction/
├── main.py                  # CLI entry point: training + next-day prediction
├── data/
│   ├── raw/                 # raw per-city CSV files (not tracked in git)
│   └── processed/           # cleaned, feature-engineered data (optional staging)
├── src/
│   ├── data_prep.py         # cleaning, interpolation, lag & temporal features
│   ├── split.py              # chronological time-series split (70/20/10)
│   ├── evaluate.py           # RMSE / MAE / R² evaluation
│   ├── aqi_utils.py          # PM2.5 → AQI category mapping (SEPA)
│   ├── visualize.py          # all plotting functions used by the notebooks
│   ├── train_models.py       # training orchestrator (KEY_CITIES, ALL_CITIES, train_city)
│   ├── predict_next_day.py   # hourly next-day forecast, single city or batch
│   ├── check_importance.py   # feature importance diagnostics
│   ├── check_overfit.py      # train/val/test generalization check
│   └── models/
│       ├── knn_model.py
│       ├── lstm_model.py
│       ├── xgboost_model.py
│       └── lightgbm_model.py
├── notebooks/
│   ├── knn_lstm_analysis.ipynb          # KNN & LSTM
│   ├── xgboost_lightgbm_analysis.ipynb  # XGBoost & LightGBM
│   └── final_comparison.ipynb           # all 4 models, extreme-peak analysis
├── results/in
│   ├── knn/ lstm/ xgboost/ lightgbm/    # saved models + scalers per city
│   ├── diagnostics/                     # saved plots
│   ├── forecast_[city]_[date].csv       # next-day predictions
│   └── summary_all_cities.csv           # training summary across cities
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

```bash
# Train + predict for one city (default: Beograd)
python main.py --city Beograd

# Train + predict for the 8 key cities
python main.py --key

# Train + predict for all 28 cities
python main.py --all

# Fast mode — skip GridSearchCV, use pre-optimized hyperparameters
python main.py --fast

# Force retraining even if saved models already exist
python main.py --force

# Skip training, only run next-day prediction (requires existing models)
python main.py --no-train
```

Programmatic usage:

```python
from src.data_prep import prepare_city_data, get_feature_columns
from src.split import time_series_split, get_X_y
from src.evaluate import evaluate_predictions

df = prepare_city_data("data/raw/Beograd.csv", city_name="Beograd")
train_df, val_df, test_df = time_series_split(df)

feature_cols = get_feature_columns(df)
X_train, y_train = get_X_y(train_df, feature_cols)
X_test, y_test = get_X_y(test_df, feature_cols)

# ... train a model ...

result = evaluate_predictions(y_test, y_pred, model_name="XGBoost")
```

## Methodology

- **Preprocessing:** time-based interpolation, MinMax scaling, lag features (near: 1–12h, far: 22–25h and 46–48h for the daily cycle), rolling statistics (12h/24h mean/std/max), and a heating-season flag
- **Split:** chronological, no shuffling — train (2016–2023), validation (2023–2025), test (2025–2026)
- **Hyperparameter search:** GridSearchCV with `TimeSeriesSplit` for all four models
- **Evaluation:** RMSE, MAE, R² on the same held-out test set across all models, plus a dedicated extreme-peak analysis (metrics computed only on hours above the SEPA "Veoma zagađen" threshold)

## Authors

Teodora Aleksić & Lenka Nikolić