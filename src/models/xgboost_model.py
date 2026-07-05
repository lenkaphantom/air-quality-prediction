import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.data_prep import prepare_city_data, POLLUTANT_COLS
from src.split import time_series_split, get_X_y
from src.evaluate import evaluate_predictions

NEAR_LAGS = [1, 2, 3, 6, 12]
FAR_LAGS  = [22, 23, 24, 25, 46, 47, 48]
ROLLING_WINDOWS_TO_KEEP = [12, 24]

DEFAULT_N_ESTIMATORS  = 500
DEFAULT_MAX_DEPTH     = 6
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_SUBSAMPLE     = 0.8


def get_xgb_feature_columns() -> list[str]:
    cols = []

    cols += POLLUTANT_COLS

    cols += ["hour", "month", "day_of_week", "heating_season"]

    for col in ["pm2_5", "pm10"]:
        for lag in NEAR_LAGS + FAR_LAGS:
            cols.append(f"{col}_lag_{lag}h")

    for window in ROLLING_WINDOWS_TO_KEEP:
        cols.append(f"pm2_5_rolling_mean_{window}h")
        cols.append(f"pm2_5_rolling_std_{window}h")
        cols.append(f"pm2_5_rolling_max_{window}h")

    return cols


def find_best_xgb_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "n_estimators":  [300, 500],
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best = grid_search.best_params_
    print(f"Najbolji parametri: {best}")
    print(f"CV RMSE: {-grid_search.best_score_:.4f}")

    return best


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> XGBRegressor:
    if params is None:
        params = {
            "n_estimators":  DEFAULT_N_ESTIMATORS,
            "max_depth":     DEFAULT_MAX_DEPTH,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "subsample":     DEFAULT_SUBSAMPLE,
        }

    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
        eval_metric="rmse",
        **params,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f"XGBoost: trening zavrsen na iteraciji {best_iter}")

    return model


def predict_xgb(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def run_xgb_pipeline(
    csv_path: str | Path,
    city_name: str,
    params: dict | None = None,
) -> dict:
    df = prepare_city_data(csv_path, city_name=city_name)
    train_df, val_df, test_df = time_series_split(df)

    feature_cols = get_xgb_feature_columns()
    X_train, y_train = get_X_y(train_df, feature_cols)
    X_val,   y_val   = get_X_y(val_df,   feature_cols)
    X_test,  y_test  = get_X_y(test_df,  feature_cols)

    model = train_xgb(X_train, y_train, X_val, y_val, params=params)

    y_pred = predict_xgb(model, X_test)
    y_pred = np.maximum(y_pred, 0.0)

    result = evaluate_predictions(y_test, y_pred, model_name=f"XGBoost ({city_name})")

    return {
        "result":  result,
        "model":   model,
        "y_test":  y_test,
        "y_pred":  y_pred,
    }


def save_xgb_model(model: XGBRegressor, city_name: str, output_dir: str | Path = "results"):
    output_dir = Path(output_dir) / "xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"xgboost_model_{city_name}.pkl")
    print(f"Sačuvano: {output_dir / f'xgboost_model_{city_name}.pkl'}")


def load_xgb_model(city_name: str, results_dir: str | Path = "results") -> XGBRegressor:
    return joblib.load(Path(results_dir) / "xgboost" / f"xgboost_model_{city_name}.pkl")


if __name__ == "__main__":
    output = run_xgb_pipeline("data/raw/beograd.csv", city_name="Beograd")
    print(output["result"])
    save_xgb_model(output["model"], "Beograd")
