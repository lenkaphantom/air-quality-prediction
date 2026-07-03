import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib

from src.data_prep import prepare_city_data, POLLUTANT_COLS
from src.split import time_series_split, get_X_y
from src.evaluate import evaluate_predictions

NEAR_LAGS = [1, 2, 3, 6, 12]
FAR_LAGS = [22, 23, 24, 25, 46, 47, 48]
ROLLING_WINDOWS_TO_KEEP = [12, 24]

# Optimalni hiperparametri pronadjeni GridSearchCV-om na Beogradu.
# Koriste se za brzi rezim (--fast) kada se treniraju svi gradovi.
DEFAULT_K = 150
DEFAULT_WEIGHTS = "distance"


def get_knn_feature_columns() -> list[str]:
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


def find_best_k(X_train: pd.DataFrame, y_train: pd.Series, k_values=None) -> tuple[int, str]:
    if k_values is None:
        k_values = [10, 20, 30, 40, 50, 75, 100, 150]

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_neighbors": k_values,
        "weights": ["uniform", "distance"],
    }

    grid_search = GridSearchCV(
        estimator=KNeighborsRegressor(n_jobs=-1),
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_k = grid_search.best_params_["n_neighbors"]
    best_weights = grid_search.best_params_["weights"]
    print(f"Najbolji K: {best_k}, weights: {best_weights}")
    print(f"CV RMSE: {-grid_search.best_score_:.4f}")

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df["rmse"] = -results_df["mean_test_score"]
    top5 = results_df.sort_values("rmse")[["param_n_neighbors", "param_weights", "rmse"]].head(5)
    print("\nTop 5 kombinacija (K, weights, CV RMSE):")
    print(top5.to_string(index=False))

    return best_k, best_weights


def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int | None = None,
    weights: str | None = None,
) -> tuple[KNeighborsRegressor, MinMaxScaler]:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if k is None or weights is None:
        k, weights = find_best_k(pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train)

    model = KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_knn(model: KNeighborsRegressor, scaler: MinMaxScaler, X: pd.DataFrame) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def run_knn_pipeline(csv_path: str | Path, city_name: str, k: int | None = None, weights: str | None = None) -> dict:
    df = prepare_city_data(csv_path, city_name=city_name)
    train_df, val_df, test_df = time_series_split(df)

    feature_cols = get_knn_feature_columns()
    X_train, y_train = get_X_y(train_df, feature_cols)
    X_val, y_val = get_X_y(val_df, feature_cols)
    X_test, y_test = get_X_y(test_df, feature_cols)

    model, scaler = train_knn(X_train, y_train, k=k, weights=weights)

    y_pred = predict_knn(model, scaler, X_test)
    result = evaluate_predictions(y_test, y_pred, model_name=f"KNN ({city_name})")

    return {
        "result": result,
        "model": model,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def save_knn_model(model, scaler, city_name: str, output_dir: str | Path = "results"):
    output_dir = Path(output_dir) / "knn"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"knn_model_{city_name}.pkl")
    joblib.dump(scaler, output_dir / f"knn_scaler_{city_name}.pkl")


if __name__ == "__main__":
    output = run_knn_pipeline("data/raw/beograd.csv", city_name="Beograd")
    print(output["result"])
    save_knn_model(output["model"], output["scaler"], "Beograd")