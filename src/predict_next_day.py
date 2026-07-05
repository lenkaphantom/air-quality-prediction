import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_prep import (
    load_raw_csv, clean_data, add_temporal_features,
    add_lag_features, add_trend_features, get_feature_columns,
    LAG_HOURS,
)
from src.aqi_utils import pm25_to_aqi_category

from src.models.knn_model import get_knn_feature_columns, predict_knn
from src.models.lstm_model import SEQUENCE_FEATURES, SEQ_LENGTH
from src.models.xgboost_model import get_xgb_feature_columns, predict_xgb
from src.models.lightgbm_model import get_lgbm_feature_columns, predict_lgbm
from src.train_models import ALL_CITIES, KEY_CITIES, RESULTS_DIR


def prepare_last_window(csv_path: str, city_name: str) -> pd.DataFrame:
    df = load_raw_csv(csv_path)
    df = clean_data(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_trend_features(df)
    df["city"] = city_name

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def _knn_predict_at_row(df: pd.DataFrame, row_mask, model, scaler) -> float:
    feature_cols = get_knn_feature_columns()
    row = df.loc[row_mask, feature_cols]
    return predict_knn(model, scaler, row)[0]


def _lstm_predict_at_row(df: pd.DataFrame, feature_time: pd.Timestamp, model, scaler) -> float:
    seq_start = feature_time - pd.Timedelta(hours=SEQ_LENGTH - 1)
    seq_mask = (df["date"] >= seq_start) & (df["date"] <= feature_time)
    seq_data = df.loc[seq_mask, SEQUENCE_FEATURES].values

    if len(seq_data) < SEQ_LENGTH:
        return np.nan

    n_features = seq_data.shape[1]
    seq_scaled = scaler.transform(seq_data.reshape(-1, n_features)).reshape(1, SEQ_LENGTH, n_features)
    return model.predict(seq_scaled, verbose=0)[0][0]


def _xgb_predict_at_row(df: pd.DataFrame, row_mask, model) -> float:
    feature_cols = get_xgb_feature_columns()
    row = df.loc[row_mask, feature_cols]
    return predict_xgb(model, row)[0]


def _lgbm_predict_at_row(df: pd.DataFrame, row_mask, model) -> float:
    feature_cols = get_lgbm_feature_columns()
    row = df.loc[row_mask, feature_cols]
    return predict_lgbm(model, row)[0]


def predict_next_day_hourly(
    csv_path: str,
    city_name: str,
    knn_model,
    knn_scaler,
    lstm_model,
    lstm_scaler,
    xgb_model=None,  
    lgbm_model=None,  
) -> pd.DataFrame:
    df = prepare_last_window(csv_path, city_name)

    last_date = df["date"].iloc[-1]
    next_day = (last_date + pd.Timedelta(hours=1)).normalize()

    rows = []
    for hour in range(24):
        target_time = next_day + pd.Timedelta(hours=hour)
        feature_time = target_time - pd.Timedelta(hours=24)

        row_mask = df["date"] == feature_time
        if not row_mask.any():
            continue

        knn_pred = max(_knn_predict_at_row(df, row_mask, knn_model, knn_scaler), 0.0)

        lstm_pred = _lstm_predict_at_row(df, feature_time, lstm_model, lstm_scaler)
        if not np.isnan(lstm_pred):
            lstm_pred = max(lstm_pred, 0.0)

        xgb_pred = None
        if xgb_model is not None:
            xgb_pred = max(_xgb_predict_at_row(df, row_mask, xgb_model), 0.0)

        lgbm_pred = None
        if lgbm_model is not None:
            lgbm_pred = max(_lgbm_predict_at_row(df, row_mask, lgbm_model), 0.0)

        rows.append({
            "time":      target_time,
            "knn_pm2_5":  round(float(knn_pred), 2),
            "knn_aqi":    pm25_to_aqi_category(knn_pred),
            "lstm_pm2_5": round(float(lstm_pred), 2) if not np.isnan(lstm_pred) else None,
            "lstm_aqi":   pm25_to_aqi_category(lstm_pred) if not np.isnan(lstm_pred) else "N/A",
            "xgb_pm2_5":  round(float(xgb_pred), 2) if xgb_pred is not None else None,
            "xgb_aqi":    pm25_to_aqi_category(xgb_pred) if xgb_pred is not None else "N/A",
            "lgbm_pm2_5": round(float(lgbm_pred), 2) if lgbm_pred is not None else None,
            "lgbm_aqi":   pm25_to_aqi_category(lgbm_pred) if lgbm_pred is not None else "N/A",
        })

    return pd.DataFrame(rows)


def run_next_day_prediction(
    csv_path: str,
    city_name: str,
    knn_model,
    knn_scaler,
    lstm_model,
    lstm_scaler,
    xgb_model=None,   
    lgbm_model=None,  
    output_dir: str | Path = "results",
):
    predictions = predict_next_day_hourly(
        csv_path, city_name,
        knn_model, knn_scaler,
        lstm_model, lstm_scaler,
        xgb_model, lgbm_model,
    )

    if predictions.empty:
        print("Nema dostupnih podataka za predikciju.")
        return predictions

    next_day_date = predictions["time"].iloc[0].date()

    print(f"\n{'='*96}")
    print(f"  Satna prognoza PM2.5 za {city_name} - {next_day_date}")
    print(f"{'='*96}")
    print(f"  {'Sat':<6} {'KNN':>8} {'KNN AQI':<18} {'LSTM':>8} {'LSTM AQI':<18} {'XGB':>8} {'XGB AQI':<18} {'LGBM':>8} {'LGBM AQI':<18}")
    print(f"  {'-'*92}")

    for _, row in predictions.iterrows():
        hour_str  = row["time"].strftime("%H:%M")
        lstm_str  = f"{row['lstm_pm2_5']:>8.2f}"  if row["lstm_pm2_5"]  is not None else f"{'N/A':>8}"
        xgb_str   = f"{row['xgb_pm2_5']:>8.2f}"   if row["xgb_pm2_5"]   is not None else f"{'N/A':>8}"
        lgbm_str  = f"{row['lgbm_pm2_5']:>8.2f}"  if row["lgbm_pm2_5"]  is not None else f"{'N/A':>8}"
        print(
            f"  {hour_str:<6} {row['knn_pm2_5']:>8.2f} {row['knn_aqi']:<18} "
            f"{lstm_str} {row['lstm_aqi']:<18} "
            f"{xgb_str} {row['xgb_aqi']:<18} "
            f"{lgbm_str} {row['lgbm_aqi']:<18}"
        )

    print(f"  {'-'*92}")
    knn_mean  = predictions["knn_pm2_5"].mean()
    lstm_mean = predictions["lstm_pm2_5"].dropna().mean() if predictions["lstm_pm2_5"].notna().any() else float("nan")
    xgb_mean  = predictions["xgb_pm2_5"].dropna().mean()  if "xgb_pm2_5"  in predictions and predictions["xgb_pm2_5"].notna().any()  else float("nan")
    lgbm_mean = predictions["lgbm_pm2_5"].dropna().mean() if "lgbm_pm2_5" in predictions and predictions["lgbm_pm2_5"].notna().any() else float("nan")
    print(f"  {'Prosek':<6} {knn_mean:>8.2f} {'':<18} {lstm_mean:>8.2f} {'':<18} {xgb_mean:>8.2f} {'':<18} {lgbm_mean:>8.2f}")
    print(f"{'='*96}")
    print(f"\nNAPOMENA: uporediti sa stvarnim izmerenim stanjem za {next_day_date}")

    out = predictions.copy()
    out.insert(0, "grad",  city_name)
    out.insert(1, "datum", str(next_day_date))
    out["sat"] = out["time"].dt.strftime("%H:%M")
    cols_to_save = ["grad", "datum", "sat",
                    "knn_pm2_5", "knn_aqi",
                    "lstm_pm2_5", "lstm_aqi",
                    "xgb_pm2_5", "xgb_aqi",
                    "lgbm_pm2_5", "lgbm_aqi"]
    out = out[cols_to_save]

    out_path = Path(output_dir) / f"forecast_{city_name}_{next_day_date}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Sacuvano: {out_path}")

    return predictions


if __name__ == "__main__":
    import joblib
    from tensorflow.keras.models import load_model

    parser = argparse.ArgumentParser(description="Satna predikcija PM2.5 za dan posle kraja dataseta.")
    parser.add_argument("--csv",         type=str, default=None)
    parser.add_argument("--city",        type=str, default="Beograd")
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--key",         action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    def _load_models(city):
        knn_model   = joblib.load(results_dir / "knn"       / f"knn_model_{city}.pkl")
        knn_scaler  = joblib.load(results_dir / "knn"       / f"knn_scaler_{city}.pkl")
        lstm_model  = load_model(results_dir  / "lstm"      / f"lstm_model_{city}.keras")
        lstm_scaler = joblib.load(results_dir / "lstm"      / f"lstm_scaler_{city}.pkl")
        xgb_model   = joblib.load(results_dir / "xgboost"  / f"xgboost_model_{city}.pkl")
        lgbm_model  = joblib.load(results_dir / "lightgbm" / f"lightgbm_model_{city}.pkl")
        return knn_model, knn_scaler, lstm_model, lstm_scaler, xgb_model, lgbm_model

    if args.all or args.key:
        all_city_csvs = {p.stem: str(p) for p in sorted(Path("data/raw").glob("*.csv"))}
        city_csvs = (
            {c: all_city_csvs[c] for c in KEY_CITIES if c in all_city_csvs}
            if args.key else all_city_csvs
        )
        for city_name, csv_path in city_csvs.items():
            model_files = [
                results_dir / "knn"       / f"knn_model_{city_name}.pkl",
                results_dir / "lstm"      / f"lstm_model_{city_name}.keras",
                results_dir / "xgboost"  / f"xgboost_model_{city_name}.pkl",
                results_dir / "lightgbm" / f"lightgbm_model_{city_name}.pkl",
            ]
            if not all(f.exists() for f in model_files):
                print(f"[PRESKACEM] {city_name} — istrenirani modeli nisu pronadjeni.")
                continue
            try:
                knn_m, knn_s, lstm_m, lstm_s, xgb_m, lgbm_m = _load_models(city_name)
                run_next_day_prediction(csv_path, city_name, knn_m, knn_s, lstm_m, lstm_s, xgb_m, lgbm_m)
            except Exception as e:
                print(f"[GRESKA] {city_name}: {e}")
    else:
        city = args.city
        csv_path = args.csv or f"data/raw/{city}.csv"
        knn_m, knn_s, lstm_m, lstm_s, xgb_m, lgbm_m = _load_models(city)
        run_next_day_prediction(csv_path, city, knn_m, knn_s, lstm_m, lstm_s, xgb_m, lgbm_m)
