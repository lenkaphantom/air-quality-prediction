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


def predict_next_day_hourly(
    csv_path: str,
    city_name: str,
    knn_model,
    knn_scaler,
    lstm_model,
    lstm_scaler,
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

        knn_pred = _knn_predict_at_row(df, row_mask, knn_model, knn_scaler)
        knn_pred = max(knn_pred, 0.0)

        lstm_pred = _lstm_predict_at_row(df, feature_time, lstm_model, lstm_scaler)
        if not np.isnan(lstm_pred):
            lstm_pred = max(lstm_pred, 0.0)

        rows.append({
            "time": target_time,
            "knn_pm2_5": round(float(knn_pred), 2),
            "knn_aqi": pm25_to_aqi_category(knn_pred),
            "lstm_pm2_5": round(float(lstm_pred), 2) if not np.isnan(lstm_pred) else None,
            "lstm_aqi": pm25_to_aqi_category(lstm_pred) if not np.isnan(lstm_pred) else "N/A",
        })

    return pd.DataFrame(rows)


def run_next_day_prediction(
    csv_path: str,
    city_name: str,
    knn_model,
    knn_scaler,
    lstm_model,
    lstm_scaler,
    output_dir: str | Path = "results",
):
    predictions = predict_next_day_hourly(
        csv_path, city_name, knn_model, knn_scaler, lstm_model, lstm_scaler
    )

    if predictions.empty:
        print("Nema dostupnih podataka za predikciju.")
        return predictions

    next_day_date = predictions["time"].iloc[0].date()

    print(f"\n{'='*72}")
    print(f"  Satna prognoza PM2.5 za {city_name} - {next_day_date}")
    print(f"{'='*72}")
    print(f"  {'Sat':<6} {'KNN PM2.5':>10}  {'KNN AQI':<20} {'LSTM PM2.5':>10}  {'LSTM AQI':<20}")
    print(f"  {'-'*68}")

    for _, row in predictions.iterrows():
        hour_str = row["time"].strftime("%H:%M")
        lstm_pm25_str = f"{row['lstm_pm2_5']:>10.2f}" if row["lstm_pm2_5"] is not None else f"{'N/A':>10}"
        print(
            f"  {hour_str:<6} {row['knn_pm2_5']:>10.2f}  {row['knn_aqi']:<20} "
            f"{lstm_pm25_str}  {row['lstm_aqi']:<20}"
        )

    print(f"  {'-'*68}")
    knn_mean = predictions["knn_pm2_5"].mean()
    lstm_vals = predictions["lstm_pm2_5"].dropna()
    lstm_mean = lstm_vals.mean() if not lstm_vals.empty else float("nan")
    print(f"  {'Prosek':<6} {knn_mean:>10.2f}  {'':<20} {lstm_mean:>10.2f}")
    print(f"{'='*72}")
    print(f"\nNAPOMENA: uporediti sa stvarnim izmerenim stanjem za {next_day_date}")
    print("(npr. sa Open-Meteo/SEPA sajta) kao nezavisnu proveru van test skupa.")

    out = predictions[["time", "knn_pm2_5", "knn_aqi", "lstm_pm2_5", "lstm_aqi"]].copy()
    out.insert(0, "grad", city_name)
    out.insert(1, "datum", str(next_day_date))
    out["sat"] = out["time"].dt.strftime("%H:%M")
    out = out[["grad", "datum", "sat", "knn_pm2_5", "knn_aqi", "lstm_pm2_5", "lstm_aqi"]]

    out_path = Path(output_dir) / f"forecast_{city_name}_{next_day_date}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Sacuvano: {out_path}")

    return predictions


if __name__ == "__main__":
    import joblib
    from tensorflow.keras.models import load_model

    parser = argparse.ArgumentParser(description="Satna predikcija PM2.5 za dan posle kraja dataseta.")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--city", type=str, default="Beograd")
    parser.add_argument("--all", action="store_true", help="Predvidja za sve gradove koji imaju istrenirane modele")
    parser.add_argument("--key", action="store_true", help="Predvidja samo za kljucne gradove")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    def _load_models(city):
        knn_model   = joblib.load(results_dir / "knn" / f"knn_model_{city}.pkl")
        knn_scaler  = joblib.load(results_dir / "knn" / f"knn_scaler_{city}.pkl")
        lstm_model  = load_model(results_dir / "lstm" / f"lstm_model_{city}.keras")
        lstm_scaler = joblib.load(results_dir / "lstm" / f"lstm_scaler_{city}.pkl")
        return knn_model, knn_scaler, lstm_model, lstm_scaler

    if args.all or args.key:
        summary_rows = []
        detailed_rows = []
        failed = []

        all_city_csvs = {
            p.stem: str(p) for p in sorted(Path("data/raw").glob("*.csv"))
        }
        city_csvs = (
            {c: all_city_csvs[c] for c in KEY_CITIES if c in all_city_csvs}
            if args.key else all_city_csvs
        )

        for city_name, csv_path in city_csvs.items():
            model_files = [
                results_dir / "knn" / f"knn_model_{city_name}.pkl",
                results_dir / "knn" / f"knn_scaler_{city_name}.pkl",
                results_dir / "lstm" / f"lstm_model_{city_name}.keras",
                results_dir / "lstm" / f"lstm_scaler_{city_name}.pkl",
            ]
            if not all(f.exists() for f in model_files):
                print(f"[PRESKACEM] {city_name} — istrenirani modeli nisu pronadjeni u {results_dir}/")
                failed.append(city_name)
                continue

            try:
                knn_m, knn_s, lstm_m, lstm_s = _load_models(city_name)
                preds = predict_next_day_hourly(csv_path, city_name, knn_m, knn_s, lstm_m, lstm_s)
                if preds.empty:
                    failed.append(city_name)
                    continue

                next_day_date = preds["time"].iloc[0].date()
                knn_mean  = preds["knn_pm2_5"].mean()
                lstm_mean = preds["lstm_pm2_5"].dropna().mean()
                knn_max   = preds["knn_pm2_5"].max()
                lstm_max  = preds["lstm_pm2_5"].dropna().max() if not preds["lstm_pm2_5"].dropna().empty else float("nan")

                knn_dominant  = preds["knn_aqi"].mode().iloc[0]
                lstm_dominant = preds["lstm_aqi"].mode().iloc[0]

                summary_rows.append({
                    "Grad":               city_name,
                    "Datum prognoze":     str(next_day_date),
                    "KNN prosek PM2.5":   round(knn_mean, 2),
                    "KNN max PM2.5":      round(knn_max, 2),
                    "KNN AQI (dom.)":     knn_dominant,
                    "LSTM prosek PM2.5":  round(lstm_mean, 2),
                    "LSTM max PM2.5":     round(lstm_max, 2),
                    "LSTM AQI (dom.)":    lstm_dominant,
                })

                preds.insert(0, "grad", city_name)
                preds.insert(1, "datum", str(next_day_date))
                preds["sat"] = preds["time"].dt.strftime("%H:%M")
                detailed_rows.append(preds[["grad", "datum", "sat",
                                            "knn_pm2_5", "knn_aqi",
                                            "lstm_pm2_5", "lstm_aqi"]])

                print(f"  OK  {city_name:<25} KNN prosek={knn_mean:.1f}  LSTM prosek={lstm_mean:.1f}")
            except Exception as e:
                print(f"[GRESKA] {city_name}: {e}")
                failed.append(city_name)

        if summary_rows:
            summary = pd.DataFrame(summary_rows).sort_values("KNN prosek PM2.5", ascending=False).reset_index(drop=True)

            print(f"\n{'='*90}")
            print(f"  SUMARNI PREGLED PROGNOZA — {summary['Datum prognoze'].iloc[0]}")
            print(f"{'='*90}")
            print(summary.to_string(index=False))

            out_path = results_dir / "forecast_summary_all_cities.csv"
            out_path.parent.mkdir(exist_ok=True)
            summary.to_csv(out_path, index=False)
            print(f"\nSacuvano (sumarni): {out_path}")

        if detailed_rows:
            detailed = pd.concat(detailed_rows, ignore_index=True)
            det_path = results_dir / "forecast_hourly_all_cities.csv"
            detailed.to_csv(det_path, index=False)
            print(f"Sacuvano (satni):   {det_path}")

        if failed:
            print(f"\nNisu obradjeni: {', '.join(failed)}")

    else:
        city = args.city
        csv_path = args.csv or f"data/raw/{city}.csv"
        knn_m, knn_s, lstm_m, lstm_s = _load_models(city)
        run_next_day_prediction(csv_path, city, knn_m, knn_s, lstm_m, lstm_s, output_dir=results_dir)