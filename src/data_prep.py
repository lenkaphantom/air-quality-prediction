import pandas as pd
import numpy as np
from pathlib import Path

POLLUTANT_COLS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "sulphur_dioxide",
    "ozone",
    "nitrogen_dioxide",
]

LAG_HOURS = 48
FORECAST_HORIZON = 24
TARGET_COL = "pm2_5"


def load_raw_csv(filepath: str | Path) -> pd.DataFrame:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            lines = f.readlines()

    header_row = next(i for i, line in enumerate(lines) if line.startswith("time"))

    df = pd.read_csv(filepath, skiprows=header_row, encoding="utf-8", encoding_errors="replace")

    rename_map = {}
    for col in df.columns:
        clean = col.split("(")[0].strip().lower().replace(" ", "_").replace(".", "_")
        rename_map[col] = clean
    df = df.rename(columns=rename_map)

    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    keep_cols = ["date"] + POLLUTANT_COLS
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Nedostaju ocekivane kolone posle parsiranja: {missing}. "
            f"Dostupne kolone su: {df.columns.tolist()}"
        )
    df = df[keep_cols]

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("date")

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "date"

    df[POLLUTANT_COLS] = df[POLLUTANT_COLS].interpolate(method="time")
    df[POLLUTANT_COLS] = df[POLLUTANT_COLS].bfill().ffill()

    df = df.reset_index()
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["heating_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["pm2_5", "pm10"]:
        for lag in range(1, LAG_HOURS + 1):
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    shifted = df["pm2_5"].shift(1)

    for window in [6, 12, 24]:
        df[f"pm2_5_rolling_mean_{window}h"] = shifted.rolling(window).mean()
        df[f"pm2_5_rolling_std_{window}h"] = shifted.rolling(window).std()
        df[f"pm2_5_rolling_max_{window}h"] = shifted.rolling(window).max()

    for window in [3, 6]:
        df[f"pm2_5_diff_{window}h"] = df["pm2_5"].shift(1) - df["pm2_5"].shift(1 + window)

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_pm2_5_t+24h"] = df[TARGET_COL].shift(-FORECAST_HORIZON)
    return df


def prepare_city_data(filepath: str | Path, city_name: str | None = None) -> pd.DataFrame:
    df = load_raw_csv(filepath)
    df = clean_data(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_trend_features(df)
    df = add_target(df)

    if city_name is not None:
        df["city"] = city_name

    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "city", "target_pm2_5_t+24h"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    example_path = Path(__file__).parent.parent / "data" / "raw" / "beograd.csv"
    if example_path.exists():
        result = prepare_city_data(example_path, city_name="Beograd")
        print(result.shape)
        print(result.head())
    else:
        print(f"Test fajl ne postoji: {example_path}")
        print("Ovo je ocekivano dok se ne ubace pravi podaci u data/raw/")