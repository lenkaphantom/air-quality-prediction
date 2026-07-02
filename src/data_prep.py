"""
Zajednicki modul za pripremu podataka.

VAZNO: Ovaj fajl mora biti IDENTICAN za oba clana tima. Ako se menja,
menja se zajedno i odmah se sinhronizuje (git pull pre nego sto se
bilo koji model trenira na sveze podacima).

Ulaz: raw CSV fajl za jedan grad (kolone: date, pm10, pm2_5,
      carbon_monoxide, sulphur_dioxide, ozone, nitrogen_dioxide)
Izlaz: DataFrame spreman za treniranje - sa lag feature-ima i
       temporalnim atributima, bez NaN vrednosti.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Nazivi kolona zagadjujucih materija nad kojima pravimo lag feature-e
POLLUTANT_COLS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "sulphur_dioxide",
    "ozone",
    "nitrogen_dioxide",
]

# Koliko sati unazad gledamo za lag feature-e (Faza 1 specifikacije: 48h)
LAG_HOURS = 48

# Cilj predikcije: t + 24h unapred
FORECAST_HORIZON = 24

# Primarni target je PM2.5 (sekundarni PM10 - ako zatreba posebno)
TARGET_COL = "pm2_5"


def load_raw_csv(filepath: str | Path) -> pd.DataFrame:
    """Ucitava sirovi CSV fajl za jedan grad i parsira datume."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ciscenje podataka:
    - postavlja date kao index (radi lakse interpolacije po vremenu)
    - interpolira nedostajuce vrednosti (linearna interpolacija po vremenu)
    - popunjava eventualne ostatke na krajevima (bfill/ffill)
    """
    df = df.copy()
    df = df.set_index("date")

    # Osiguravamo da imamo kompletan satni raspon (popunjava rupe u vremenu)
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "date"

    # Linearna interpolacija po vremenskom indeksu
    df[POLLUTANT_COLS] = df[POLLUTANT_COLS].interpolate(method="time")

    # Ako i dalje ima NaN na pocetku/kraju serije - popuni najblizom vrednoscu
    df[POLLUTANT_COLS] = df[POLLUTANT_COLS].bfill().ffill()

    df = df.reset_index()
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje temporalne atribute: hour, month, day_of_week."""
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje lagovane atribute za PM2.5 i PM10 za prethodnih 48h
    (lag_1h ... lag_48h), kako je definisano u specifikaciji.
    """
    df = df.copy()
    for col in ["pm2_5", "pm10"]:
        for lag in range(1, LAG_HOURS + 1):
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje ciljnu kolonu: PM2.5 vrednost 24h unapred (t + 24h)."""
    df = df.copy()
    df["target_pm2_5_t+24h"] = df[TARGET_COL].shift(-FORECAST_HORIZON)
    return df


def prepare_city_data(filepath: str | Path, city_name: str | None = None) -> pd.DataFrame:
    """
    Glavna funkcija - prima putanju do raw CSV-a jednog grada i vraca
    potpuno pripremljen DataFrame (ciscenje + lag + temporalni + target),
    bez NaN redova (koji nastaju usled shift-a na pocetku/kraju serije).

    Ovo je funkcija koju obe pozivate identicno za svaki grad.
    """
    df = load_raw_csv(filepath)
    df = clean_data(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_target(df)

    if city_name is not None:
        df["city"] = city_name

    # Prvih LAG_HOURS redova nema kompletne lagove,
    # poslednjih FORECAST_HORIZON redova nema target - izbacujemo ih
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Vraca listu kolona koje se koriste kao ulazni feature-i za model
    (sve osim date, city i target kolone).
    """
    exclude = {"date", "city", "target_pm2_5_t+24h"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    # Brzi test na jednom gradu - promeni putanju po potrebi
    example_path = Path(__file__).parent.parent / "data" / "raw" / "beograd.csv"
    if example_path.exists():
        result = prepare_city_data(example_path, city_name="Beograd")
        print(result.shape)
        print(result.head())
    else:
        print(f"Test fajl ne postoji: {example_path}")
        print("Ovo je ocekivano dok se ne ubace pravi podaci u data/raw/")
