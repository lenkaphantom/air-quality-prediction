# Prostorno-vremenska analiza i predikcija AQI u Srbiji

Projekat iz predmeta Računarska inteligencija — Teodora Aleksić i Lenka Nikolić.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Struktura projekta

```
air-quality-prediction/
├── data/
│   ├── raw/              # sirovi CSV-jevi, jedan po gradu (28 fajlova)
│   └── processed/        # ocisceni i feature-engineered podaci
├── src/
│   ├── data_prep.py      # ciscenje, interpolacija, lag i temporalni feature-i
│   ├── split.py           # time-series split (70/20/10)
│   ├── evaluate.py        # RMSE/MAE/R2 - ista funkcija za sve modele
│   ├── aqi_utils.py       # mapiranje PM2.5 -> AQI kategorija (SEPA)
│   └── models/
│       ├── xgboost_model.py    # Teodora
│       ├── lightgbm_model.py   # Teodora
│       ├── knn_model.py        # Lenka
│       └── lstm_model.py       # Lenka
├── notebooks/              # EDA, vizuelizacija - slobodno, svako svoje
├── results/                 # sacuvani modeli (.pkl/.h5), grafici, metrike
├── requirements.txt
└── .gitignore
```

Svaki folder sa Python fajlovima koji se importuju kao paket (`src/`,
`src/models/`) ima `__init__.py` (prazan fajl) — bez njega Python ne
tretira folder kao paket i importi kao `from src.data_prep import ...`
ne rade. `notebooks/` i `results/` nemaju `__init__.py` jer se ne
importuju kao kod, samo sadrže fajlove.

## Zajednički deo (VEĆ NAPRAVLJEN — ne diraj bez dogovora)

Da bi rezultati XGBoost/LightGBM i KNN/LSTM bili uporedivi u Fazi 3,
oba člana tima MORAJU koristiti identičan pipeline pripreme podataka
i identičnu podelu na train/val/test. Zato su ova tri fajla zajednička:

- **`src/data_prep.py`** — `prepare_city_data(filepath, city_name)` učitava
  jedan CSV, čisti ga (interpolacija po vremenu), dodaje lag_1h...lag_48h
  za PM2.5 i PM10, dodaje hour/month/day_of_week, i target kolonu
  (`target_pm2_5_t+24h`). Vraća čist DataFrame bez NaN redova.
- **`src/split.py`** — `time_series_split(df)` deli po fiksnim datumima
  (trening do jun 2023, validacija do jun 2025, test do jun 2026),
  bez mešanja, hronološki.
- **`src/evaluate.py`** — `evaluate_predictions(y_true, y_pred, model_name)`
  računa RMSE/MAE/R² na isti način za sve modele.

Ako nešto od ovoga treba izmeniti (npr. drugačiji broj lag sati), **menja
se zajedno, pa se odmah `git pull`-uje** pre nego što bilo ko ponovo
trenira model — inače brojevi nisu uporedivi.

## Kako koristiti pipeline (primer)

```python
from src.data_prep import prepare_city_data, get_feature_columns
from src.split import time_series_split, get_X_y
from src.evaluate import evaluate_predictions

df = prepare_city_data("data/raw/beograd.csv", city_name="Beograd")
train_df, val_df, test_df = time_series_split(df)

feature_cols = get_feature_columns(df)
X_train, y_train = get_X_y(train_df, feature_cols)
X_test, y_test = get_X_y(test_df, feature_cols)

# ... treniranje modela ...

result = evaluate_predictions(y_test, y_pred, model_name="XGBoost")
```

## Podela rada

| Ko | Modeli | Fajl |
|---|---|---|
| Teodora | XGBoost, LightGBM | `src/models/xgboost_model.py`, `lightgbm_model.py` |
| Lenka | KNN, LSTM | `src/models/knn_model.py`, `lstm_model.py` |

Faza 3 (uporedna evaluacija svih modela) radi se zajedno, kad oba
para modela budu istrenirana i evaluirana pomoću `evaluate.py`.

## Napomena o podacima

CSV fajlovi (28 gradova, ~87.600 redova svaki) se NE guraju na GitHub
(prevelike su, u `.gitignore`-u). Dogovorite se gde ih delite (Drive,
lokalno) i svako ih stavi u svoj lokalni `data/raw/`.
