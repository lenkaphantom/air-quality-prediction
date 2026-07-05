import argparse
import joblib
from pathlib import Path

from tensorflow.keras.models import load_model

from src.train_models import (
    ALL_CITIES, KEY_CITIES, RESULTS_DIR,
    models_exist, train_city, _ask_train,
)
from src.predict_next_day import run_next_day_prediction


def _load_models(city_name: str):
    knn_model   = joblib.load(RESULTS_DIR / "knn"       / f"knn_model_{city_name}.pkl")
    knn_scaler  = joblib.load(RESULTS_DIR / "knn"       / f"knn_scaler_{city_name}.pkl")
    lstm_model  = load_model(RESULTS_DIR  / "lstm"      / f"lstm_model_{city_name}.keras")
    lstm_scaler = joblib.load(RESULTS_DIR / "lstm"      / f"lstm_scaler_{city_name}.pkl")
    xgb_model   = joblib.load(RESULTS_DIR / "xgboost"   / f"xgboost_model_{city_name}.pkl")
    lgbm_model  = joblib.load(RESULTS_DIR / "lightgbm"  / f"lightgbm_model_{city_name}.pkl")
    return knn_model, knn_scaler, lstm_model, lstm_scaler, xgb_model, lgbm_model


def run_pipeline(csv_path: str, city_name: str, fast: bool, force: bool, no_train: bool):
    if not no_train:
        exists = models_exist(city_name)
        any_missing = not all(exists.values())

        if force:
            do_knn = do_lstm = do_xgb = do_lgbm = True
        elif any_missing:
            for model_name, model_exists in exists.items():
                if not model_exists:
                    print(f"\n  [{model_name.upper()}] Model za {city_name} ne postoji, treniram automatski.")
            do_knn, do_lstm, do_xgb, do_lgbm = _ask_train(city_name, fast)
        else:
            do_knn, do_lstm, do_xgb, do_lgbm = _ask_train(city_name, fast)

        if any([do_knn, do_lstm, do_xgb, do_lgbm]):
            train_city(
                csv_path, city_name,
                train_knn=do_knn, train_lstm=do_lstm,
                train_xgb=do_xgb, train_lgbm=do_lgbm,
                fast=fast,
            )

    exists_after = models_exist(city_name)
    if not all(exists_after.values()):
        missing = [m.upper() for m, e in exists_after.items() if not e]
        print(f"\n[UPOZORENJE] Nedostaju modeli: {', '.join(missing)} - predikcija preskocena.")
        return

    print(f"\n{'='*60}")
    print(f"  Ucitavam modele za {city_name}...")
    knn_m, knn_s, lstm_m, lstm_s, xgb_m, lgbm_m = _load_models(city_name)

    run_next_day_prediction(
        csv_path, city_name,
        knn_m, knn_s,
        lstm_m, lstm_s,
        xgb_m, lgbm_m,
        output_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trening + satna predikcija PM2.5 za naredni dan."
    )
    parser.add_argument("--city",     type=str, default="Beograd", help="Naziv grada")
    parser.add_argument("--all",      action="store_true", help="Svi gradovi")
    parser.add_argument("--key",      action="store_true", help="Kljucni gradovi")
    parser.add_argument("--fast",     action="store_true", help="Brzi rezim (bez GridSearchCV)")
    parser.add_argument("--force",    action="store_true", help="Treniraj ponovo bez pitanja")
    parser.add_argument("--no-train", action="store_true", help="Preskoci treniranje, samo predikcija")
    args = parser.parse_args()

    if args.all or args.key:
        city_subset = ALL_CITIES if args.all else {
            c: ALL_CITIES[c] for c in KEY_CITIES if c in ALL_CITIES
        }
        for city_name, csv_path in city_subset.items():
            run_pipeline(csv_path, city_name, fast=args.fast, force=args.force, no_train=args.no_train)
    else:
        csv_path = ALL_CITIES.get(args.city, f"data/raw/{args.city}.csv")
        run_pipeline(csv_path, args.city, fast=args.fast, force=args.force, no_train=args.no_train)
