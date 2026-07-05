from pathlib import Path
import pandas as pd

from src.aqi_utils import pm25_to_aqi_category
from src.evaluate import compare_models
from src.models.knn_model import run_knn_pipeline, save_knn_model, DEFAULT_K, DEFAULT_WEIGHTS
from src.models.lstm_model import run_lstm_pipeline, save_lstm_model
from src.models.xgboost_model import run_xgb_pipeline, save_xgb_model, DEFAULT_N_ESTIMATORS, DEFAULT_MAX_DEPTH, DEFAULT_LEARNING_RATE, DEFAULT_SUBSAMPLE
from src.models.lightgbm_model import run_lgbm_pipeline, save_lgbm_model, DEFAULT_NUM_LEAVES

ALL_CITIES = {
    city_path.stem: str(city_path)
    for city_path in sorted(Path("data/raw").glob("*.csv"))
}

KEY_CITIES = ["Beograd", "Novi Sad", "Nis", "Bor", "Valjevo", "Kostolac", "Smederevo", "Kopaonik"]

RESULTS_DIR = Path("results")


def models_exist(city_name: str) -> dict[str, bool]:
    return {
        "knn":      (RESULTS_DIR / "knn"       / f"knn_model_{city_name}.pkl").exists(),
        "lstm":     (RESULTS_DIR / "lstm"      / f"lstm_model_{city_name}.keras").exists(),
        "xgboost":  (RESULTS_DIR / "xgboost"  / f"xgboost_model_{city_name}.pkl").exists(),
        "lightgbm": (RESULTS_DIR / "lightgbm" / f"lightgbm_model_{city_name}.pkl").exists(),
    }


def train_city(
    csv_path: str,
    city_name: str,
    train_knn: bool = True,
    train_lstm: bool = True,
    train_xgb: bool = True,       
    train_lgbm: bool = True,      
    fast: bool = False,
) -> dict:
    print(f"\n{'='*60}\n  {city_name}\n{'='*60}")

    knn_output = lstm_output = xgb_output = lgbm_output = None

    if train_knn:
        print("\n--- Treniram KNN ---")
        if fast:
            print(f"  [brzi rezim] K={DEFAULT_K}, weights={DEFAULT_WEIGHTS!r} (bez GridSearchCV)")
        knn_output = run_knn_pipeline(
            csv_path, city_name=city_name,
            k=DEFAULT_K if fast else None,
            weights=DEFAULT_WEIGHTS if fast else None,
        )
        print(knn_output["result"])
        save_knn_model(knn_output["model"], knn_output["scaler"], city_name)
    else:
        print("\n[KNN] preskoceno.")

    if train_lstm:
        print("\n--- Treniram LSTM ---")
        lstm_output = run_lstm_pipeline(csv_path, city_name=city_name)
        print(lstm_output["result"])
        save_lstm_model(lstm_output["model"], lstm_output["scaler"], city_name)
    else:
        print("\n[LSTM] preskoceno.")

    if train_xgb:
        print("\n--- Treniram XGBoost ---")
        if fast:
            print(f"  [brzi rezim] n_estimators={DEFAULT_N_ESTIMATORS}, max_depth={DEFAULT_MAX_DEPTH} (bez GridSearchCV)")
        xgb_params = {
            "n_estimators":  DEFAULT_N_ESTIMATORS,
            "max_depth":     DEFAULT_MAX_DEPTH,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "subsample":     DEFAULT_SUBSAMPLE,
        } if fast else None
        xgb_output = run_xgb_pipeline(csv_path, city_name=city_name, params=xgb_params)
        print(xgb_output["result"])
        save_xgb_model(xgb_output["model"], city_name)
    else:
        print("\n[XGBoost] preskoceno.")

    if train_lgbm:
        print("\n--- Treniram LightGBM ---")
        if fast:
            print(f"  [brzi rezim] num_leaves={DEFAULT_NUM_LEAVES} (bez GridSearchCV)")
        lgbm_params = {
            "num_leaves":    DEFAULT_NUM_LEAVES,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "subsample":     DEFAULT_SUBSAMPLE,
        } if fast else None
        lgbm_output = run_lgbm_pipeline(csv_path, city_name=city_name, params=lgbm_params)
        print(lgbm_output["result"])
        save_lgbm_model(lgbm_output["model"], city_name)
    else:
        print("\n[LightGBM] preskoceno.")

    all_results = [
        out["result"]
        for out in [knn_output, lstm_output, xgb_output, lgbm_output]
        if out is not None
    ]
    if len(all_results) >= 2:
        print("\n--- Raspodela AQI kategorija na test skupu ---")
        for label, out in [("KNN", knn_output), ("LSTM", lstm_output),
                           ("XGBoost", xgb_output), ("LightGBM", lgbm_output)]:
            if out is not None:
                aqi = pd.Series(out["y_pred"]).apply(pm25_to_aqi_category)
                print(f"{label}:\n", aqi.value_counts().to_string())

        comparison = compare_models(all_results)
        print("\n--- Uporedna tabela (svi modeli) ---")
        print(comparison.to_string(index=False))

    return {
        "knn":      knn_output,
        "lstm":     lstm_output,
        "xgboost":  xgb_output,
        "lightgbm": lgbm_output,
    }


def _ask_train(city_name: str, fast: bool) -> tuple[bool, bool, bool, bool]:
    exists = models_exist(city_name)
    flags = {"knn": True, "lstm": True, "xgboost": True, "lightgbm": True}

    for model_name, model_exists in exists.items():
        if model_exists:
            print(f"\n  [{model_name.upper()}] Model za {city_name} vec postoji.")
            while True:
                odgovor = input(f"  Ponovo trenirati {model_name.upper()}? [d/n]: ").strip().lower()
                if odgovor in ("d", "da", "y", "yes"):
                    break
                elif odgovor in ("n", "ne", "no"):
                    flags[model_name] = False
                    break
                print("  Unesite 'd' (da) ili 'n' (ne).")

    return flags["knn"], flags["lstm"], flags["xgboost"], flags["lightgbm"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treniranje svih modela za predikciju PM2.5.")
    parser.add_argument("--city",  type=str, default="Beograd")
    parser.add_argument("--all",   action="store_true", help="Svi gradovi")
    parser.add_argument("--key",   action="store_true", help=f"Kljucni gradovi: {', '.join(KEY_CITIES)}")
    parser.add_argument("--fast",  action="store_true", help="Brzi rezim: preskoci GridSearchCV")
    parser.add_argument("--force", action="store_true", help="Treniraj ponovo cak i ako modeli postoje")
    args = parser.parse_args()

    if args.all or args.key:
        city_subset = ALL_CITIES if args.all else {
            c: ALL_CITIES[c] for c in KEY_CITIES if c in ALL_CITIES
        }
        all_results = []
        failed = []

        for city_name, csv_path in city_subset.items():
            if args.force:
                do_knn = do_lstm = do_xgb = do_lgbm = True
            else:
                do_knn, do_lstm, do_xgb, do_lgbm = _ask_train(city_name, args.fast)

            if not any([do_knn, do_lstm, do_xgb, do_lgbm]):
                print(f"  Preskacam {city_name}.")
                continue

            try:
                output = train_city(
                    csv_path, city_name,
                    train_knn=do_knn, train_lstm=do_lstm,
                    train_xgb=do_xgb, train_lgbm=do_lgbm,
                    fast=args.fast,
                )
                row = {"Grad": city_name}
                for model_key, label_prefix in [
                    ("knn", "KNN"), ("lstm", "LSTM"),
                    ("xgboost", "XGB"), ("lightgbm", "LGBM"),
                ]:
                    if output[model_key]:
                        res = output[model_key]["result"]
                        row |= {
                            f"{label_prefix} RMSE": res["RMSE"],
                            f"{label_prefix} MAE":  res["MAE"],
                            f"{label_prefix} R2":   res["R2"],
                        }
                all_results.append(row)
            except Exception as e:
                print(f"\n[GRESKA] {city_name}: {e}")
                failed.append(city_name)

        if all_results:
            summary = pd.DataFrame(all_results)
            if "KNN RMSE" in summary.columns:
                summary = summary.sort_values("KNN RMSE")
            summary = summary.reset_index(drop=True)

            print(f"\n{'='*80}")
            print("  SUMARNI IZVESTAJ TRENIRANJA")
            print(f"{'='*80}")
            print(summary.to_string(index=False))

            out_path = RESULTS_DIR / "summary_all_cities.csv"
            out_path.parent.mkdir(exist_ok=True)
            summary.to_csv(out_path, index=False)
            print(f"\nSacuvano: {out_path}")

        if failed:
            print(f"\nNisu obradjeni: {', '.join(failed)}")

    else:
        csv_path = ALL_CITIES.get(args.city, f"data/raw/{args.city}.csv")

        if args.force:
            do_knn = do_lstm = do_xgb = do_lgbm = True
        else:
            do_knn, do_lstm, do_xgb, do_lgbm = _ask_train(args.city, args.fast)

        if any([do_knn, do_lstm, do_xgb, do_lgbm]):
            train_city(
                csv_path, args.city,
                train_knn=do_knn, train_lstm=do_lstm,
                train_xgb=do_xgb, train_lgbm=do_lgbm,
                fast=args.fast,
            )
        else:
            print("Treniranje preskoceno.")
