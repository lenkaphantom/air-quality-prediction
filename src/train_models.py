from pathlib import Path
import pandas as pd

from src.aqi_utils import pm25_to_aqi_category
from src.evaluate import compare_models
from src.models.knn_model import run_knn_pipeline, save_knn_model, DEFAULT_K, DEFAULT_WEIGHTS
from src.models.lstm_model import run_lstm_pipeline, save_lstm_model

ALL_CITIES = {
    city_path.stem: str(city_path)
    for city_path in sorted(Path("data/raw").glob("*.csv"))
}

KEY_CITIES = ["Beograd", "Novi Sad", "Nis", "Bor", "Valjevo", "Kostolac", "Smederevo"]

RESULTS_DIR = Path("results")


def models_exist(city_name: str) -> dict[str, bool]:
    return {
        "knn":  (RESULTS_DIR / "knn"  / f"knn_model_{city_name}.pkl").exists(),
        "lstm": (RESULTS_DIR / "lstm" / f"lstm_model_{city_name}.keras").exists(),
    }


def train_city(
    csv_path: str,
    city_name: str,
    train_knn: bool = True,
    train_lstm: bool = True,
    fast: bool = False,
) -> dict:
    print(f"\n{'='*60}\n  {city_name}\n{'='*60}")

    knn_output = lstm_output = None

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

    if knn_output and lstm_output:
        knn_aqi  = pd.Series(knn_output["y_pred"]).apply(pm25_to_aqi_category)
        lstm_aqi = pd.Series(lstm_output["y_pred"]).apply(pm25_to_aqi_category)

        print("\n--- Raspodela AQI kategorija na test skupu ---")
        print("KNN:\n",  knn_aqi.value_counts().to_string())
        print("LSTM:\n", lstm_aqi.value_counts().to_string())

        comparison = compare_models([knn_output["result"], lstm_output["result"]])
        print("\n--- Uporedna tabela (KNN vs LSTM) ---")
        print(comparison.to_string(index=False))

    return {
        "knn":  knn_output,
        "lstm": lstm_output,
    }


def _ask_train(city_name: str, fast: bool) -> tuple[bool, bool]:
    exists = models_exist(city_name)
    train_knn_flag  = True
    train_lstm_flag = True

    for model_name, model_exists in exists.items():
        if model_exists:
            print(f"\n  [{model_name.upper()}] Model za {city_name} vec postoji.")
            while True:
                odgovor = input(f"  Ponovo trenirati {model_name.upper()}? [d/n]: ").strip().lower()
                if odgovor in ("d", "da", "y", "yes"):
                    break
                elif odgovor in ("n", "ne", "no"):
                    if model_name == "knn":
                        train_knn_flag = False
                    else:
                        train_lstm_flag = False
                    break
                print("  Unesite 'd' (da) ili 'n' (ne).")

    return train_knn_flag, train_lstm_flag


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treniranje KNN i LSTM modela za predikciju PM2.5.")
    parser.add_argument("--city", type=str, default="Beograd")
    parser.add_argument("--all",  action="store_true", help="Svi gradovi")
    parser.add_argument("--key",  action="store_true", help=f"Kljucni gradovi: {', '.join(KEY_CITIES)}")
    parser.add_argument("--fast", action="store_true", help="Brzi rezim: preskoci GridSearchCV za KNN")
    parser.add_argument("--force", action="store_true", help="Treniraj ponovo cak i ako modeli postoje (bez pitanja)")
    args = parser.parse_args()

    if args.all or args.key:
        city_subset = ALL_CITIES if args.all else {
            c: ALL_CITIES[c] for c in KEY_CITIES if c in ALL_CITIES
        }
        all_results = []
        failed = []

        for city_name, csv_path in city_subset.items():
            if args.force:
                do_knn, do_lstm = True, True
            else:
                do_knn, do_lstm = _ask_train(city_name, args.fast)

            if not do_knn and not do_lstm:
                print(f"  Preskacam {city_name}.")
                continue

            try:
                output = train_city(csv_path, city_name, train_knn=do_knn, train_lstm=do_lstm, fast=args.fast)
                row = {"Grad": city_name}
                if output["knn"]:
                    row |= {"KNN RMSE": output["knn"]["result"]["RMSE"],
                             "KNN MAE":  output["knn"]["result"]["MAE"],
                             "KNN R2":   output["knn"]["result"]["R2"]}
                if output["lstm"]:
                    row |= {"LSTM RMSE": output["lstm"]["result"]["RMSE"],
                             "LSTM MAE":  output["lstm"]["result"]["MAE"],
                             "LSTM R2":   output["lstm"]["result"]["R2"]}
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
            do_knn, do_lstm = True, True
        else:
            do_knn, do_lstm = _ask_train(args.city, args.fast)

        if do_knn or do_lstm:
            train_city(csv_path, args.city, train_knn=do_knn, train_lstm=do_lstm, fast=args.fast)
        else:
            print("Treniranje preskoceno.")
