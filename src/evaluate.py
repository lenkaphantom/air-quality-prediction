"""
Zajednicki modul za evaluaciju modela.

Koristite OVU funkciju za sve modele (XGBoost, LightGBM, KNN, LSTM) kako bi
brojevi u finalnoj uporednoj tabeli (Faza 3) bili racunati na isti nacin.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_predictions(y_true, y_pred, model_name: str = "") -> dict:
    """
    Racuna RMSE, MAE i R2 za date stvarne i predvidjene vrednosti.
    Vraca dict pogodan za sakupljanje u zajednicku tabelu poredjenja.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Prima listu dict-ova (izlaz iz evaluate_predictions za razlicite modele)
    i vraca sortiranu tabelu poredjenja (najbolji RMSE prvi).
    """
    df = pd.DataFrame(results)
    return df.sort_values("RMSE").reset_index(drop=True)


if __name__ == "__main__":
    # Sanity check
    y_true = [10, 20, 30, 40]
    y_pred = [12, 18, 33, 37]
    print(evaluate_predictions(y_true, y_pred, model_name="test_model"))
