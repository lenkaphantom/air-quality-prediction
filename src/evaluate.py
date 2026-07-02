import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_predictions(y_true, y_pred, model_name: str = "") -> dict:
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
    df = pd.DataFrame(results)
    return df.sort_values("RMSE").reset_index(drop=True)


if __name__ == "__main__":
    y_true = [10, 20, 30, 40]
    y_pred = [12, 18, 33, 37]
    print(evaluate_predictions(y_true, y_pred, model_name="test_model"))
