import numpy as np
import pandas as pd

from src.data_prep import prepare_city_data, get_feature_columns
from src.split import time_series_split, get_X_y
from src.evaluate import evaluate_predictions

from src.models.knn_model import get_knn_feature_columns, train_knn, predict_knn
from src.models.lstm_model import (
    build_sequences, scale_sequence_data, train_lstm,
)


def check_knn_overfit(csv_path: str, city_name: str):
    print(f"\n{'='*60}\nKNN - {city_name}\n{'='*60}")

    df = prepare_city_data(csv_path, city_name=city_name)
    train_df, val_df, test_df = time_series_split(df)

    feature_cols = get_knn_feature_columns()
    X_train, y_train = get_X_y(train_df, feature_cols)
    X_val, y_val = get_X_y(val_df, feature_cols)
    X_test, y_test = get_X_y(test_df, feature_cols)

    model, scaler = train_knn(X_train, y_train, k=150, weights="distance")

    results = []
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = predict_knn(model, scaler, X)
        results.append(evaluate_predictions(y, y_pred, model_name=f"KNN-{name}"))

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df


def check_lstm_overfit(csv_path: str, city_name: str):
    print(f"\n{'='*60}\nLSTM - {city_name}\n{'='*60}")

    df = prepare_city_data(csv_path, city_name=city_name)
    train_df, val_df, test_df = time_series_split(df)

    X_train, y_train = build_sequences(train_df)
    X_val, y_val = build_sequences(val_df)
    X_test, y_test = build_sequences(test_df)

    X_train, X_val, X_test, scaler = scale_sequence_data(X_train, X_val, X_test)

    model = train_lstm(X_train, y_train, X_val, y_val)

    results = []
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X, verbose=0).flatten()
        results.append(evaluate_predictions(y, y_pred, model_name=f"LSTM-{name}"))

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    knn_results = check_knn_overfit("data/raw/beograd.csv", "Beograd")
    lstm_results = check_lstm_overfit("data/raw/beograd.csv", "Beograd")

    print(f"\n{'='*60}\nZBIRNI PREGLED\n{'='*60}")
    print(pd.concat([knn_results, lstm_results], ignore_index=True).to_string(index=False))