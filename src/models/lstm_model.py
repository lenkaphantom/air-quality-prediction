import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data_prep import prepare_city_data, POLLUTANT_COLS
from src.split import time_series_split
from src.evaluate import evaluate_predictions

SEQ_LENGTH = 48
ROLLING_FEATURES = [
    "pm2_5_rolling_mean_12h", "pm2_5_rolling_std_12h", "pm2_5_rolling_max_12h",
    "pm2_5_rolling_mean_24h", "pm2_5_rolling_std_24h", "pm2_5_rolling_max_24h",
]
SEQUENCE_FEATURES = (
    POLLUTANT_COLS
    + ["hour", "month", "day_of_week", "heating_season"]
    + ROLLING_FEATURES
)
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_sequences(df: pd.DataFrame, seq_length: int = SEQ_LENGTH):
    values = df[SEQUENCE_FEATURES].values
    target = df["target_pm2_5_t+24h"].values

    X, y = [], []
    for i in range(seq_length, len(df)):
        X.append(values[i - seq_length:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def scale_sequence_data(X_train, X_val, X_test):
    n_features = X_train.shape[2]
    scaler = MinMaxScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    def _scale(X):
        shape = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = scaler.transform(X_2d)
        return X_scaled.reshape(shape)

    return _scale(X_train), _scale(X_val), _scale(X_test), scaler


def build_lstm_model(seq_length: int, n_features: int, learning_rate: float = 0.0005) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def train_lstm(
    X_train, y_train, X_val, y_val,
    epochs: int = 50, batch_size: int = 64,
) -> Sequential:
    set_seeds()

    model = build_lstm_model(seq_length=X_train.shape[1], n_features=X_train.shape[2])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
        shuffle=False,
    )
    return model


def run_lstm_pipeline(csv_path: str | Path, city_name: str, epochs: int = 100) -> dict:
    set_seeds()

    df = prepare_city_data(csv_path, city_name=city_name)
    train_df, val_df, test_df = time_series_split(df)

    X_train, y_train = build_sequences(train_df)
    X_val, y_val = build_sequences(val_df)
    X_test, y_test = build_sequences(test_df)

    X_train, X_val, X_test, scaler = scale_sequence_data(X_train, X_val, X_test)

    model = train_lstm(X_train, y_train, X_val, y_val, epochs=epochs)

    y_pred = model.predict(X_test).flatten()
    result = evaluate_predictions(y_test, y_pred, model_name=f"LSTM ({city_name})")

    return {
        "result": result,
        "model": model,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def save_lstm_model(model, scaler, city_name: str, output_dir: str | Path = "results"):
    import joblib
    output_dir = Path(output_dir) / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / f"lstm_model_{city_name}.keras")
    joblib.dump(scaler, output_dir / f"lstm_scaler_{city_name}.pkl")


if __name__ == "__main__":
    output = run_lstm_pipeline("data/raw/beograd.csv", city_name="Beograd")
    print(output["result"])
    save_lstm_model(output["model"], output["scaler"], "Beograd")