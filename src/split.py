import pandas as pd

TRAIN_END = "2023-06-01"
VAL_END = "2025-06-01"


def time_series_split(df: pd.DataFrame, date_col: str = "date"):
    df = df.sort_values(date_col).reset_index(drop=True)

    train_df = df[df[date_col] < TRAIN_END].reset_index(drop=True)
    val_df = df[(df[date_col] >= TRAIN_END) & (df[date_col] < VAL_END)].reset_index(drop=True)
    test_df = df[df[date_col] >= VAL_END].reset_index(drop=True)

    return train_df, val_df, test_df


def get_X_y(df: pd.DataFrame, feature_cols: list[str], target_col: str = "target_pm2_5_t+24h"):
    X = df[feature_cols]
    y = df[target_col]
    return X, y


if __name__ == "__main__":
    dummy = pd.DataFrame({
        "date": pd.date_range("2016-06-01", "2026-06-01", freq="h"),
    })
    dummy["target_pm2_5_t+24h"] = 0

    train, val, test = time_series_split(dummy)
    total = len(train) + len(val) + len(test)
    print(f"Train: {len(train)} ({len(train)/total:.1%})")
    print(f"Val:   {len(val)} ({len(val)/total:.1%})")
    print(f"Test:  {len(test)} ({len(test)/total:.1%})")
