from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from src.data_prep import prepare_city_data, get_feature_columns
from src.split import time_series_split, get_X_y


def check_feature_importance(csv_path: str, city_name: str, top_n: int = 30):
    df = prepare_city_data(csv_path, city_name=city_name)
    train_df, val_df, test_df = time_series_split(df)

    feature_cols = get_feature_columns(df)
    X_train, y_train = get_X_y(train_df, feature_cols)

    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} najvaznijih feature-a:")
    print(importance_df.head(top_n).to_string(index=False))

    def categorize(name):
        if "rolling" in name:
            return "rolling"
        if "diff" in name:
            return "diff"
        if "lag" in name:
            return "lag"
        if name in ("hour", "month", "day_of_week", "heating_season"):
            return "temporal"
        return "raw_pollutant"

    importance_df["category"] = importance_df["feature"].apply(categorize)
    print("\nUkupna vaznost po kategoriji feature-a:")
    print(importance_df.groupby("category")["importance"].sum().sort_values(ascending=False))

    return importance_df


if __name__ == "__main__":
    check_feature_importance("data/raw/beograd.csv", "Beograd")