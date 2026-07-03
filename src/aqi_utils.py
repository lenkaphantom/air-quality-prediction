import pandas as pd


# Granicne vrednosti po zvanicnoj SEPA metodologiji za PM2.5 (ug/m3):
# Izvor: https://vazduh.sepa.gov.rs/
AQI_BINS = [
    (5,            "Dobar"),
    (15,           "Prihvatljiv"),
    (50,           "Umeren"),
    (90,           "Zagadjen"),
    (140,          "Veoma zagadjen"),
    (float("inf"), "Izuzetno zagadjen"),
]


def pm25_to_aqi_category(pm25_value: float) -> str:
    for upper_bound, category in AQI_BINS:
        if pm25_value <= upper_bound:
            return category
    return AQI_BINS[-1][1]


def add_aqi_column(df: pd.DataFrame, pm25_col: str) -> pd.DataFrame:
    df = df.copy()
    df["aqi_category"] = df[pm25_col].apply(pm25_to_aqi_category)
    return df


if __name__ == "__main__":
    for test_val in [10, 40, 60, 90, 150]:
        print(f"PM2.5 = {test_val} -> {pm25_to_aqi_category(test_val)}")
