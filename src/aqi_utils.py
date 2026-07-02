"""
Izvodjenje AQI kategorije opasnosti iz predvidjene PM2.5 vrednosti,
prema SEPA metodologiji (bez dodatnog treniranja modela - cisto
programsko mapiranje numericke vrednosti u kategoriju).

NAPOMENA: Granicne vrednosti ispod su postavljene prema uobicajenim
SEPA/EU pragovima za PM2.5 (24h prosek, ug/m3). Pre finalne predaje
projekta PROVERITI tacne brojeve na www.sepa.gov.rs jer se metodologija
moze razlikovati u detaljima - ovo je polazna, razumna aproksimacija
da mozete odmah da radite dalje.
"""

import pandas as pd

# (gornja granica, naziv kategorije) - PM2.5 u ug/m3, rastuce
AQI_BINS = [
    (25, "Odlican"),
    (50, "Dobar"),
    (75, "Prihvatljiv"),
    (100, "Zagadjen"),
    (float("inf"), "Jako zagadjen"),
]


def pm25_to_aqi_category(pm25_value: float) -> str:
    """Mapira jednu PM2.5 vrednost u AQI kategoriju."""
    for upper_bound, category in AQI_BINS:
        if pm25_value <= upper_bound:
            return category
    return AQI_BINS[-1][1]  # fallback, ne bi trebalo da se desi


def add_aqi_column(df: pd.DataFrame, pm25_col: str) -> pd.DataFrame:
    """Dodaje kolonu 'aqi_category' na osnovu PM2.5 kolone."""
    df = df.copy()
    df["aqi_category"] = df[pm25_col].apply(pm25_to_aqi_category)
    return df


if __name__ == "__main__":
    for test_val in [10, 40, 60, 90, 150]:
        print(f"PM2.5 = {test_val} -> {pm25_to_aqi_category(test_val)}")
