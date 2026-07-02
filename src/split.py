"""
Zajednicki modul za Time-Series Split.

VAZNO: I ovo mora biti identicno za oba clana tima - ako se granice
split-a razlikuju, rezultati modela nisu uporedivi u Fazi 3.

Podela po specifikaciji:
- Trening:    jun 2016 - jun 2023  (70%)
- Validacija: jun 2023 - jun 2025  (20%)
- Test:       jun 2025 - jun 2026  (10%)

Bez nasumicnog mesanja - striktno hronoloski, da se ne desi
"curenje" podataka iz buducnosti u proslost.
"""

import pandas as pd

TRAIN_END = "2023-06-01"
VAL_END = "2025-06-01"
# sve posle VAL_END ide u test skup


def time_series_split(df: pd.DataFrame, date_col: str = "date"):
    """
    Deli DataFrame na train/val/test hronoloski, prema fiksnim datumskim
    granicama definisanim u specifikaciji.

    Vraca: (train_df, val_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    train_df = df[df[date_col] < TRAIN_END].reset_index(drop=True)
    val_df = df[(df[date_col] >= TRAIN_END) & (df[date_col] < VAL_END)].reset_index(drop=True)
    test_df = df[df[date_col] >= VAL_END].reset_index(drop=True)

    return train_df, val_df, test_df


def get_X_y(df: pd.DataFrame, feature_cols: list[str], target_col: str = "target_pm2_5_t+24h"):
    """Pomocna funkcija - razdvaja feature-e (X) i target (y) iz DataFrame-a."""
    X = df[feature_cols]
    y = df[target_col]
    return X, y


if __name__ == "__main__":
    # Sanity check sa fiktivnim podacima
    dummy = pd.DataFrame({
        "date": pd.date_range("2016-06-01", "2026-06-01", freq="h"),
    })
    dummy["target_pm2_5_t+24h"] = 0

    train, val, test = time_series_split(dummy)
    total = len(train) + len(val) + len(test)
    print(f"Train: {len(train)} ({len(train)/total:.1%})")
    print(f"Val:   {len(val)} ({len(val)/total:.1%})")
    print(f"Test:  {len(test)} ({len(test)/total:.1%})")
