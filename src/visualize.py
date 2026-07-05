from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.aqi_utils import pm25_to_aqi_category

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 11

AQI_ORDER = ["Dobar", "Prihvatljiv", "Umeren", "Zagadjen", "Veoma zagadjen", "Izuzetno zagadjen"]


def _ensure_dir(results_dir: str | Path) -> Path:
    out = Path(results_dir) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    return out


def print_metrics_table(results: list[dict]) -> pd.DataFrame:
    results_df = pd.DataFrame(results).set_index("model").sort_values("RMSE")

    print("=== Rezultati na test skupu ===")
    print(results_df.to_string())
    print()
    print("RMSE - greška u μg/m³, veće greške se kaznjavaju jace (kvadratna kazna)")
    print("MAE  - prosecna apsolutna greska u μg/m³")
    print("R²   - udeo varijanse PM2.5 koji model objasnjava (1.0 = savrseno)")

    return results_df


def plot_metrics_comparison(results_df: pd.DataFrame, city: str, results_dir: str | Path = "results"):
    out_dir = _ensure_dir(results_dir)
    palette = ["steelblue", "darkorange", "seagreen", "crimson", "purple"][: len(results_df)]
    metrics = ["RMSE", "MAE", "R2"]
    labels = {"RMSE": "RMSE (μg/m³) ↓", "MAE": "MAE (μg/m³) ↓", "R2": "R² ↑"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Poređenje modela — {city}", fontsize=13)

    plot_df = results_df.reset_index()

    for ax, metric in zip(axes, metrics):
        sorted_df = plot_df.sort_values(metric, ascending=(metric != "R2"))
        bars = ax.bar(sorted_df["model"], sorted_df[metric], color=palette, edgecolor="white", width=0.6)
        ax.set_title(labels[metric])
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df["model"], rotation=15, ha="right")
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / f"comparison_metrics_{city}.png", dpi=120)
    plt.show()


def plot_timeseries(
    dates_and_series: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]],
    city: str,
    filename: str,
    results_dir: str | Path = "results",
    n_hours: int = 500,
):
    out_dir = _ensure_dir(results_dir)
    n_models = len(dates_and_series)

    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), sharex=False)
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Stvarne vs. predviđene vrednosti PM2.5 — {city} (prvih {n_hours} sati test skupa)", fontsize=13)

    for ax, (dates, y_true, y_pred, label, color) in zip(axes, dates_and_series):
        ax.plot(dates[:n_hours], np.array(y_true)[:n_hours], label="Stvarno", color="gray", linewidth=1.2, alpha=0.8)
        ax.plot(dates[:n_hours], np.array(y_pred)[:n_hours], label=label, color=color, linewidth=1.0, alpha=0.9)
        ax.set_ylabel("PM2.5 (μg/m³)")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(8))

    axes[-1].set_xlabel("Datum")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120)
    plt.show()
    print("Grafik sacuvan.")


def plot_timeseries_all_models(
    dates_main, y_test, dates_lstm, y_test_lstm,
    predictions: dict, city: str, results_dir: str | Path = "results", n_hours: int = 300,
):
    out_dir = _ensure_dir(results_dir)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(dates_main[:n_hours], np.array(y_test)[:n_hours], label="Stvarno", color="black", linewidth=1.5, alpha=0.7)

    for label, (y_pred, color) in predictions.items():
        dates = dates_lstm if "LSTM" in label else dates_main
        ax.plot(dates[:n_hours], np.array(y_pred)[:n_hours], label=label, color=color, linewidth=1.0, alpha=0.85)

    ax.set_title(f"Predikcija PM2.5 — svi modeli — {city} (prvih {n_hours} sati test skupa)")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_xlabel("Datum")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_dir / f"timeseries_all_models_{city}.png", dpi=120)
    plt.show()


def plot_scatter_comparison(
    scatter_data: list[tuple[np.ndarray, np.ndarray, str, str, dict]],
    city: str,
    filename: str,
    results_dir: str | Path = "results",
):
    out_dir = _ensure_dir(results_dir)
    n_models = len(scatter_data)

    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Scatter: Stvarno vs. predviđeno — {city}", fontsize=13)

    for ax, (y_true, y_pred, label, color, res) in zip(axes, scatter_data):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        ax.scatter(y_true, y_pred, alpha=0.12, s=6, color=color)
        lim = [0, max(float(y_true.max()), float(y_pred.max())) * 1.05]
        ax.plot(lim, lim, "r--", linewidth=1.2, label="Savršena predikcija")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Stvarno PM2.5 (μg/m³)")
        ax.set_ylabel("Predviđeno PM2.5 (μg/m³)")
        ax.set_title(f"{label}   RMSE={res['RMSE']}  R²={res['R2']}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120)
    plt.show()


def plot_feature_importance(
    models_and_features: list[tuple[object, list[str], str, str]],
    city: str,
    results_dir: str | Path = "results",
    top_n: int = 20,
):
    out_dir = _ensure_dir(results_dir)
    n_models = len(models_and_features)

    fig, axes = plt.subplots(1, n_models, figsize=(7.5 * n_models, 7))
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Top {top_n} najvažnijih feature-a — {city}", fontsize=13)

    for ax, (model, feature_cols, label, color) in zip(axes, models_and_features):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)

        sns.barplot(data=importance_df, y="feature", x="importance", ax=ax, color=color, orient="h")
        ax.set_title(label)
        ax.set_xlabel("Važnost")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(out_dir / f"feature_importance_{city}.png", dpi=120)
    plt.show()
    print("Feature importance grafik sacuvan.")


def plot_aqi_distribution(
    y_true, predictions: dict, city: str, filename: str, results_dir: str | Path = "results",
):
    out_dir = _ensure_dir(results_dir)
    palette = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]

    aqi_true = pd.Series(np.array(y_true)).apply(pm25_to_aqi_category)
    data = {"Stvarno": aqi_true.value_counts()}
    for label, y_pred in predictions.items():
        data[label] = pd.Series(np.array(y_pred)).apply(pm25_to_aqi_category).value_counts()

    present = [c for c in AQI_ORDER if any(c in d.index for d in data.values())]
    aqi_df = pd.DataFrame(data).reindex(present).fillna(0).astype(int)

    colors = ["#7f8c8d"] + palette[: len(predictions)]

    fig, ax = plt.subplots(figsize=(10 + 2 * len(predictions), 5))
    aqi_df.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Raspodela AQI kategorija — {city}")
    ax.set_xlabel("AQI kategorija")
    ax.set_ylabel("Broj sati")
    ax.legend(title="Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120)
    plt.show()


def plot_aqi_distribution_per_model(
    pairs: list[tuple[np.ndarray, np.ndarray, str, str]],
    city: str,
    filename: str,
    results_dir: str | Path = "results",
):
    out_dir = _ensure_dir(results_dir)
    n_models = len(pairs)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Raspodela AQI kategorija - {city}", fontsize=13)

    for ax, (y_true, y_pred, label, color) in zip(axes, pairs):
        aqi_true = pd.Series(np.array(y_true)).apply(pm25_to_aqi_category)
        aqi_pred = pd.Series(np.array(y_pred)).apply(pm25_to_aqi_category)

        present = [c for c in AQI_ORDER if c in aqi_true.values or c in aqi_pred.values]
        df_plot = pd.DataFrame({
            "Stvarno": aqi_true.value_counts(),
            label: aqi_pred.value_counts(),
        }).reindex(present).fillna(0).astype(int)

        df_plot.plot(kind="bar", ax=ax, color=["#7f8c8d", color], edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("AQI kategorija")
        ax.set_ylabel("Broj sati")
        ax.legend(title="Model")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=120)
    plt.show()


def plot_extreme_peaks_analysis(
    dates, y_true, predictions: dict, city: str,
    results_dir: str | Path = "results", threshold: float = 90.0,
):
    from src.evaluate import evaluate_predictions

    out_dir = _ensure_dir(results_dir)
    y_true = np.array(y_true)
    peak_mask = y_true > threshold
    n_peaks = peak_mask.sum()

    print(f"Broj sati sa PM2.5 > {threshold} μg/m³ (\"Veoma zagađen\" i gore) u test skupu: {n_peaks}")
    print(f"To je {n_peaks / len(y_true):.2%} test skupa.\n")

    if n_peaks == 0:
        print("Nema ekstremnih pikova u ovom test skupu za izabrani prag/grad.")
        return None

    peak_results = []
    for label, (y_pred, _color) in predictions.items():
        y_pred = np.array(y_pred)
        if len(y_pred) != len(y_true):
            print(f"[Preskačem {label}] duzina niza se ne poklapa sa y_true (verovatno LSTM offset).")
            continue
        peak_results.append(evaluate_predictions(y_true[peak_mask], y_pred[peak_mask], model_name=label))

    peak_df = pd.DataFrame(peak_results).set_index("model").sort_values("RMSE")
    print(f"=== Metrike SAMO na ekstremnim pikovima (PM2.5 > {threshold}) ===")
    print(peak_df.to_string())

    fig, ax = plt.subplots(figsize=(14, 5))
    dates = np.array(dates)
    ax.scatter(dates[peak_mask], y_true[peak_mask], label="Stvarno", color="black", s=18, zorder=5)

    for label, (y_pred, color) in predictions.items():
        y_pred = np.array(y_pred)
        if len(y_pred) != len(y_true):
            continue
        ax.scatter(dates[peak_mask], y_pred[peak_mask], label=label, color=color, s=12, alpha=0.7)

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.5, label=f"Prag ({threshold} μg/m³)")
    ax.set_title(f"Ponasanje modela tokom ekstremnih pikova zagađenja — {city}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.legend(loc="upper right", fontsize=9)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_dir / f"extreme_peaks_{city}.png", dpi=120)
    plt.show()

    return peak_df