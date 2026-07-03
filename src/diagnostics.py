import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_diagnostics(y_test, y_pred, model_name: str, output_dir: str | Path = "results/diagnostics"):
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    residuals = y_test - y_pred

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Dijagnostika grešaka — {model_name}", fontsize=14)

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.15, s=8)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Savršena predikcija")
    ax.set_xlabel("Stvarna PM2.5 (μg/m³)")
    ax.set_ylabel("Predviđena PM2.5 (μg/m³)")
    ax.set_title("Stvarno vs. predviđeno")
    ax.legend()

    ax = axes[1]
    n_show = min(500, len(y_test))
    ax.plot(y_test[:n_show], label="Stvarno", linewidth=1)
    ax.plot(y_pred[:n_show], label="Predviđeno", linewidth=1, alpha=0.8)
    ax.set_xlabel("Vremenski korak (test skup)")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_title(f"Vremenska serija (prvih {n_show} sati test skupa)")
    ax.legend()

    ax = axes[2]
    ax.scatter(y_test, residuals, alpha=0.15, s=8)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Stvarna PM2.5 (μg/m³)")
    ax.set_ylabel("Greška (stvarno - predviđeno)")
    ax.set_title("Greška po nivou zagađenja")

    plt.tight_layout()
    save_path = output_dir / f"diagnostics_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(save_path, dpi=120)
    print(f"Sačuvano: {save_path}")
    plt.close()

    correlation = np.corrcoef(y_test, np.abs(residuals))[0, 1]
    print(f"Korelacija |greška| sa stvarnom PM2.5 vrednošću: {correlation:.3f}")
    print("(vrednost blizu 1 = model dosta vise gresi kod visokih koncentracija)")


if __name__ == "__main__":
    from src.models.knn_model import run_knn_pipeline
    from src.models.lstm_model import run_lstm_pipeline

    knn_output = run_knn_pipeline("data/raw/beograd.csv", city_name="Beograd")
    plot_diagnostics(knn_output["y_test"], knn_output["y_pred"], "KNN (Beograd)")

    lstm_output = run_lstm_pipeline("data/raw/beograd.csv", city_name="Beograd")
    plot_diagnostics(lstm_output["y_test"], lstm_output["y_pred"], "LSTM (Beograd)")