# scripts/evaluate_models.py
"""
Additional evaluation diagnostics beyond the core metrics saved by
run_pipeline.py: an error-over-time plot for every model, read from
outputs/forecasts/all_forecasts.csv.

Run after run_pipeline.py:

    python scripts/evaluate_models.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from electricity_demand.config import FORECAST_DIR, FIGURE_DIR


def main():
    forecasts_path = FORECAST_DIR / "all_forecasts.csv"
    if not forecasts_path.exists():
        raise FileNotFoundError(
            f"{forecasts_path} not found. Run 'python scripts/run_pipeline.py' first."
        )

    df = pd.read_csv(forecasts_path, index_col=0, parse_dates=True)
    actual = df["actual"]
    model_cols = [c for c in df.columns if c != "actual"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for col in model_cols:
        error = df[col] - actual
        ax.plot(df.index, error, label=col, linewidth=1.1)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Forecast error over time (forecast - actual), by model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error, GW")
    ax.legend()
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "error_diagnostics.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
