# src/electricity_demand/plotting.py
"""Shared plotting helpers used by the pipeline and by exploration notebooks."""

import matplotlib.pyplot as plt


def plot_forecasts(train, test, forecasts, history_weeks=52, title="Forecasts vs actual weekly load"):
    """
    Plot recent training history, actual test values, and each model's forecast.

    Parameters
    ----------
    train, test : pd.Series
        Training and test target series.
    forecasts : dict[str, pd.Series]
        Mapping of model name -> forecast series aligned to `test.index`.
    history_weeks : int
        Number of trailing training weeks to show for context.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(train.index[-history_weeks:], train.iloc[-history_weeks:],
            label="Training data", color="grey", linewidth=1.2)
    ax.plot(test.index, test, label="Actual", color="black", linewidth=2)

    for name, pred in forecasts.items():
        pred_aligned = pred.reindex(test.index)
        ax.plot(test.index, pred_aligned, label=name, linestyle="--", linewidth=1.3)

    ax.axvline(test.index[0], color="grey", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average load, GW")
    ax.legend()
    fig.tight_layout()
    return fig
