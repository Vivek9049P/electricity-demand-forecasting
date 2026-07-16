# src/electricity_demand/models/benchmarks.py
"""
Classical benchmark forecasts: mean, naive, seasonal naive, and drift.

Each function returns a pd.Series of length `horizon` indexed by the
supplied `index`, so it can be dropped straight into the `forecasts`
dict in pipeline.run_pipeline alongside the more complex models.
"""

import numpy as np
import pandas as pd


def mean_forecast(train, horizon, index):
    return pd.Series(train.mean(), index=index)


def naive_forecast(train, horizon, index):
    return pd.Series(train.iloc[-1], index=index)


def seasonal_naive_forecast(train, horizon, seasonality, index):
    """
    Repeat the last full seasonal cycle observed in `train`, tiled forward
    to cover `horizon` steps. Uses only training data, so this is a genuine
    multi-step-ahead forecast with no test-set leakage.
    """
    last_cycle = train.iloc[-seasonality:].to_numpy()
    reps = int(np.ceil(horizon / seasonality))
    values = np.tile(last_cycle, reps)[:horizon]
    return pd.Series(values, index=index)


def drift_forecast(train, horizon, index):
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    values = train.iloc[-1] + slope * np.arange(1, horizon + 1)
    return pd.Series(values, index=index)
