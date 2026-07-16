# src/electricity_demand/evaluation.py
"""
Forecast evaluation metrics: MAE, RMSE, MASE, and bias.

MASE is scaled against a seasonal-naive (52-week) error computed on the
*training* set only, so evaluation never depends on quantities that
would not be known at the time the forecast was made.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from electricity_demand.config import SEASONAL_PERIOD


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mase(y_true, y_pred, y_train, seasonality=SEASONAL_PERIOD):
    """Mean Absolute Scaled Error, scaled by the in-sample seasonal-naive error."""
    y_train = pd.Series(y_train)
    naive_errors = np.abs(
        y_train.iloc[seasonality:].to_numpy() - y_train.iloc[:-seasonality].to_numpy()
    )
    scale = naive_errors.mean()
    if scale == 0 or np.isnan(scale):
        raise ValueError("MASE scale is zero or undefined; check y_train length/seasonality.")
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def evaluate_forecast(name, y_true, y_pred, y_train, seasonality=SEASONAL_PERIOD):
    """
    Compute MAE, RMSE, MASE, and bias for a single forecast.

    Returns a dict suitable for collecting into a pd.DataFrame, one row
    per model.
    """
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(np.asarray(y_pred), index=y_true.index).astype(float)

    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=seasonality),
        "Bias": float(np.mean(y_pred - y_true)),
    }
