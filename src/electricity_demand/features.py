# src/electricity_demand/features.py
"""
Feature engineering for the weekly electricity-demand series.

All lag and rolling-window features use ``.shift()`` before any window
operation, so that the row for week *t* only ever uses information
available up to week *t - 1*. This is what avoids look-ahead / data
leakage when the resulting table is later split into train and test
periods (see README.md, "Data leakage").
"""

import numpy as np
import pandas as pd

from electricity_demand.config import (
    HEATING_BASE_TEMP,
    COOLING_BASE_TEMP,
)

LAGS = (1, 2, 4, 8, 13, 26, 52)
ROLLING_WINDOWS = (4, 13, 52)
FOURIER_HARMONICS = (1, 2, 3)


def add_degree_days(daily_temp, heating_base=HEATING_BASE_TEMP, cooling_base=COOLING_BASE_TEMP):
    """Compute daily heating/cooling degree-day values from mean temperature."""
    out = daily_temp.copy()
    out["heating_degree"] = np.maximum(heating_base - out["temp_mean"], 0)
    out["cooling_degree"] = np.maximum(out["temp_mean"] - cooling_base, 0)
    return out


def resample_temperature_to_weekly(daily_temp_features, target_index):
    """Aggregate daily temperature/degree-day features to weekly and align to target_index."""
    weekly = daily_temp_features.resample("W-SUN").agg(
        {"temp_mean": "mean", "heating_degree": "sum", "cooling_degree": "sum"}
    )
    weekly = weekly.rename(
        columns={"heating_degree": "heating_degree_days", "cooling_degree": "cooling_degree_days"}
    )
    weekly = weekly.reindex(target_index).interpolate("time")
    return weekly


def make_ml_table(data, target_col="load_gw", lags=LAGS, rolling_windows=ROLLING_WINDOWS):
    """
    Build the supervised-learning feature table used by the feature-based model.

    Parameters
    ----------
    data : pd.DataFrame
        Weekly data indexed by date, containing at least ``target_col`` and
        optionally exogenous covariates (temperature, holiday, etc.) which
        are passed through unchanged.
    target_col : str
        Name of the target column (default ``load_gw``).
    lags : tuple of int
        Lag lengths (in weeks) to include.
    rolling_windows : tuple of int
        Rolling-mean window lengths (in weeks), computed on the
        already-lagged (shift(1)) series.

    Returns
    -------
    pd.DataFrame
        Feature table with the target column renamed to ``load_gw`` and
        all NA rows (from the initial lag/rolling burn-in) dropped.
    """
    df = pd.DataFrame({target_col: data[target_col]})

    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    shifted = df[target_col].shift(1)
    for window in rolling_windows:
        df[f"roll_mean_{window}"] = shifted.rolling(window).mean()

    week = df.index.isocalendar().week.astype(int)
    df["week"] = week
    df["year"] = df.index.year
    for k in FOURIER_HARMONICS:
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * week / 52)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * week / 52)

    exog_cols = [
        col for col in data.columns
        if col != target_col and col not in df.columns
    ]
    if exog_cols:
        df = df.join(data[exog_cols])

    return df.dropna()
