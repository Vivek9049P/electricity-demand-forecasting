# tests/test_benchmarks.py
import numpy as np
import pandas as pd

from electricity_demand.models.benchmarks import (
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    drift_forecast,
)


def _sample_train(n=208):
    idx = pd.date_range("2015-01-04", periods=n, freq="W")
    values = 50 + 5 * np.sin(2 * np.pi * np.arange(n) / 52)
    return pd.Series(values, index=idx, name="load_gw")


def test_forecast_lengths_match_horizon():
    train = _sample_train()
    horizon = 10
    index = pd.date_range(train.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")

    for forecast in [
        mean_forecast(train, horizon, index=index),
        naive_forecast(train, horizon, index=index),
        seasonal_naive_forecast(train, horizon, seasonality=52, index=index),
        drift_forecast(train, horizon, index=index),
    ]:
        assert len(forecast) == horizon
        assert list(forecast.index) == list(index)


def test_naive_forecast_repeats_last_value():
    train = _sample_train()
    horizon = 5
    index = pd.date_range(train.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")
    forecast = naive_forecast(train, horizon, index=index)
    assert (forecast == train.iloc[-1]).all()


def test_seasonal_naive_matches_last_cycle():
    train = _sample_train()
    horizon = 52
    index = pd.date_range(train.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")
    forecast = seasonal_naive_forecast(train, horizon, seasonality=52, index=index)
    np.testing.assert_allclose(forecast.to_numpy(), train.iloc[-52:].to_numpy())
