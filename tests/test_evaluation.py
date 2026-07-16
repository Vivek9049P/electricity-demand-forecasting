# tests/test_evaluation.py
import numpy as np
import pandas as pd
import pytest

from electricity_demand.evaluation import evaluate_forecast, mase, rmse


def _sample_train(n=208):
    idx = pd.date_range("2015-01-04", periods=n, freq="W")
    values = 50 + 5 * np.sin(2 * np.pi * np.arange(n) / 52)
    return pd.Series(values, index=idx)


def test_mase_zero_for_perfect_forecast():
    train = _sample_train()
    y_true = train.iloc[-52:]
    y_pred = y_true.copy()
    assert mase(y_true, y_pred, train, seasonality=52) == pytest.approx(0.0)


def test_rmse_matches_manual_calculation():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert rmse(y_true, y_pred) == pytest.approx(expected)


def test_evaluate_forecast_returns_expected_keys():
    train = _sample_train()
    y_true = train.iloc[-10:]
    y_pred = y_true + 1.0
    result = evaluate_forecast("dummy_model", y_true, y_pred, train)

    assert set(result) == {"model", "MAE", "RMSE", "MASE", "Bias"}
    assert result["model"] == "dummy_model"
    assert result["Bias"] == pytest.approx(1.0)
    assert result["MAE"] == pytest.approx(1.0)
