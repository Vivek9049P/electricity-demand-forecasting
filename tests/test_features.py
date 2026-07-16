# tests/test_features.py
import numpy as np
import pandas as pd

from electricity_demand.features import make_ml_table


def _sample_data(n=160):
    idx = pd.date_range("2015-01-04", periods=n, freq="W")
    load = 50 + 5 * np.sin(2 * np.pi * np.arange(n) / 52) + np.arange(n) * 0.01
    temp = 10 + 8 * np.sin(2 * np.pi * (np.arange(n) - 26) / 52)
    return pd.DataFrame({"load_gw": load, "temp_mean": temp}, index=idx)


def test_output_has_no_missing_target_values():
    data = _sample_data()
    table = make_ml_table(data)
    assert table["load_gw"].isna().sum() == 0


def test_lag_1_feature_does_not_use_future_values():
    data = _sample_data()
    table = make_ml_table(data)
    # For every remaining row, lag_1 must equal the *previous* week's actual load,
    # never the current or a future week's value.
    aligned_actual = data["load_gw"].reindex(table.index)
    aligned_prev = data["load_gw"].shift(1).reindex(table.index)
    assert np.allclose(table["lag_1"].to_numpy(), aligned_prev.to_numpy())
    assert not np.allclose(table["lag_1"].to_numpy(), aligned_actual.to_numpy())


def test_exogenous_column_is_passed_through():
    data = _sample_data()
    table = make_ml_table(data)
    assert "temp_mean" in table.columns
