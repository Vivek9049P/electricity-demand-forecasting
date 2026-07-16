# src/electricity_demand/models/feature_models.py
"""Feature-based (gradient boosting) forecasting model."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from electricity_demand.config import RANDOM_STATE
from electricity_demand.features import make_ml_table as _make_ml_table

# Re-exported so callers only need to import from this module.
make_ml_table = _make_ml_table


def fit_gradient_boosting(X_train, y_train, **kwargs):
    """Fit a HistGradientBoostingRegressor with sensible defaults for this dataset size."""
    params = dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, random_state=RANDOM_STATE)
    params.update(kwargs)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


def predict_feature_model(model, X_test, index=None):
    """Predict with a fitted feature-based model and return a pd.Series."""
    preds = model.predict(X_test)
    return pd.Series(preds, index=index if index is not None else X_test.index)
