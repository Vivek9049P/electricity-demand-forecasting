# src/electricity_demand/models/sarimax.py
"""
SARIMA / SARIMAX model fitting and forecasting.

The default (order, seasonal_order) come from the AIC grid search over
p in [0,6], d in [0,2], q in [0,6] performed in
notebooks/A1_full_pipeline.ipynb (see README, "SARIMAX model", for the
justification of d=1, D=1, and m=52).
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX

from electricity_demand.config import SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER


def fit_sarimax(y_train, X_train=None, order=SARIMAX_ORDER, seasonal_order=SARIMAX_SEASONAL_ORDER):
    """Fit a (SARIMAX if X_train is given, else SARIMA) model on the training series."""
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 else None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarimax(model_fit, horizon, X_test=None, index=None):
    """
    Forecast `horizon` steps ahead and return the point forecast as a pd.Series.

    If `X_test` contains covariates whose test-period values were actually
    observed (e.g. realised temperature, rather than a weather forecast),
    the result is a conditional/explanatory forecast, not a true
    operational forecast -- see README, "Data" and "Data leakage".
    """
    forecast_obj = model_fit.get_forecast(steps=horizon, exog=X_test)
    mean = forecast_obj.predicted_mean

    if index is not None:
        mean.index = index

    return mean


def forecast_sarimax_with_interval(model_fit, horizon, X_test=None, index=None, alpha=0.05):
    """Like `forecast_sarimax`, but also returns the confidence interval (for plotting)."""
    forecast_obj = model_fit.get_forecast(steps=horizon, exog=X_test)
    mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=alpha)

    if index is not None:
        mean.index = index
        conf_int.index = index

    return mean, conf_int
