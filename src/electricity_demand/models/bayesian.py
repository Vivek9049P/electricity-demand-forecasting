# src/electricity_demand/models/bayesian.py
"""
Placeholder for a Bayesian forecasting model (e.g. a seasonal regression
with priors over trend, seasonal, holiday, and temperature coefficients,
fitted with PyMC or a similar probabilistic-programming library).

Not implemented in this submission. `pipeline.run_pipeline` only calls
into this module when `include_bayesian=True`; left False by default so
that the base pipeline has no extra dependency on a PPL backend.

A real implementation should return:
  - a point forecast (posterior predictive mean), and
  - posterior predictive intervals (e.g. 5th/95th percentiles),
so it can be compared to the SARIMAX confidence interval on equal terms.
"""


def fit_bayesian_model(y_train, X_train=None):
    raise NotImplementedError(
        "Bayesian model not implemented. See module docstring for the "
        "expected interface if you add one (e.g. with PyMC)."
    )


def forecast_bayesian_model(model_fit, horizon, X_test=None, index=None):
    raise NotImplementedError("Bayesian model not implemented.")
