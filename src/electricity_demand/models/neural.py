# src/electricity_demand/models/neural.py
"""
Placeholder for a neural forecasting model (e.g. LSTM, N-BEATS, N-HiTS, or
a Temporal Fusion Transformer).

An hourly LSTM was explored in notebooks/A1_full_pipeline.ipynb using a
recursive multi-step rollout; it performed far worse than every classical
or feature-based model on the 2-year weekly-resampled comparison (see
reports/, Section 7), because one-step errors compound over the ~17,500
recursive hourly steps needed to cover a 2-year horizon. `pipeline.run_pipeline`
only calls into this module when `include_neural=True`; left False by
default given that result and the extra dependency (TensorFlow/PyTorch)
it would add to a base install.

Future work (see README, "Good practice") should prefer a direct
multi-step or sequence-to-sequence architecture over single-step
recursive rollout, and evaluate over a shorter, more realistic horizon.
"""


def fit_neural_model(y_train, X_train=None):
    raise NotImplementedError(
        "Neural model not implemented in the base pipeline. See module "
        "docstring and notebooks/A1_full_pipeline.ipynb for the explored LSTM."
    )


def forecast_neural_model(model_fit, horizon, X_test=None, index=None):
    raise NotImplementedError("Neural model not implemented in the base pipeline.")
