# src/electricity_demand/pipeline.py

import pandas as pd

from electricity_demand.config import (
    PROCESSED_DATA_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    FIGURE_DIR,
    TEST_WEEKS,
)

from electricity_demand.data import load_processed_data
from electricity_demand.evaluation import evaluate_forecast
from electricity_demand.plotting import plot_forecasts

from electricity_demand.models.benchmarks import (
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    drift_forecast,
)

from electricity_demand.models.sarimax import (
    fit_sarimax,
    forecast_sarimax,
)

from electricity_demand.models.feature_models import (
    make_ml_table,
    fit_gradient_boosting,
    predict_feature_model,
)


def run_pipeline(
    test_weeks=TEST_WEEKS,
    include_sarimax=True,
    include_feature_model=True,
    include_bayesian=False,
    include_neural=False,
):
    """
    Run the full electricity-demand forecasting workflow.

    Steps:
    1. Load processed weekly data.
    2. Split into train and test sets.
    3. Fit benchmark models.
    4. Fit optional SARIMAX, feature-based, Bayesian, and neural models.
    5. Evaluate forecasts.
    6. Save forecasts, metrics, and figures.
    """

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    data = load_processed_data()

    y = data["load_gw"]

    train = y.iloc[:-test_weeks]
    test = y.iloc[-test_weeks:]

    horizon = len(test)

    forecasts = {}
    metrics = []

    # --------------------------------------------------
    # 2. Benchmarks
    # --------------------------------------------------

    forecasts["mean"] = mean_forecast(train, horizon, index=test.index)
    forecasts["naive"] = naive_forecast(train, horizon, index=test.index)
    forecasts["seasonal_naive"] = seasonal_naive_forecast(
        train,
        horizon,
        seasonality=52,
        index=test.index,
    )
    forecasts["drift"] = drift_forecast(train, horizon, index=test.index)

    # --------------------------------------------------
    # 3. SARIMAX
    # --------------------------------------------------

    if include_sarimax:
        exog_cols = [
            col for col in [
                "temp_mean",
                "heating_degree_days",
                "cooling_degree_days",
                "holiday_days",
                "has_holiday",
            ]
            if col in data.columns
        ]

        if exog_cols:
            X = data[exog_cols]
            X_train = X.iloc[:-test_weeks]
            X_test = X.iloc[-test_weeks:]
        else:
            X_train = None
            X_test = None

        sarimax_fit = fit_sarimax(
            y_train=train,
            X_train=X_train,
        )

        forecasts["sarimax"] = forecast_sarimax(
            model_fit=sarimax_fit,
            horizon=horizon,
            X_test=X_test,
            index=test.index,
        )

    # --------------------------------------------------
    # 4. Feature-based model
    # --------------------------------------------------

    if include_feature_model:
        ml_data = make_ml_table(data)
        ml_train = ml_data.iloc[:-test_weeks]
        ml_test = ml_data.iloc[-test_weeks:]

        target = "load_gw"
        feature_cols = [col for col in ml_data.columns if col != target]

        X_train = ml_train[feature_cols]
        y_train = ml_train[target]

        X_test = ml_test[feature_cols]
        y_test = ml_test[target]

        model = fit_gradient_boosting(X_train, y_train)

        forecasts["feature_model"] = predict_feature_model(
            model,
            X_test,
            index=y_test.index,
        )

    # --------------------------------------------------
    # 5. Bayesian and neural models
    # --------------------------------------------------
    # These can be added here, but I would make them optional
    # because they may take longer and require extra dependencies.

    if include_bayesian:
        raise NotImplementedError(
            "Add Bayesian model workflow here."
        )

    if include_neural:
        raise NotImplementedError(
            "Add neural model workflow here."
        )

    # --------------------------------------------------
    # 6. Evaluate all forecasts
    # --------------------------------------------------

    for name, pred in forecasts.items():
        pred_aligned = pred.reindex(test.index)

        metrics.append(
            evaluate_forecast(
                name=name,
                y_true=test,
                y_pred=pred_aligned,
                y_train=train,
            )
        )

    metrics_df = pd.DataFrame(metrics).sort_values("MASE")

    # --------------------------------------------------
    # 7. Save outputs
    # --------------------------------------------------

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    forecast_df = pd.DataFrame({"actual": test})

    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)

    forecast_df.to_csv(FORECAST_DIR / "all_forecasts.csv")
    metrics_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    fig = plot_forecasts(
        train=train,
        test=test,
        forecasts=forecasts,
    )

    fig.savefig(FIGURE_DIR / "forecast_comparison.png", dpi=300, bbox_inches="tight")

    return metrics_df, forecast_df