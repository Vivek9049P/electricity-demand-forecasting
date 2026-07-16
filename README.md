# Forecasting German Electricity Demand

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting German electricity demand.

The project uses hourly electricity load data from Open Power System Data, aggregates it to weekly average load, and compares a range of forecasting models, from simple statistical benchmarks to SARIMAX, a feature-based gradient-boosting model, and an exploratory LSTM.

## Project aim

The aim of this project is to forecast weekly German electricity demand and compare the performance, interpretability, and complexity of different forecasting approaches.

The main research questions are:

1. How well do simple benchmark methods forecast weekly electricity demand?
2. Does a SARIMAX model improve on seasonal benchmarks?
3. Do temperature and holiday covariates improve forecast accuracy?
4. Do feature-based or neural models justify their additional complexity?
5. Which model would be most appropriate for an operational forecasting setting?

Full answers to these questions, and the assignment's Part 7 questions, are in `reports/A1_report.docx`.

## Data

The target series is German electricity load from Open Power System Data:
<https://data.open-power-system-data.org/time_series/>

The original data are hourly electricity load observations (`DE_load_actual_entsoe_transparency`). These are cleaned, aggregated to weekly average load, and converted from MW to GW.

The main target variable is:

```text
load_gw
```

Optional covariates include:

```text
temp_mean
heating_degree_days
cooling_degree_days
```

Temperature data are Berlin daily mean temperature from the Open-Meteo archive API, used as a representative proxy for German demand-relevant weather.

Temperature features are external covariates and should be treated carefully. In a real forecasting setting, future temperature would not be known exactly and would need to come from a weather forecast. Because the SARIMAX model in this project uses realised (actually observed) future temperature in the test period, its forecast is a **conditional / explanatory forecast**, not a true operational forecast. This is discussed explicitly in `reports/A1_report.docx`.

Holiday features are not currently implemented (see "Future improvements" below); unlike temperature, they would be known in advance and so are a valid, leakage-free covariate for future work.

## Repository structure

```text
electricity-demand-forecasting/
│
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/            # downloaded by scripts/download_data.py (not committed)
│   ├── interim/
│   └── processed/      # built by scripts/make_features.py (not committed)
│
├── src/
│   └── electricity_demand/
│       ├── __init__.py
│       ├── config.py           # paths and modelling constants
│       ├── pipeline.py         # run_pipeline(): the full workflow
│       ├── data.py             # loading / cleaning raw data
│       ├── features.py         # lag, rolling, calendar, degree-day features
│       ├── evaluation.py       # MAE, RMSE, MASE, bias
│       ├── plotting.py         # shared forecast plot
│       └── models/
│           ├── __init__.py
│           ├── benchmarks.py       # mean, naive, seasonal naive, drift
│           ├── sarimax.py          # SARIMA / SARIMAX fit + forecast
│           ├── feature_models.py   # gradient boosting feature model
│           ├── bayesian.py         # not implemented (see docstring)
│           └── neural.py           # not implemented (see docstring)
│
├── scripts/
│   ├── download_data.py    # download raw OPSD load + Open-Meteo temperature
│   ├── make_features.py    # build data/processed/weekly_features.csv
│   ├── run_pipeline.py     # entry point: python scripts/run_pipeline.py
│   └── evaluate_models.py  # extra error-diagnostic plot
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── metrics/
│   └── model_objects/
│
├── reports/
│   ├── A1_report.pdf
│   └── figures/
│
├── notebooks/
│   └── A1_full_pipeline.ipynb   # original exploratory analysis (Parts 1-6)
│
└── tests/
    ├── test_features.py
    ├── test_evaluation.py
    └── test_benchmarks.py
```

## Pipeline overview

The main modelling workflow is controlled by:

```text
scripts/run_pipeline.py
```

This script calls the reusable pipeline function in:

```text
src/electricity_demand/pipeline.py
```

The pipeline performs the following steps:

1. Load the processed weekly data (`data.load_processed_data`).
2. Split into training and test sets (final 104 weeks held out).
3. Fit benchmark forecasting models (mean, naive, seasonal naive, drift).
4. Fit a SARIMAX model, with temperature as an optional exogenous regressor.
5. Fit a gradient-boosting feature-based model on lag/rolling/calendar/temperature features.
6. (Optional, off by default) Bayesian and neural models — see `models/bayesian.py` and `models/neural.py`.
7. Evaluate every forecast with the same metrics on the same test period.
8. Save forecasts, metrics, and a comparison figure to `outputs/`.

The preferred command-line workflow is:

```bash
python scripts/run_pipeline.py
```

## Installation

Create and activate a virtual environment.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the package and its dependencies (editable install, so `import electricity_demand` works from anywhere):

```bash
pip install -r requirements.txt
pip install -e .
```

Alternatively, using conda:

```bash
conda env create -f environment.yml
conda activate electricity-demand-forecasting
```

## Reproducing the analysis

From a fresh clone of the repository:

```bash
python scripts/download_data.py     # downloads raw OPSD + Open-Meteo data to data/raw/
python scripts/make_features.py     # builds data/processed/weekly_features.csv
python scripts/run_pipeline.py      # fits all models, saves forecasts/metrics/figures
python scripts/evaluate_models.py   # optional: extra error-diagnostic plot
```

This creates or updates:

```text
outputs/forecasts/all_forecasts.csv
outputs/metrics/model_comparison.csv
outputs/figures/forecast_comparison.png
outputs/figures/error_diagnostics.png
```

The full exploratory analysis (EDA, stationarity testing, AIC grid search, LSTM experiment) that this package was refactored from is preserved in `notebooks/A1_full_pipeline.ipynb`.

## Models

### Benchmark models (`models/benchmarks.py`)

```text
Mean forecast
Naive forecast
Seasonal naive forecast (52-week)
Drift forecast
```

The seasonal naive model is especially important here because weekly electricity demand has strong annual seasonality (STL seasonal strength ≈ 0.87 on the full series), and turns out to be a difficult baseline for the more complex models to beat.

### SARIMA / SARIMAX model (`models/sarimax.py`)

The default order was chosen via an AIC grid search over `p ∈ [0,6]`, `d ∈ [0,2]`, `q ∈ [0,6]` with the seasonal order fixed at `(1, 1, 1, 52)`:

```python
order = (3, 1, 6)
seasonal_order = (1, 1, 1, 52)
```

`d=1, D=1` were chosen because ADF/KPSS tests agree the series is stationary only after both a regular and a seasonal (52-week) difference are applied; `m=52` follows from the clear annual cycle in the ACF and STL decomposition. See `reports/A1_report.docx`, Section 9.3, for the full justification.

When temperature covariates are supplied via `exog`, this becomes a SARIMAX conditional forecast (see "Data" above).

### Feature-based machine-learning model (`models/feature_models.py`)

A `HistGradientBoostingRegressor` fitted on a supervised-learning table of:

```text
lags:          1, 2, 4, 8, 13, 26, 52 weeks
rolling means: 4, 13, 52 weeks (computed on the lag-1 series)
calendar:      sin/cos Fourier terms of ISO week, harmonics 1-3
exogenous:     temp_mean, heating_degree_days, cooling_degree_days
```

All lag and rolling features use `.shift()` before any window operation, so no row uses information from its own or a later week (see `tests/test_features.py`).

### Bayesian and neural models (not implemented)

`models/bayesian.py` and `models/neural.py` define the expected interface but raise `NotImplementedError`; `pipeline.run_pipeline` only calls them when `include_bayesian=True` / `include_neural=True`. An hourly LSTM was explored in `notebooks/A1_full_pipeline.ipynb` using a recursive multi-step rollout; it performed roughly an order of magnitude worse than every other model on the 2-year horizon once errors compounded over ~17,500 recursive hourly steps (see `reports/A1_report.docx`, Section 7).

## Evaluation

All models are evaluated on the same held-out 104-week (2-year) test period.

Metrics (`src/electricity_demand/evaluation.py`):

```text
MAE
RMSE
MASE   (scaled by the in-sample 52-week seasonal-naive error)
Bias
```

Results (from `notebooks/A1_full_pipeline.ipynb`, 2015-2020 data, 104-week test horizon):

| Model                                | MAE (GW) | RMSE (GW) | MASE   | Bias (GW) |
|---------------------------------------|---------:|----------:|-------:|----------:|
| Feature model (GBR + temperature)     |    2.034 |     2.715 |  1.520 |     1.321 |
| Seasonal naive                        |    2.063 |     2.672 |  1.541 |     1.441 |
| SARIMAX + temperature (conditional)   |    2.862 |     3.641 |  2.139 |     2.624 |
| SARIMA(3,1,6)x(1,1,1,52)              |    3.210 |     3.936 |  2.398 |     3.023 |
| Naive                                 |    3.783 |     4.459 |  2.827 |    -0.882 |
| Mean                                  |    3.789 |     4.397 |  2.831 |     0.481 |
| Drift                                 |    4.340 |     5.118 |  3.243 |     1.007 |
| LSTM (hourly, resampled to weekly)    |   14.512 |    15.155 | 10.843 |    14.512 |

Only the feature-based model narrowly improves on seasonal naive; see `reports/A1_report.docx` for the full discussion of why the added complexity of SARIMAX and the LSTM does not pay off here.

## Train-test split

The default test set is the final 104 weeks of the series (a two-year evaluation period). The split is strictly chronological — **not** a random split — because this is a time-series forecasting problem.

## Data leakage

Two forms of leakage are relevant to this project, and both are addressed explicitly:

```text
1. Lag/rolling features must not use future target values.
   -> Addressed with .shift() before every window operation (features.py);
      verified in tests/test_features.py.

2. Using observed future temperature without labelling the forecast conditional.
   -> Addressed by explicitly documenting the SARIMAX + temperature result
      as a conditional/explanatory forecast, not an operational one
      (see "Data" above and reports/A1_report.docx).
```

All preprocessing that learns from the data (e.g. scalers, if added) should be fitted on the training set only.

## Outputs

`outputs/forecasts/all_forecasts.csv` — one row per test-period week, columns:

```text
actual, mean, naive, seasonal_naive, drift, sarimax, feature_model
```

`outputs/metrics/model_comparison.csv` — one row per model, columns:

```text
model, MAE, RMSE, MASE, Bias
```

`outputs/figures/` — `forecast_comparison.png`, `error_diagnostics.png`.

## Report

The full report (`reports/A1_report.docx`) covers:

```text
1. Introduction
2. Data and exploratory analysis
3. Benchmark forecasts
4. SARIMA model
5. SARIMAX with temperature (conditional forecast)
6. Feature-based regression model
7. LSTM neural network
8. Model comparison
9. Answers to assignment questions (Part 7)
10. Future improvements
References
```

## Testing

```bash
pytest
```

Current tests cover:

```text
forecast lengths match the test horizon (test_benchmarks.py)
seasonal naive repeats the last observed cycle (test_benchmarks.py)
MASE is zero for a perfect forecast (test_evaluation.py)
evaluate_forecast returns the expected metric keys (test_evaluation.py)
the processed feature table has no missing target values (test_features.py)
lag features use only past, not future, values (test_features.py)
```

## Good practice followed in this repository

```text
Reusable code lives in src/electricity_demand/, kept import-safe (no network
  calls on import) and installable via pip install -e .
Notebooks (notebooks/) are for exploration; scripts/ and src/ are for the
  reproducible pipeline.
Raw and processed data are not committed (.gitignore); the pipeline
  regenerates them from scripts/download_data.py and make_features.py.
The pipeline runs end-to-end from a fresh clone via three commands.
Random seeds are fixed (RANDOM_STATE in config.py).
Every model is compared against the seasonal naive benchmark on identical
  metrics and an identical test period.
Covariates are explicitly labelled as known-at-origin (holiday, if added)
  or not (temperature), and the SARIMAX result is labelled conditional.
Unit tests cover forecast shape, metric correctness, and absence of
  feature leakage.
```

## Expected submission contents

```text
README.md
requirements.txt / environment.yml
source code in src/
pipeline entry point in scripts/run_pipeline.py
notebooks/A1_full_pipeline.ipynb (exploration and full Part 1-6 analysis)
reports/A1_report.docx (final report, including Part 7 answers)
```

This repository runs from a fresh clone using the instructions above.
