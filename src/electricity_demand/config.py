# src/electricity_demand/config.py
"""
Central configuration for paths and modelling constants.

Keeping these in one place means every script and module agrees on
where data lives and which train/test split and random seed to use,
instead of each file hard-coding its own values.
"""

from pathlib import Path

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
# ROOT resolves to the repository root regardless of the current
# working directory the script is invoked from.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_OBJECT_DIR = OUTPUT_DIR / "model_objects"

RAW_LOAD_FILE = RAW_DATA_DIR / "time_series_60min_singleindex.csv"
RAW_TEMPERATURE_FILE = RAW_DATA_DIR / "berlin_temperature.csv"
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "weekly_features.csv"

# --------------------------------------------------------------------
# Data source constants
# --------------------------------------------------------------------
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
BERLIN_LATITUDE = 52.52
BERLIN_LONGITUDE = 13.41

START_DATE = "2015-01-01"

# Heating/cooling degree-day base temperatures (deg C)
HEATING_BASE_TEMP = 15.0
COOLING_BASE_TEMP = 18.0

# --------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------
SEASONAL_PERIOD = 52          # weeks in a year
TEST_WEEKS = 104               # 2-year evaluation horizon
RANDOM_STATE = 0

# Best SARIMA order found via the AIC grid search in notebooks/A1_full_pipeline.ipynb
SARIMAX_ORDER = (3, 1, 6)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, SEASONAL_PERIOD)
