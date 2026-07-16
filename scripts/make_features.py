# scripts/make_features.py
"""
Build the processed weekly modelling table from the raw data files
downloaded by scripts/download_data.py, and save it to
data/processed/weekly_features.csv.

Run after download_data.py:

    python scripts/make_features.py
"""

import pandas as pd

from electricity_demand.config import (
    RAW_LOAD_FILE,
    RAW_TEMPERATURE_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_FEATURES_FILE,
    START_DATE,
)
from electricity_demand.data import load_raw_load_series, aggregate_to_weekly
from electricity_demand.features import add_degree_days, resample_temperature_to_weekly


def main():
    if not RAW_LOAD_FILE.exists() or not RAW_TEMPERATURE_FILE.exists():
        raise FileNotFoundError(
            "Raw data not found. Run 'python scripts/download_data.py' first."
        )

    # Load and weekly-aggregate electricity demand
    hourly_load = load_raw_load_series(RAW_LOAD_FILE, start_date=START_DATE)
    weekly_load = aggregate_to_weekly(hourly_load)

    # Load and weekly-aggregate temperature / degree days
    daily_temp = pd.read_csv(RAW_TEMPERATURE_FILE, index_col=0, parse_dates=True)
    daily_temp = add_degree_days(daily_temp)
    weekly_temp = resample_temperature_to_weekly(daily_temp, target_index=weekly_load.index)

    # Merge into a single processed feature table
    features = pd.DataFrame({"load_gw": weekly_load}).join(weekly_temp)
    features = features.interpolate("time").dropna()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_FEATURES_FILE)
    print(f"Saved processed feature table ({features.shape[0]} rows) to {PROCESSED_FEATURES_FILE}")


if __name__ == "__main__":
    main()
