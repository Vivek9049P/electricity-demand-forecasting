# src/electricity_demand/data.py
"""
Loading and cleaning of the raw electricity load and temperature data.

Downloading is kept in scripts/download_data.py so that this module
never triggers network access on import — code that only needs to
*read* processed data (the pipeline, tests, notebooks) can import this
module safely offline.
"""

import numpy as np
import pandas as pd

from electricity_demand.config import (
    PROCESSED_FEATURES_FILE,
    LOAD_COLUMN,
)


def load_raw_load_series(raw_csv_path, start_date="2015-01-01"):
    """
    Read the OPSD hourly CSV and return a cleaned hourly load series in MW.

    Parameters
    ----------
    raw_csv_path : str or Path
        Path to the downloaded OPSD time_series_60min_singleindex.csv file.
    start_date : str
        Earliest date to keep (inclusive).
    """
    df = pd.read_csv(
        raw_csv_path,
        usecols=["utc_timestamp", LOAD_COLUMN],
        parse_dates=["utc_timestamp"],
    )
    df = df.rename(columns={"utc_timestamp": "date", LOAD_COLUMN: "load_mw"})
    df = df.set_index("date").sort_index()

    load = df["load_mw"].astype(float)
    load = load[load.notna()]
    load = load[start_date:]
    return load


def aggregate_to_weekly(hourly_load_mw):
    """Aggregate an hourly load series (MW) to weekly mean load in GW."""
    weekly = hourly_load_mw.resample("W").mean() / 1000.0
    weekly = weekly.asfreq("W").interpolate("time")
    weekly.name = "load_gw"
    return weekly


def load_processed_data():
    """
    Load the processed weekly modelling table.

    Returns a DataFrame indexed by week-ending date with at least a
    ``load_gw`` column, plus any temperature/holiday covariates created
    by scripts/make_features.py.

    Raises
    ------
    FileNotFoundError
        If the processed dataset has not been created yet. Run
        ``python scripts/download_data.py`` followed by
        ``python scripts/make_features.py`` first.
    """
    if not PROCESSED_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_FEATURES_FILE}.\n"
            "Run 'python scripts/download_data.py' then "
            "'python scripts/make_features.py' to create it."
        )

    data = pd.read_csv(PROCESSED_FEATURES_FILE, index_col=0, parse_dates=True)
    data = data.sort_index()
    return data
