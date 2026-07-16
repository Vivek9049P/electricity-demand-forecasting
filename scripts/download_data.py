# scripts/download_data.py
"""
Download the raw electricity load (OPSD) and Berlin temperature
(Open-Meteo) data used by the rest of the pipeline, and save them
under data/raw/.

Run once (or whenever you want to refresh the raw data):

    python scripts/download_data.py
"""

import requests
import pandas as pd

from electricity_demand.config import (
    OPSD_URL,
    OPEN_METEO_URL,
    BERLIN_LATITUDE,
    BERLIN_LONGITUDE,
    START_DATE,
    RAW_DATA_DIR,
    RAW_LOAD_FILE,
    RAW_TEMPERATURE_FILE,
)


def download_load_data():
    print(f"Downloading electricity load data from {OPSD_URL} ...")
    response = requests.get(OPSD_URL, timeout=120)
    response.raise_for_status()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_LOAD_FILE.write_bytes(response.content)
    print(f"Saved to {RAW_LOAD_FILE}")


def download_temperature_data(start_date=START_DATE, end_date=None):
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"Downloading Berlin temperature data ({start_date} to {end_date}) ...")
    params = {
        "latitude": BERLIN_LATITUDE,
        "longitude": BERLIN_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    response.raise_for_status()
    daily = response.json()["daily"]

    temp = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_mean": daily["temperature_2m_mean"],
    }).set_index("date")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp.to_csv(RAW_TEMPERATURE_FILE)
    print(f"Saved to {RAW_TEMPERATURE_FILE}")


def main():
    download_load_data()
    download_temperature_data()


if __name__ == "__main__":
    main()
