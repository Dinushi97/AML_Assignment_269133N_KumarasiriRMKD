"""Download the latest Sri Lanka weekly dengue data from the denguedatahub GitHub repo and export to CSV.

    python -m src.fetch_data
Used by the scheduled retraining workflow (.github/workflows/retrain.yml).
"""
import io
import tempfile
import urllib.request
from pathlib import Path

import pyreadr

from src.features import PROVINCE_CSV, RAW_CSV

BASE = "https://raw.githubusercontent.com/thiyangt/denguedatahub/main/data/"


def fetch(name: str):
    with urllib.request.urlopen(BASE + name + ".rda", timeout=60) as r, tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"{name}.rda"
        p.write_bytes(r.read())
        return pyreadr.read_r(str(p))[name]


if __name__ == "__main__":
    weekly = fetch("srilanka_weekly_data")
    weekly.to_csv(RAW_CSV, index=False)
    fetch("sl_province_districts").to_csv(PROVINCE_CSV, index=False)
    print(f"saved {len(weekly)} rows to {RAW_CSV}; last week starts {weekly['start.date'].iloc[-1]}")
