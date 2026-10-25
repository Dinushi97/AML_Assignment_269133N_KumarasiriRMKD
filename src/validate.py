"""Data validation gate run before every (re)training, as part of the MLOps pipeline."""
from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = {"year", "week", "start.date", "end.date", "district", "cases"}
EXPECTED_DISTRICTS = 26


class DataValidationError(ValueError):
    pass


def validate_raw(d: pd.DataFrame) -> dict:
    """Raise DataValidationError on schema/quality problems, return a small report otherwise."""
    problems = []
    missing = EXPECTED_COLUMNS - set(d.columns)
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
    else:
        if d["cases"].isna().any():
            problems.append(f"{int(d['cases'].isna().sum())} missing case counts")
        if (d["cases"] < 0).any():
            problems.append("negative case counts")
        if d["district"].nunique() != EXPECTED_DISTRICTS:
            problems.append(f"expected {EXPECTED_DISTRICTS} districts, found {d['district'].nunique()}")
        if d.duplicated(["year", "week", "district"]).any():
            problems.append("duplicate (year, week, district) rows")
        dates = pd.to_datetime(d["start.date"], format="%m/%d/%Y", errors="coerce")
        if dates.isna().any():
            problems.append("unparseable start dates")
        # a weekly count above 5x the historical maximum is almost certainly a data-entry error
        if d["cases"].max() > 15000:
            problems.append(f"implausible weekly count {d['cases'].max()}")
    if problems:
        raise DataValidationError("; ".join(problems))
    return {"rows": len(d), "districts": int(d["district"].nunique()),
            "first_week": str(pd.to_datetime(d["start.date"], format="%m/%d/%Y").min().date()),
            "last_week": str(pd.to_datetime(d["start.date"], format="%m/%d/%Y").max().date())}
