"""Shared data loading, cleaning and feature engineering.

Used by the notebook, the training pipeline (src/train.py) and the Streamlit app,
so the features seen at inference time are computed exactly as in training.

Problem: at the end of report week t, predict whether a district's dengue cases in
week t+2 will exceed that district's "high-transmission" threshold
(75th percentile of its weekly cases in 2012-2021, inside the training period).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "srilanka_weekly_dengue.csv"
PROVINCE_CSV = ROOT / "data" / "raw" / "sl_province_districts.csv"

HORIZON = 2                     # weeks ahead
THRESHOLD_QUANTILE = 0.75       # district "high-transmission" cut-off
TRAIN_YEARS = (2007, 2021)      # model training years
REFERENCE_YEARS = (2012, 2021)  # thresholds + seasonal profiles: most recent decade of training data only
VALID_YEARS = (2022, 2023)
TEST_START_YEAR = 2024

FEATURES = [
    "rel_now", "rel_lag1", "rel_lag2", "rel_lag3",
    "rel_ma4", "rel_ma8", "rel_ma12", "rel_ma52",
    "growth_1w", "growth_4w", "volatility_8w",
    "rel_same_week_last_year", "seasonal_norm",
    "target_week_sin", "target_week_cos",
    "national_growth_1w", "national_growth_4w", "share_districts_high",
    "district_baseline",
]

FEATURE_DESCRIPTIONS = {
    "rel_now": "This week's cases vs district threshold (log ratio)",
    "rel_lag1": "Cases 1 week ago vs threshold (log ratio)",
    "rel_lag2": "Cases 2 weeks ago vs threshold (log ratio)",
    "rel_lag3": "Cases 3 weeks ago vs threshold (log ratio)",
    "rel_ma4": "4-week average vs threshold (log ratio)",
    "rel_ma8": "8-week average vs threshold (log ratio)",
    "rel_ma12": "12-week average vs threshold (log ratio)",
    "rel_ma52": "52-week average vs threshold (log ratio)",
    "growth_1w": "Week-on-week growth (log)",
    "growth_4w": "Growth over 4 weeks (log)",
    "volatility_8w": "Std. dev. of weekly log cases, last 8 weeks",
    "rel_same_week_last_year": "Cases in the target week last year vs threshold",
    "seasonal_norm": "Typical level for the target week (2012-21 median)",
    "target_week_sin": "Seasonality of target week (sine)",
    "target_week_cos": "Seasonality of target week (cosine)",
    "national_growth_1w": "Sri Lanka total cases, 1-week growth (log)",
    "national_growth_4w": "Sri Lanka total cases, 4-week growth (log)",
    "share_districts_high": "Share of districts above their threshold this week",
    "district_baseline": "District's usual burden: log(1 + threshold)",
}


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["start.date"], format="%m/%d/%Y")
    return d


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Complete district x week grid, impute missing weeks, drop the lone 2006 week."""
    dates = np.sort(raw["date"].unique())
    districts = sorted(raw["district"].unique())
    grid = pd.MultiIndex.from_product([districts, dates], names=["district", "date"]).to_frame(index=False)
    df = grid.merge(raw[["district", "date", "cases"]], on=["district", "date"], how="left")
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    df["was_imputed"] = df["cases"].isna()
    # Data-quality rule: a whole calendar year of zeros, between two years with >= 50 cases each,
    # is a reporting failure (missing data recorded as 0), not a real absence of dengue.
    # (Triggers only for Monaragala 2020: 0 cases in 52 weeks vs 330 in 2019 and 192 in 2021.)
    yr = df["date"].dt.year
    tot = df.groupby(["district", yr])["cases"].sum(min_count=1)
    df["unreliable"] = False
    for (dist, y), v in tot.items():
        if v == 0 and tot.get((dist, y - 1), 0) >= 50 and tot.get((dist, y + 1), 0) >= 50:
            m = (df["district"] == dist) & (yr == y)
            df.loc[m, "cases"] = np.nan
            df.loc[m, "unreliable"] = True
    # Data-quality rule 2: isolated implausible weeks. A count below 10% of BOTH neighbouring weeks
    # (or above 10x both), when the neighbours are >= 30, is a missed report recorded as 0 or a
    # row misaligned between districts (e.g. week of 30 Mar 2026: Colombo 9, Kilinochchi 117).
    # Such values are treated as missing and interpolated like any other missing week.
    gc = df.groupby("district")["cases"]
    prev, nxt = gc.shift(1), gc.shift(-1)
    lo, hi = np.fmin(prev, nxt), np.fmax(prev, nxt)
    implausible = ((df["cases"] < 0.1 * lo) & (lo >= 30)) | ((df["cases"] > 10 * hi) & (df["cases"] >= 30))
    df["implausible"] = implausible.fillna(False)
    df.loc[df["implausible"], "cases"] = np.nan
    df["was_imputed"] = df["was_imputed"] | df["implausible"]

    # linear interpolation within each district (counts are smooth week to week)
    df["cases"] = df.groupby("district")["cases"].transform(lambda s: s.interpolate(limit_direction="both"))
    df["cases"] = df["cases"].round().clip(lower=0)
    df = df[df["date"] >= "2007-01-01"].reset_index(drop=True)
    prov = pd.read_csv(PROVINCE_CSV)
    df = df.merge(prov, on="district", how="left")
    return df


# --------------------------------------------------------------------------- #
# Reference statistics learned from the training years only (no leakage)
# --------------------------------------------------------------------------- #
def fit_reference(df: pd.DataFrame) -> dict:
    tr = df[df["date"].dt.year.between(*REFERENCE_YEARS) & ~df["unreliable"]].copy()
    thr = tr.groupby("district")["cases"].quantile(THRESHOLD_QUANTILE).clip(lower=1)
    tr["woy"] = tr["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    tr["rel"] = np.log1p(tr["cases"]) - np.log1p(tr["district"].map(thr))
    seas = tr.groupby(["district", "woy"])["rel"].median()
    # smooth the seasonal profile over +-2 weeks (circular)
    smooth = {}
    for dist, s in seas.groupby(level=0):
        v = s.droplevel(0).reindex(range(1, 53)).interpolate(limit_direction="both").values
        v = np.concatenate([v[-2:], v, v[:2]])
        smooth[dist] = np.convolve(v, np.ones(5) / 5, mode="valid").tolist()
    return {"threshold": thr.to_dict(), "seasonal": smooth}


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, ref: dict, horizon: int = HORIZON) -> pd.DataFrame:
    """Return one row per district-week with features known at week t and target at t+horizon."""
    df = df.sort_values(["district", "date"]).reset_index(drop=True).copy()
    thr = df["district"].map(ref["threshold"])
    lthr = np.log1p(thr)
    lc = np.log1p(df["cases"])
    g = df.groupby("district")["cases"]
    glc = lc.groupby(df["district"])

    out = df[["district", "province", "date", "cases"]].copy()
    out["threshold"] = thr
    out["rel_now"] = lc - lthr
    for k in (1, 2, 3):
        out[f"rel_lag{k}"] = np.log1p(g.shift(k)) - lthr
    for w in (4, 8, 12, 52):
        out[f"rel_ma{w}"] = np.log1p(g.transform(lambda s, w=w: s.rolling(w).mean())) - lthr
    out["growth_1w"] = lc - glc.shift(1)
    out["growth_4w"] = lc - glc.shift(4)
    out["volatility_8w"] = glc.transform(lambda s: s.rolling(8).std())
    out["rel_same_week_last_year"] = np.log1p(g.shift(52 - horizon)) - lthr

    target_date = df["date"] + pd.Timedelta(weeks=horizon)
    twoy = target_date.dt.isocalendar().week.astype(int).clip(upper=52)
    out["target_date"] = target_date
    out["seasonal_norm"] = [ref["seasonal"][d][w - 1] for d, w in zip(df["district"], twoy)]
    out["target_week_sin"] = np.sin(2 * np.pi * twoy / 52)
    out["target_week_cos"] = np.cos(2 * np.pi * twoy / 52)

    nat = np.log1p(df.groupby("date")["cases"].sum())
    out["national_growth_1w"] = df["date"].map(nat - nat.shift(1))
    out["national_growth_4w"] = df["date"].map(nat - nat.shift(4))
    out["share_districts_high"] = (out["rel_now"] > 0).groupby(df["date"]).transform("mean")
    out["district_baseline"] = lthr

    # rows whose 52-week look-back or target week touches unreliable (imputed-for-a-year) data
    unrel = df["unreliable"].astype(int) if "unreliable" in df else pd.Series(0, index=df.index)
    gu = unrel.groupby(df["district"])
    back = gu.transform(lambda s: s.rolling(53, min_periods=1).max())
    fwd = sum(gu.shift(-k).fillna(0) for k in range(1, horizon + 1))
    out["unreliable_window"] = (back + fwd) > 0

    out["target_cases"] = g.shift(-horizon)
    out["high_risk"] = (out["target_cases"] > thr).astype("float")
    out.loc[out["target_cases"].isna(), "high_risk"] = np.nan
    return out


def split(feat: pd.DataFrame):
    """Chronological split by report week: train 2007-21 | valid 2022-23 | test 2024+.

    Rows are *purged* when their target week (t+2) falls in the next period, so no
    training label overlaps the validation/test weeks.
    """
    f = feat.dropna(subset=FEATURES + ["high_risk"])
    f = f[~f["unreliable_window"]]
    yr, tyr = f["date"].dt.year, f["target_date"].dt.year
    train = f[yr.between(*TRAIN_YEARS) & (tyr <= TRAIN_YEARS[1])]
    valid = f[yr.between(*VALID_YEARS) & (tyr <= VALID_YEARS[1])]
    test = f[yr >= TEST_START_YEAR]
    return train, valid, test


def time_series_folds(dev: pd.DataFrame, val_years=(2019, 2020, 2021, 2022, 2023)):
    """Expanding-window CV: train on report weeks whose target is before year Y, validate on year Y."""
    yr = dev["date"].dt.year.values
    tyr = dev["target_date"].dt.year.values
    idx = np.arange(len(dev))
    return [(idx[tyr < y], idx[(yr == y) & (tyr == y)]) for y in val_years]
