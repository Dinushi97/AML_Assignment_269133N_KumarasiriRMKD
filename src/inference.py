"""Inference helpers shared by the Streamlit app and the tests."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import ROOT, build_features

MODEL_PATH = ROOT / "models" / "dengue_model.pkl"
HISTORY_PATH = ROOT / "data" / "processed" / "dengue_weekly_clean.csv"
WINDOW_WEEKS = 60  # enough history for the 52-week features


def load_bundle(path: Path | str = MODEL_PATH) -> dict:
    return joblib.load(path)


def load_history(path: Path | str = HISTORY_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def _window(hist: pd.DataFrame, report_date) -> pd.DataFrame:
    report_date = pd.Timestamp(report_date)
    start = report_date - pd.Timedelta(weeks=WINDOW_WEEKS)
    return hist[(hist["date"] >= start) & (hist["date"] <= report_date)].copy()


def features_for_week(bundle: dict, hist: pd.DataFrame, report_date, overrides: dict | None = None) -> pd.DataFrame:
    """Feature rows for every district at `report_date`.

    overrides: {district: {date: cases}} lets the user replace recent counts (what-if analysis
    or entering this week's numbers before they are in the dataset).
    """
    w = _window(hist, report_date)
    if overrides:
        for dist, vals in overrides.items():
            for d, c in vals.items():
                w.loc[(w["district"] == dist) & (w["date"] == pd.Timestamp(d)), "cases"] = max(float(c), 0.0)
    f = build_features(w, bundle["reference"], horizon=bundle["horizon_weeks"])
    return f[f["date"] == pd.Timestamp(report_date)].reset_index(drop=True)


def predict_week(bundle: dict, hist: pd.DataFrame, report_date, overrides: dict | None = None) -> pd.DataFrame:
    f = features_for_week(bundle, hist, report_date, overrides)
    f["probability"] = bundle["model"].predict_proba(f[bundle["features"]])[:, 1]
    f["high_risk_pred"] = (f["probability"] >= bundle["decision_threshold"]).astype(int)
    # actual outcome if it is already known in the history
    actual = hist.set_index(["district", "date"])["cases"]
    f["actual_target_cases"] = [actual.get((d, t), np.nan) for d, t in zip(f["district"], f["target_date"])]
    return f.sort_values("probability", ascending=False).reset_index(drop=True)


def risk_band(p: float, thr: float) -> str:
    if p >= max(thr + 0.25, 0.75):
        return "Very high"
    if p >= thr:
        return "High"
    if p >= thr * 0.6:
        return "Elevated"
    return "Low"
