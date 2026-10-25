import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (FEATURES, REFERENCE_YEARS, build_features, clean, fit_reference, load_raw, split,
                          time_series_folds)
from src.validate import DataValidationError, validate_raw


@pytest.fixture(scope="module")
def data():
    raw = load_raw()
    df = clean(raw)
    ref = fit_reference(df)
    return raw, df, ref, build_features(df, ref)


def test_raw_data_passes_validation(data):
    rep = validate_raw(data[0])
    assert rep["districts"] == 26


def test_validation_rejects_bad_data(data):
    bad = data[0].copy()
    bad.loc[0, "cases"] = -5
    with pytest.raises(DataValidationError):
        validate_raw(bad)


def test_clean_grid_is_complete(data):
    _, df, _, _ = data
    assert df["cases"].notna().all()
    assert df.groupby("district").size().nunique() == 1


def test_no_future_leakage_in_features(data):
    """Changing future counts must not change features at week t."""
    _, df, ref, feat = data
    t = pd.Timestamp("2020-06-06")
    t = df["date"][df["date"] >= t].min()
    df2 = df.copy()
    df2.loc[df2["date"] > t, "cases"] *= 10
    f1 = feat[feat["date"] == t].set_index("district")[FEATURES]
    f2 = build_features(df2, ref)
    f2 = f2[f2["date"] == t].set_index("district")[FEATURES]
    pd.testing.assert_frame_equal(f1.sort_index(), f2.sort_index())


def test_reference_uses_only_reference_years(data):
    _, df, ref, _ = data
    df2 = df.copy()
    df2.loc[df2["date"].dt.year > REFERENCE_YEARS[1], "cases"] = 99999
    assert fit_reference(df2)["threshold"] == ref["threshold"]


def test_split_is_chronological_and_purged(data):
    train, valid, test = split(data[3])
    assert train["target_date"].max() < valid["date"].min()
    assert valid["target_date"].max() < test["date"].min()
    dev = pd.concat([train, valid]).reset_index(drop=True)
    for tr, va in time_series_folds(dev):
        assert dev.loc[tr, "target_date"].max() < dev.loc[va, "date"].min()


@pytest.mark.skipif(not (ROOT / "models" / "dengue_model.pkl").exists(), reason="model not trained yet")
def test_saved_model_and_app_inference():
    from src.inference import load_bundle, load_history, predict_week
    b = load_bundle()
    hist = load_history()
    ho = pd.read_csv(ROOT / "data" / "holdout_sample.csv", parse_dates=["date", "target_date"])
    p_direct = b["model"].predict_proba(ho[b["features"]])[:, 1][0]
    pw = predict_week(b, hist, ho["date"].iloc[0])
    p_app = pw.loc[pw["district"] == ho["district"].iloc[0], "probability"].iloc[0]
    assert 0 <= p_direct <= 1
    assert abs(p_direct - p_app) < 1e-6, "app feature window must reproduce training features"
    assert len(pw) == 26
