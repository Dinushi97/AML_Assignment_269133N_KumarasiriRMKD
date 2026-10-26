"""CV-only refinements of the tuned spline GAM (the 2024-26 test set is NOT used here).

Each variant is scored with the notebook's expanding-window CV (validation years 2019-2023, mean PR-AUC).
Run:  python -m scripts.gam_cv_experiments
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.features import FEATURES, build_features, clean, fit_reference, load_raw, split, time_series_folds
from src.models import CANDIDATE_INTERACTIONS, NEW_ALGORITHM, SplineGAMClassifier

warnings.filterwarnings("ignore")
best = dict(json.load(open("models/best_params.json"))[NEW_ALGORITHM])
k = int(best.pop("n_interactions", 0))
base = {**best, "interactions": CANDIDATE_INTERACTIONS[:k]}
df = clean(load_raw())
train, valid, _ = split(build_features(df, fit_reference(df)))
dev = pd.concat([train, valid]).reset_index(drop=True)
X, y, years = dev[FEATURES], dev["high_risk"].astype(int).values, dev["date"].dt.year.values
folds = time_series_folds(dev)
LEVEL = ("rel_now", "rel_lag1", "rel_lag2", "rel_lag3", "rel_ma4", "rel_ma8", "rel_ma12")


def recency(half_life):
    return lambda yrs: 0.5 ** ((yrs.max() - yrs) / half_life)


VARIANTS = {
    "tuned (selected)": ({}, None),
    "no interactions (purely additive)": ({"interactions": ()}, None),
    "all 3 candidate interactions": ({"interactions": CANDIDATE_INTERACTIONS}, None),
    "monotone curves for level features": ({"monotone_features": LEVEL}, None),
    "lam x 10 (smoother)": ({"lam": best["lam"] * 10}, None),
    "lam / 10 (wigglier)": ({"lam": best["lam"] / 10}, None),
    "recency weighting, half-life 4 y": ({}, recency(4)),
    "recency weighting, half-life 8 y": ({}, recency(8)),
}

if __name__ == "__main__":
    rows = []
    for name, (params, wfn) in VARIANTS.items():
        t = time.time(); sc = []
        for tr, va in folds:
            m = SplineGAMClassifier(**{**base, **params})
            m.fit(X.iloc[tr], y[tr], sample_weight=wfn(years[tr]) if wfn else None)
            sc.append(average_precision_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
        rows.append((name, np.mean(sc)))
        print(f"{name:38s} CV PR-AUC {np.mean(sc):.4f}  ({time.time()-t:.0f}s)", flush=True)
    pd.DataFrame(rows, columns=["variant", "cv_pr_auc"]).to_csv("reports/gam_cv_ablation.csv", index=False)
