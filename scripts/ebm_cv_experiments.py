"""CV-only experiments to improve the EBM (the 2024-26 test set is NOT used here).

Each variant is scored with the same expanding-window CV as the notebook (validation years
2019-2023, mean PR-AUC across folds). Run:  python -m scripts.ebm_cv_experiments
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import average_precision_score

from src.features import FEATURES, build_features, clean, fit_reference, load_raw, split, time_series_folds
from src.models import SEED

base = json.load(open("models/best_params.json"))["EBM (new algorithm)"]
train, valid, _ = split(build_features(clean(load_raw()), fit_reference(clean(load_raw()))))
dev = pd.concat([train, valid]).reset_index(drop=True)
X, y, years = dev[FEATURES], dev["high_risk"].astype(int).values, dev["date"].dt.year.values
folds = time_series_folds(dev)

LEVEL = ["rel_now", "rel_lag1", "rel_lag2", "rel_lag3", "rel_ma4", "rel_ma8", "rel_ma12"]
mono = [1 if f in LEVEL else 0 for f in FEATURES]


def recency(half_life):
    def w(yrs):
        return 0.5 ** ((yrs.max() - yrs) / half_life)
    return w


VARIANTS = {
    "current": ({}, None),
    "recency hl=4y": ({}, recency(4)),
    "recency hl=8y": ({}, recency(8)),
    "monotone level": ({"monotone_constraints": mono}, None),
    "interactions=10": ({"interactions": 10}, None),
    "interactions=20": ({"interactions": 20}, None),
    "max_bins=128": ({"max_bins": 128}, None),
    "lr=0.02 + more rounds": ({"learning_rate": 0.02, "max_rounds": 50000, "early_stopping_rounds": 200}, None),
}
only = sys.argv[1:]  # optional: run a subset, e.g. combinations
if only:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in only}


def cv_score(params, weight_fn, bags=4):
    p = {**base, **params, "outer_bags": bags}
    scores = []
    for tr, va in folds:
        m = ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1, **p)
        sw = weight_fn(years[tr]) if weight_fn else None
        m.fit(X.iloc[tr], y[tr], sample_weight=sw)
        scores.append(average_precision_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
    return np.mean(scores), scores


if __name__ == "__main__":
    for name, (params, wfn) in VARIANTS.items():
        t = time.time()
        mean, s = cv_score(params, wfn)
        print(f"{name:26s} CV PR-AUC {mean:.4f}  folds {np.round(s, 3)}  ({time.time()-t:.0f}s)", flush=True)
