"""Screen other algorithms that were not covered in the lectures and are not deep learning.

Selection is by expanding-window CV (2019-2023); test PR-AUC is printed for reference only.
Run:  python -m scripts.other_algorithms
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES, build_features, clean, fit_reference, load_raw, split, time_series_folds
from src.models import NEW_ALGORITHM, SEED, make_model

warnings.filterwarnings("ignore")
params = json.load(open("models/best_params.json"))
df = clean(load_raw())
train, valid, test = split(build_features(df, fit_reference(df)))
dev = pd.concat([train, valid]).reset_index(drop=True)
X, y = dev[FEATURES], dev["high_risk"].astype(int).values
Xt, yt = test[FEATURES], test["high_risk"].astype(int).values
folds = time_series_folds(dev)


def gam():
    from pygam import LogisticGAM
    class GAM(LogisticGAM):  # sklearn-style wrapper
        def fit(self, X, y): return super().fit(np.asarray(X), y)
        def predict_proba(self, X):
            p = super().predict_proba(np.asarray(X)); return np.c_[1 - p, p]
    return GAM(n_splines=10, lam=1.0)


def rulefit():
    from imodels import RuleFitClassifier
    return RuleFitClassifier(max_rules=60, random_state=SEED)


def ngboost():
    from ngboost import NGBClassifier
    return NGBClassifier(n_estimators=400, learning_rate=0.03, verbose=False, random_state=SEED)


CANDIDATES = {
    "EBM (current)": lambda: make_model(NEW_ALGORITHM, params[NEW_ALGORITHM]),
    "Gaussian Naive Bayes": lambda: GaussianNB(),
    "LDA": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "QDA": lambda: make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.1)),
    "Spline GAM (pyGAM)": gam,
    "RuleFit": rulefit,
    "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=20, max_features=0.5,
                                                n_jobs=-1, random_state=SEED),
    "NGBoost": ngboost,
    "Stacking (EBM+LR+kNN+RF+XGB -> LR)": lambda: StackingClassifier(
        [(n, make_model(n, params[n])) for n in [NEW_ALGORITHM, "Logistic Regression", "k-NN", "Random Forest", "XGBoost"]],
        final_estimator=LogisticRegression(), cv=3, n_jobs=1),
}

if __name__ == "__main__":
    rows = []
    for name, build in CANDIDATES.items():
        t = time.time()
        try:
            cv = [average_precision_score(y[va], build().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1])
                  for tr, va in folds]
            te = average_precision_score(yt, build().fit(X, y).predict_proba(Xt)[:, 1])
        except Exception as e:
            print(f"{name}: failed {type(e).__name__} {e}", flush=True); continue
        rows.append((name, np.mean(cv), te))
        print(f"{name:38s} CV PR-AUC {np.mean(cv):.4f}   test PR-AUC {te:.4f}   ({time.time()-t:.0f}s)", flush=True)
    pd.DataFrame(rows, columns=["algorithm", "cv_pr_auc", "test_pr_auc"]).to_csv("reports/other_algorithms.csv", index=False)
