"""Model helpers: the new algorithm (spline GAM), the standard comparison models, and baselines."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 42
NEW_ALGORITHM = "Spline GAM (new algorithm)"

# candidate pairwise interactions, in order of priority (tuning chooses how many to use)
CANDIDATE_INTERACTIONS = (("rel_now", "district_baseline"), ("target_week_sin", "share_districts_high"),
                          ("rel_ma4", "growth_4w"))


def make_model(name: str, params: dict):
    """Build an estimator from tuned hyperparameters (as saved in models/best_params.json)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    p = dict(params)
    if name == NEW_ALGORITHM:
        k = int(p.pop("n_interactions", 0))
        return SplineGAMClassifier(interactions=CANDIDATE_INTERACTIONS[:k], **p)
    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(solver="saga", max_iter=3000, **p))])
    if name == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=SEED, **p)
    if name == "k-NN":
        from sklearn.neighbors import KNeighborsClassifier
        return Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(**p))])
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=-1, random_state=SEED, **p)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED, **p)
    raise ValueError(f"Unknown model {name}")


class PersistenceBaseline(BaseEstimator, ClassifierMixin):
    """'Next is like now': predicts high risk if this week's cases already exceed the threshold."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (np.asarray(X["rel_now"]) > 0).astype(float)
        return np.c_[1 - p, p]

    def predict(self, X):
        return self.predict_proba(X)[:, 1].astype(int)


class SplineGAMClassifier(BaseEstimator, ClassifierMixin):
    """Logistic generalized additive model with penalised B-splines (P-splines), fitted with pyGAM.

    logit P(high risk) = intercept + f_1(x_1) + ... + f_p(x_p) + sum of tensor-product terms f_jk(x_j, x_k)

    Each f is a B-spline (cubic by default) with ``n_splines`` basis functions; a second-difference penalty of
    strength ``lam`` keeps the curves smooth. Defaults are pyGAM's own (20 splines, lam 0.6, no interactions). The model is fitted by penalised iteratively re-weighted least squares.
    Every prediction is an exact sum of readable terms (see ``term_contributions``).
    """

    def __init__(self, n_splines=20, lam=0.6, spline_order=3, interactions=(), te_splines=6, monotone_features=(),
                 max_iter=200):
        self.n_splines = n_splines
        self.monotone_features = monotone_features
        self.lam = lam
        self.spline_order = spline_order
        self.interactions = interactions
        self.te_splines = te_splines
        self.max_iter = max_iter

    def _array(self, X):
        if hasattr(X, "columns"):
            return X[self.feature_names_in_].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def fit(self, X, y, sample_weight=None):
        import warnings
        from pygam import LogisticGAM, s, te
        self.feature_names_in_ = np.array(list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])])
        names = list(self.feature_names_in_)
        def spline(j):
            con = "monotonic_inc" if names[j] in self.monotone_features else None
            return s(j, n_splines=self.n_splines, spline_order=self.spline_order, lam=self.lam, constraints=con)
        terms = spline(0)
        for j in range(1, len(names)):
            terms += spline(j)
        for a, b in self.interactions:
            terms += te(names.index(a), names.index(b), n_splines=self.te_splines, lam=self.lam)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gam_ = LogisticGAM(terms, max_iter=self.max_iter).fit(self._array(X), np.asarray(y), weights=sample_weight)
        self.term_names_ = names + [f"{a} & {b}" for a, b in self.interactions]
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = len(names)
        return self

    @property
    def intercept_(self):
        return float(self.gam_.coef_[-1])

    def term_contributions(self, X):
        """Per-term contributions to the log-odds (rows = samples). intercept_ + row sum = logit exactly."""
        import pandas as pd
        Xa = self._array(X)
        parts = np.column_stack([self.gam_.partial_dependence(term=i, X=Xa) for i in range(len(self.term_names_))])
        return pd.DataFrame(parts, columns=self.term_names_, index=getattr(X, "index", None))

    def decision_function(self, X):
        return self.gam_._linear_predictor(self._array(X))

    def predict_proba(self, X):
        p = self.gam_.predict_proba(self._array(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def gam_contributions(model, X):
    """Exact GAM explanation: (intercept, Series of per-term log-odds contributions) for the first row of X."""
    return model.intercept_, model.term_contributions(X).iloc[0]
