"""Model helpers: the new algorithm (EBM), the standard comparison models, and baselines."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 42
NEW_ALGORITHM = "EBM (new algorithm)"


def make_model(name: str, params: dict):
    """Build an estimator from tuned hyperparameters (as saved in models/best_params.json)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    p = dict(params)
    if name == NEW_ALGORITHM:
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1, **p)
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


def ebm_contributions(model, X):
    """Exact EBM explanation: (intercept, Series of per-term log-odds contributions) for the first row of X."""
    import pandas as pd
    contrib = pd.Series(model.eval_terms(X)[0], index=list(model.term_names_))
    return float(np.atleast_1d(model.intercept_)[0]), contrib
