"""Model helpers: baselines and a time-series-aware stacking ensemble."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression


SEED = 42


def make_model(name: str, params: dict):
    """Build an estimator from tuned hyperparameters (as saved in models/best_params.json)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    p = dict(params)
    if name == "Logistic Regression":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(solver="saga", max_iter=3000, **p))])
    if name == "Random Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=SEED, **p)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED, **p)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=-1, **p)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=SEED, verbose=0, thread_count=-1, **p)
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


class TimeSeriesStack(BaseEstimator, ClassifierMixin):
    """Stacking with out-of-fold predictions from expanding-window time-series folds.

    sklearn's StackingClassifier uses shuffled K-fold, which would let base models
    see the future. Here each base model is fit on years < Y and predicts year Y,
    the meta-learner (logistic regression) is trained on those honest predictions,
    then the base models are refit on all development data.
    """

    def __init__(self, base_models: dict, folds, meta_C: float = 1.0):
        self.base_models = base_models
        self.folds = folds
        self.meta_C = meta_C

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        y = np.asarray(y)
        names = list(self.base_models)
        oof_idx = np.concatenate([va for _, va in self.folds])
        oof = np.zeros((len(oof_idx), len(names)))
        for j, n in enumerate(names):
            preds = []
            for tr, va in self.folds:
                m = clone(self.base_models[n]).fit(X.iloc[tr], y[tr])
                preds.append(m.predict_proba(X.iloc[va])[:, 1])
            oof[:, j] = np.concatenate(preds)
        self.oof_pred_ = oof
        self.meta_ = LogisticRegression(C=self.meta_C, max_iter=1000).fit(self._logit(oof), y[oof_idx])
        self.fitted_ = {n: clone(m).fit(X, y) for n, m in self.base_models.items()}
        self.classes_ = np.array([0, 1])
        return self

    def base_predictions(self, X):
        return np.column_stack([self.fitted_[n].predict_proba(X)[:, 1] for n in self.base_models])

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._logit(self.base_predictions(X)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
