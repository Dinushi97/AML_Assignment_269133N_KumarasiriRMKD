"""Reproducible training pipeline (the MLOps part of the project).

    python -m src.train                  # retrain the selected model with the tuned hyperparameters
    python -m src.train --full           # same, then refit on *all* labelled weeks for serving
    python -m src.train --no-mlflow      # skip experiment tracking
    python -m src.train --data <csv> --out <dir>/dengue_model.pkl   # how the Azure ML job calls it

Steps: validate raw data -> clean/impute -> features -> chronological split -> fit ->
evaluate on test vs the persistence baseline -> quality gate -> version + save -> log to MLflow.
A new model is only written if it passes the gate, so a bad data drop can't silently
replace the model behind the app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.features import (FEATURE_DESCRIPTIONS, FEATURES, HORIZON, RAW_CSV, REFERENCE_YEARS, ROOT, build_features,
                          clean, fit_reference, load_raw, split, time_series_folds)
from src.models import PersistenceBaseline, TimeSeriesStack, make_model
from src.validate import validate_raw

MODELS = ROOT / "models"
STACK = "Stacking (RF+XGB+LGBM+Cat → LR)"
STACK_BASES = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]


def evaluate(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "f1": f1_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "balanced_accuracy": balanced_accuracy_score(y, yhat), "brier": brier_score_loss(y, p)}


def build(name: str, params: dict, folds):
    if name == STACK:
        return TimeSeriesStack({n: make_model(n, params[n]) for n in STACK_BASES}, folds)
    return make_model(name, params[name])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model name (default: the one selected in the notebook)")
    ap.add_argument("--full", action="store_true", help="refit on all labelled weeks after passing the gate")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out", default=str(MODELS / "dengue_model.pkl"))
    ap.add_argument("--data", default=str(RAW_CSV), help="raw weekly CSV (an Azure ML data asset in the cloud job)")
    args = ap.parse_args(argv)

    meta = json.load(open(MODELS / "model_metadata.json"))
    params = json.load(open(MODELS / "best_params.json"))
    name = args.model or meta["model_name"]
    thr = float(meta["decision_threshold"]) if name == meta["model_name"] else 0.5
    t0 = time.time()

    # 1. validate
    raw = load_raw(args.data)
    data_report = validate_raw(raw)
    data_hash = hashlib.sha256(open(args.data, "rb").read()).hexdigest()[:10]
    print("data ok:", data_report, "| sha256:", data_hash)

    # 2-3. preprocess + features
    df = clean(raw)
    ref = fit_reference(df)
    feat = build_features(df, ref)
    train, valid, test = split(feat)
    dev = pd.concat([train, valid]).reset_index(drop=True)
    folds = time_series_folds(dev)

    # 4. fit
    model = build(name, params, folds).fit(dev[FEATURES], dev["high_risk"].astype(int).values)

    # 5. evaluate on the untouched test period
    y = test["high_risk"].astype(int).values
    m = evaluate(y, model.predict_proba(test[FEATURES])[:, 1], thr)
    base = evaluate(y, PersistenceBaseline().predict_proba(test[FEATURES])[:, 1], 0.5)
    print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print("persistence: " + ", ".join(f"{k}={v:.3f}" for k, v in base.items()))

    # 6. quality gate
    passed = m["pr_auc"] > base["pr_auc"] and m["f1"] >= base["f1"]
    print("quality gate:", "PASSED" if passed else "FAILED")

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + data_hash
    if passed:
        if args.full:  # serving model: use every labelled week
            allf = feat.dropna(subset=FEATURES + ["high_risk"]).reset_index(drop=True)
            model = build(name, params, time_series_folds(allf, val_years=(2021, 2022, 2023, 2024, 2025))).fit(
                allf[FEATURES], allf["high_risk"].astype(int).values)
        bundle = {"model": model, "model_name": name, "features": FEATURES, "feature_descriptions": FEATURE_DESCRIPTIONS,
                  "decision_threshold": thr, "reference": ref, "horizon_weeks": HORIZON, "reference_years": REFERENCE_YEARS,
                  "trained_on": "all labelled weeks" if args.full else f"{dev['date'].min().date()} to {dev['date'].max().date()}",
                  "test_metrics": {k: round(v, 4) for k, v in m.items()}, "version": version, "data_sha256": data_hash,
                  "best_params": params.get(name, {n: params[n] for n in STACK_BASES})}
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out)
        df.to_csv(ROOT / "data" / "processed" / "dengue_weekly_clean.csv", index=False)  # history used by the app
        if out.parent != MODELS:  # a cloud job: ship everything the app needs next to the model
            df.to_csv(out.parent / "dengue_weekly_clean.csv", index=False)
            json.dump({k: v for k, v in bundle.items() if k not in ("model", "reference")},
                      open(out.parent / "model_metadata.json", "w"), indent=2, default=str)
        reg = MODELS / "registry"; reg.mkdir(exist_ok=True)
        json.dump({k: v for k, v in bundle.items() if k not in ("model", "reference")},
                  open(reg / f"{version}.json", "w"), indent=2, default=str)
        print("saved", args.out, "version", version)

    # 7. experiment tracking
    if not args.no_mlflow:
        try:
            import mlflow
            on_azure = "azureml" in os.environ.get("MLFLOW_TRACKING_URI", "")
            if not on_azure:  # local: SQLite store; in an Azure ML job the workspace is already the tracking server
                mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
                if mlflow.get_experiment_by_name("dengue-early-warning") is None:
                    mlflow.create_experiment("dengue-early-warning", artifact_location=(ROOT / "mlruns").as_uri())
                mlflow.set_experiment("dengue-early-warning")
            with mlflow.start_run(run_name=None if on_azure else f"{name} {version}"):
                mlflow.set_tags({"model": name, "data_sha256": data_hash, "gate": "passed" if passed else "failed",
                                 "full_refit": str(args.full)})
                flat = params.get(name, {})
                mlflow.log_params({**{k: v for k, v in flat.items()}, "decision_threshold": thr, "horizon": HORIZON,
                                   "n_dev": len(dev), "n_test": len(test)})
                mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
                mlflow.log_metrics({f"baseline_{k}": v for k, v in base.items()})
                mlflow.log_metric("train_seconds", time.time() - t0)
                mlflow.log_dict(data_report, "data_report.json")
                if passed:
                    mlflow.log_artifact(args.out)
            print("logged to MLflow:", "Azure ML workspace" if on_azure else ROOT / "mlflow.db")
        except ImportError:
            print("mlflow not installed – skipping tracking")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
