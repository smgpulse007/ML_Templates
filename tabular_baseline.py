"""Reproducible binary classification with train-only preprocessing and tuning."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Experiment:
    search: GridSearchCV
    report: dict
    train_positions: np.ndarray
    test_positions: np.ndarray


def _validate(frame: pd.DataFrame, target: str):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("data must be a nonempty pandas DataFrame")
    if (not frame.columns.is_unique or
            any(not isinstance(c, str) or not c.strip() for c in frame.columns)):
        raise ValueError("column names must be unique nonempty strings")
    if target not in frame or len(frame.columns) < 2:
        raise ValueError("target must exist alongside at least one feature column")
    y = frame[target]
    if (y.isna().any() or any(isinstance(value, (bool, np.bool_)) for value in y)
            or set(y.unique()) != {0, 1}):
        raise ValueError("target must contain both binary labels 0 and 1, without missing values")
    features = frame.drop(columns=[target]).copy()
    numeric, categorical = [], []
    for name in features:
        column = features[name]
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            if pd.api.types.is_complex_dtype(column):
                raise ValueError("complex features are not supported")
            values = column.to_numpy(dtype=np.float64, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError("numeric features cannot contain infinity")
            features[name] = values
            numeric.append(name)
        else:
            # Keep categorical types unambiguous, including pandas StringDtype.
            values = column.to_numpy(dtype=object, na_value=np.nan)
            present = [v for v in values if not pd.isna(v)]
            if any(not isinstance(v, str) for v in present):
                raise ValueError("categorical features must contain strings or missing values")
            if "__missing__" in present:
                raise ValueError("the category __missing__ is reserved for missing values")
            features[name] = values
            categorical.append(name)
    return features, y.to_numpy(dtype=np.int64), numeric, categorical


def run_experiment(frame: pd.DataFrame, target: str, *, seed: int = 42,
                   test_size: float = 0.25, folds: int = 3) -> Experiment:
    """Fit only on a stratified training split; evaluate the selected model once.

    Designed for independent rows, numeric/string features and a 0/1 target.
    Callers must remove target proxies and choose group/time splits for dependent
    observations. This helper cannot infer those domain-specific boundaries.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer from 0 through 2**32 - 1")
    if isinstance(folds, bool) or not isinstance(folds, Integral) or folds < 2:
        raise ValueError("folds must be an integer of at least 2")
    if isinstance(test_size, bool) or not isinstance(test_size, Real) or not 0 < test_size < 1:
        raise ValueError("test_size must be a fraction between 0 and 1")
    features, y, numeric, categorical = _validate(frame, target)
    if np.bincount(y).min() < 2:
        raise ValueError("each class needs at least two rows for a stratified holdout")
    try:
        train, test = train_test_split(np.arange(len(frame)), test_size=test_size,
                                       stratify=y, random_state=int(seed))
    except ValueError as exc:
        raise ValueError("the requested holdout cannot retain both classes") from exc
    if np.bincount(y[train], minlength=2).min() < folds or len(np.unique(y[test])) != 2:
        raise ValueError("training needs at least folds rows per class and holdout needs both classes")
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer(transformers, remainder="drop")),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=int(seed))),
    ])
    cv = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    search = GridSearchCV(pipeline, {"model__C": [0.1, 1.0, 10.0]}, scoring="average_precision",
                          cv=cv, n_jobs=1, error_score="raise", refit=True)
    search.fit(features.iloc[train], y[train])
    # No fitted preprocessing, hyperparameter choice or threshold tuning uses holdout.
    probability = search.predict_proba(features.iloc[test])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    heldout_y = y[test]
    report = {
        "task": "binary_classification", "split": "stratified_random",
        "seed": int(seed), "rows": len(frame), "train_rows": len(train), "holdout_rows": len(test),
        "features": {"numeric": len(numeric), "categorical": len(categorical)},
        "selection": {"metric": "average_precision", "folds": int(folds),
                      "best_mean_cv_score": float(search.best_score_), "best_C": float(search.best_params_["model__C"])},
        "holdout": {"average_precision": float(average_precision_score(heldout_y, probability)),
                    "roc_auc": float(roc_auc_score(heldout_y, probability)),
                    "balanced_accuracy": float(balanced_accuracy_score(heldout_y, prediction)),
                    "positive_rate": float(np.mean(heldout_y)),
                    "confusion_matrix_labels": [0, 1],
                    "confusion_matrix": confusion_matrix(heldout_y, prediction, labels=[0, 1]).tolist(),
                    "decision_threshold": 0.5},
        "limits": ["Independent rows only; choose group/time splits for dependent observations.",
                   "Class-weighted probabilities are not calibrated probabilities.",
                   "Remove target leakage and ID columns before training; this template cannot detect them."],
    }
    return Experiment(search, report, train, test)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True, help="column with binary labels 0 and 1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args(argv)
    try:
        if args.csv.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("CSV exceeds the 20 MiB example limit")
        with args.csv.open(encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), [])
        if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
            raise ValueError("CSV headers must be unique nonempty strings")
        frame = pd.read_csv(args.csv)
        if len(frame) > 100000 or len(frame.columns) > 200:
            raise ValueError("CSV exceeds 100,000 rows or 200 columns")
        result = run_experiment(frame, args.target, seed=args.seed, folds=args.folds, test_size=args.test_size)
    except (OSError, ValueError, TypeError, csv.Error):
        # A parser/estimator error can include raw values. Keep CLI output generic.
        print(json.dumps({"error": "Unable to train. Check CSV limits, numeric/string features, binary target, and split/fold settings."}), file=sys.stderr)
        return 2
    print(json.dumps(result.report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
