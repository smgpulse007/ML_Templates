"""Train and evaluate an actual model on locally generated synthetic rows."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from tabular_baseline import run_experiment


def synthetic_data(seed=42, rows=400):
    values, labels = make_classification(n_samples=rows, n_features=5, n_informative=3,
                                         n_redundant=0, weights=[0.8, 0.2], random_state=seed)
    frame = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(5)])
    rng = np.random.default_rng(seed)
    frame["channel"] = rng.choice(["web", "mobile", "desktop"], size=rows)
    frame.loc[rng.random(rows) < 0.08, "feature_0"] = np.nan
    frame.loc[rng.random(rows) < 0.08, "channel"] = np.nan
    frame["outcome"] = labels
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_experiment(synthetic_data(), "outcome")
    report = {"data_source": "generated_synthetic_fixture", **result.report}
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("ML Templates | real training on synthetic data")
    print(f"Rows: {report['rows']} | train: {report['train_rows']} | holdout: {report['holdout_rows']}")
    print("Numeric + categorical features; missing values handled inside CV folds.")
    print("Select C using 3-fold training average precision; holdout stays separate.")
    print(f"Selected C: {report['selection']['best_C']}")
    print(f"Training CV average precision: {report['selection']['best_mean_cv_score']:.4f}")
    print(f"Holdout average precision: {report['holdout']['average_precision']:.4f}")
    print(f"Holdout positive rate: {report['holdout']['positive_rate']:.4f}")
    print(f"Holdout ROC AUC: {report['holdout']['roc_auc']:.4f}")
    print(f"Holdout balanced accuracy at 0.5: {report['holdout']['balanced_accuracy']:.4f}")
    print(f"Confusion matrix [0, 1]: {report['holdout']['confusion_matrix']}")
    print("Synthetic demonstration only; these scores are not real-world validation.")


if __name__ == "__main__":
    main()
