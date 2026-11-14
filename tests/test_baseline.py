import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

from demo import synthetic_data
from tabular_baseline import run_experiment

ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    def test_mixed_missing_data_and_report(self):
        frame = synthetic_data(rows=120)
        before = frame.copy(deep=True)
        result = run_experiment(frame, "outcome")
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(result.report["rows"], 120)
        self.assertEqual(result.report["features"], {"numeric": 5, "categorical": 1})
        self.assertEqual(result.report["holdout"]["decision_threshold"], 0.5)
        self.assertEqual(set(result.train_positions) & set(result.test_positions), set())
        self.assertEqual(len(result.train_positions) + len(result.test_positions), len(frame))
        json.dumps(result.report, allow_nan=False)

    def test_holdout_values_do_not_fit_preprocessing(self):
        frame = synthetic_data(rows=120)
        train, test = train_test_split(np.arange(len(frame)), test_size=0.25,
                                       stratify=frame.outcome, random_state=42)
        frame.iloc[test, frame.columns.get_loc("feature_0")] = 1e9
        frame.iloc[test, frame.columns.get_loc("channel")] = "holdout_only"
        result = run_experiment(frame, "outcome")
        prep = result.search.best_estimator_.named_steps["preprocess"]
        median = prep.named_transformers_["numeric"].named_steps["impute"].statistics_[0]
        self.assertAlmostEqual(median, np.nanmedian(frame.iloc[train].feature_0))
        self.assertLess(abs(prep.named_transformers_["numeric"].named_steps["scale"].mean_[0]), 100)
        categories = prep.named_transformers_["categorical"].named_steps["encode"].categories_[0]
        self.assertNotIn("holdout_only", categories)

    def test_every_imputer_fit_uses_only_fold_training_rows(self):
        frame = synthetic_data(rows=120)
        fitted_rows = []
        original_fit = SimpleImputer.fit

        def observed_fit(instance, data, *args, **kwargs):
            fitted_rows.append(set(data.index))
            return original_fit(instance, data, *args, **kwargs)

        with patch.object(SimpleImputer, "fit", new=observed_fit):
            result = run_experiment(frame, "outcome")
        training = set(result.train_positions)
        holdout = set(result.test_positions)
        self.assertTrue(fitted_rows)
        self.assertTrue(all(rows <= training and rows.isdisjoint(holdout) for rows in fitted_rows))
        self.assertTrue(any(len(rows) < len(training) for rows in fitted_rows))
        self.assertIn(training, fitted_rows)  # Final refit uses the complete training split.

    def test_holdout_changes_cannot_change_selection_or_fitted_model(self):
        frame = synthetic_data(rows=120)
        frame.index = [f"record-{i}" for i in range(len(frame))]
        original = run_experiment(frame, "outcome")
        altered = frame.copy(deep=True)
        altered.iloc[original.test_positions, altered.columns.get_loc("feature_0")] = 1e9
        altered.iloc[original.test_positions, altered.columns.get_loc("channel")] = "only_in_changed_holdout"
        changed = run_experiment(altered, "outcome")
        self.assertEqual(original.search.best_params_, changed.search.best_params_)
        self.assertEqual(original.search.best_score_, changed.search.best_score_)
        one = original.search.best_estimator_.named_steps["model"]
        two = changed.search.best_estimator_.named_steps["model"]
        np.testing.assert_array_equal(one.coef_, two.coef_)
        np.testing.assert_array_equal(one.intercept_, two.intercept_)
        np.testing.assert_array_equal(original.train_positions, changed.train_positions)

    def test_report_matches_actual_heldout_predictions(self):
        frame = synthetic_data(rows=120)
        result = run_experiment(frame, "outcome")
        heldout = frame.iloc[result.test_positions]
        p = result.search.predict_proba(heldout.drop(columns="outcome"))[:, 1]
        self.assertAlmostEqual(result.report["holdout"]["average_precision"],
                               average_precision_score(heldout.outcome, p))
        self.assertEqual(result.report["holdout"]["confusion_matrix"],
                         confusion_matrix(heldout.outcome, p >= 0.5, labels=[0, 1]).tolist())

    def test_same_seed_reproduces_split_and_report(self):
        frame = synthetic_data(rows=100)
        one, two = run_experiment(frame, "outcome"), run_experiment(frame, "outcome")
        np.testing.assert_array_equal(one.train_positions, two.train_positions)
        self.assertEqual(one.report, two.report)

    def test_categorical_only_and_unknown_category(self):
        frame = pd.DataFrame({"label": ["yes", "no"] * 30, "outcome": [1, 0] * 30})
        result = run_experiment(frame, "outcome")
        probabilities = result.search.predict_proba(pd.DataFrame({"label": ["unseen"]}))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(result.report["features"]["numeric"], 0)

    def test_numeric_only_and_all_missing_feature(self):
        frame = synthetic_data(rows=120).drop(columns="channel")
        frame["empty"] = np.nan
        result = run_experiment(frame, "outcome")
        self.assertEqual(result.report["features"]["numeric"], 6)
        self.assertEqual(result.report["features"]["categorical"], 0)

    def test_pandas_nullable_dtypes(self):
        frame = pd.DataFrame({"number": pd.Series([1, None, 3, 4] * 20, dtype="Int64"),
                              "group": pd.Series(["x", None, "y", "z"] * 20, dtype="string"),
                              "outcome": [0, 1, 0, 1] * 20})
        result = run_experiment(frame, "outcome")
        self.assertTrue(np.isfinite(result.report["holdout"]["average_precision"]))

    def test_target_excluded_and_input_index_does_not_change_split(self):
        frame = synthetic_data(rows=100)
        frame.index = ["repeated"] * len(frame)
        result = run_experiment(frame, "outcome")
        feature_names = result.search.best_estimator_.named_steps["preprocess"].feature_names_in_
        self.assertNotIn("outcome", feature_names)
        self.assertEqual(len(result.test_positions), 25)

    def test_invalid_targets_and_columns(self):
        good = synthetic_data(rows=50)
        variants = [pd.DataFrame(), good.drop(columns="outcome"), good[["outcome"]],
                    good.assign(outcome=0), good.assign(outcome=np.nan),
                    good.assign(outcome="positive"), good.assign(outcome=np.arange(50))]
        duplicate = good.copy()
        duplicate.columns = ["a"] * len(duplicate.columns)
        variants.append(duplicate)
        for frame in variants:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                run_experiment(frame, "outcome")

    def test_invalid_features(self):
        good = synthetic_data(rows=50)
        for value in (np.inf, complex(1, 2), pd.Timestamp("2026-01-01"), True):
            frame = good.assign(bad=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_experiment(frame, "outcome")

    def test_reserved_category_and_boolean_target_rejected(self):
        frame = synthetic_data(rows=50)
        with self.assertRaisesRegex(ValueError, "reserved"):
            run_experiment(frame.assign(channel="__missing__"), "outcome")
        with self.assertRaisesRegex(ValueError, "binary labels"):
            run_experiment(frame.assign(outcome=frame.outcome.astype(bool)), "outcome")

    def test_invalid_controls_and_small_classes(self):
        good = synthetic_data(rows=50)
        for controls in ({"seed": -1}, {"seed": True}, {"seed": 2**32}, {"folds": 1},
                         {"folds": 1000}, {"folds": 2.5}, {"test_size": 1}, {"test_size": np.nan},
                         {"test_size": True}, {"test_size": 0.999}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                run_experiment(good, "outcome", **controls)
        with self.assertRaises(ValueError):
            run_experiment(pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 0, 0, 1]}), "y")


class CliTests(unittest.TestCase):
    def test_real_csv_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            synthetic_data(rows=100).to_csv(path, index=False)
            result = subprocess.run([sys.executable, str(ROOT / "tabular_baseline.py"),
                                     "--csv", str(path), "--target", "outcome"],
                                    cwd=directory, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["rows"], 100)
            self.assertEqual(report["train_rows"] + report["holdout_rows"], 100)

    def test_cli_errors_do_not_echo_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            for content in ("x,y\nprivate_fixture_content,unknown\n", "x,x,y\n1,2,0\n", "", "x" * 150000 + ",y\n1,0\n"):
                path.write_text(content, encoding="utf-8")
                result = subprocess.run([sys.executable, str(ROOT / "tabular_baseline.py"),
                                         "--csv", str(path), "--target", "y"],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("error", json.loads(result.stderr))
                self.assertNotIn("private_fixture_content", result.stderr)


if __name__ == "__main__":
    unittest.main()
