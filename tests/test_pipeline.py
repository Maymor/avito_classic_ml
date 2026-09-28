"""Финальная конфигурация и ограничения, которые легко нарушить при упаковке."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bot_solution.advanced_features import build_advanced_features
from bot_solution.evaluation import metrics
from bot_solution.features import prepare_events
from bot_solution.models import MODEL_SPECS, FINAL_WEIGHTS, columns_for, make_model, blend_predictions
from bot_solution.pipeline import checksum, matrix_fingerprint, reference_metadata, save_submission
from test_solution import example_data


class FinalRecipeTests(unittest.TestCase):
    def test_feature_sets_and_fixed_recipe(self):
        meta, raw = example_data()
        clean, _ = prepare_events(raw, meta)
        features = build_advanced_features(clean, meta)
        self.assertEqual(features.shape[1], 472)
        self.assertEqual(len(MODEL_SPECS), 4)
        self.assertEqual(list(FINAL_WEIGHTS.values()), [0.25] * 4)
        for spec in MODEL_SPECS.values():
            self.assertEqual(spec["seed"], 42)
            self.assertEqual(len(columns_for(spec, features)), 250 if spec["features"] == "original" else 472)

    def test_model_factories_run_on_tiny_numeric_data(self):
        meta, raw = example_data()
        clean, _ = prepare_events(raw, meta)
        features = build_advanced_features(clean, meta)
        for name in ("catboost_depth5", "lightgbm_15"):
            spec = {**MODEL_SPECS[name], "iterations": 2}
            model = make_model(spec, threads=1)
            columns = columns_for(spec, features)
            model.fit(features[columns], np.array([0, 1]))
            score = model.predict_proba(features[columns])[:, 1]
            self.assertTrue(np.isfinite(score).all())
            self.assertTrue(((score >= 0) & (score <= 1)).all())

    def test_blend_is_row_independent_and_rejects_mismatched_shapes(self):
        scores = {name: np.array([0.2, 0.8]) for name in MODEL_SPECS}
        expected = blend_predictions(scores, FINAL_WEIGHTS)
        scores["catboost_depth5"][0] = 0.9
        self.assertEqual(blend_predictions(scores, FINAL_WEIGHTS)[1], expected[1])
        scores["catboost_depth6"] = np.array([0.5])
        with self.assertRaises(ValueError):
            blend_predictions(scores, FINAL_WEIGHTS)

    def test_csv_format_and_feature_fingerprint(self):
        meta, raw = example_data()
        clean, _ = prepare_events(raw, meta)
        features = build_advanced_features(clean, meta).reset_index(drop=True)
        before = matrix_fingerprint(features)
        changed = features.copy()
        changed.loc[0, "n_events"] += 1
        self.assertNotEqual(before, matrix_fingerprint(changed))
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one.csv", Path(directory) / "two.csv"
            save_submission(meta, np.array([0.12345678901234567, 0.9]), first)
            save_submission(meta, np.array([0.12345678901234567, 0.9]), second)
            self.assertEqual(checksum(first), checksum(second))
            self.assertEqual(first.read_text().splitlines()[0], "cookie_id,score")

    def test_reference_is_audit_metadata_not_cookie_answers(self):
        reference = reference_metadata()
        self.assertEqual(reference["n_test_cookies"], 4909)
        self.assertEqual(reference["n_features"], 472)
        self.assertEqual(reference["threads"], 8)
        for value in reference["input_sha256"].values():
            self.assertEqual(len(value), 64)
        self.assertEqual(len(reference["submission_sha256"]), 64)
        self.assertNotIn("cookie_id", json.dumps(reference))

    def test_metrics_reject_missing_class_and_invalid_scores(self):
        for y, score in (([0, 0], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([0, 1], [0.1])):
            with self.assertRaises(ValueError):
                metrics(np.array(y), np.array(score))


if __name__ == "__main__":
    unittest.main()
