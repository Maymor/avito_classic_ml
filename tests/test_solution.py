"""Checks for the consequential risks: time leakage, order, ties and coverage."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from metric import precision_at_recall
from bot_solution.features import EVENT_COLUMNS, build_features, parse_user_agent, prepare_events
from bot_solution.evaluation import operating_point
from bot_solution.pipeline import validate_submission


def example_data():
    meta = pd.DataFrame({
        "cookie_id": ["a", "b"],
        "cookie_created_at": pd.to_datetime(["2026-04-01", "2026-04-02"]),
        "window_start_ts": pd.to_datetime(["2026-04-06", "2026-04-06"]),
        "window_end_ts": pd.to_datetime(["2026-04-07", "2026-04-07"]),
    })
    rows = []
    for ts, name, item, x, page in (
        ("2026-04-05 23:59:59", "login", None, None, None),
        ("2026-04-06 00:00:00", "search_results_view", None, None, 1),
        ("2026-04-06 00:00:01", "item_view", 1, 10, None),
        ("2026-04-06 00:00:01", "item_view", 2, 20, None),
        ("2026-04-06 00:00:02", "photo_swipe", 1, 15, None),
        ("2026-04-06 23:59:59", "captcha_shown", None, None, None),
        ("2026-04-07 00:00:00", "contact_phone_show", 1, 50, None),
    ):
        row = dict.fromkeys(EVENT_COLUMNS, np.nan)
        row.update({
            "cookie_id": "a", "event_ts": pd.Timestamp(ts), "eid": 100,
            "event_name": name, "platform": "WEB", "user_agent": "Mozilla/5.0 Chrome/119.0",
            "item_id": item, "item_category": "  Cars  ", "item_location": "Moscow",
            "seller_type": "private", "pointer_x": x, "pointer_y": x,
            "search_query": "  Big   CAR ", "search_page": page,
        })
        rows.append(row)
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return meta, events


class FeatureTests(unittest.TestCase):
    def test_half_open_window_and_exact_duplicates(self):
        meta, events = example_data()
        events = pd.concat([events, events.iloc[[1]]], ignore_index=True)
        clean, audit = prepare_events(events, meta)
        self.assertEqual(len(clean), 5)
        self.assertEqual(audit["events_before_window"], 1)
        self.assertEqual(audit["events_at_or_after_window_end"], 1)
        self.assertEqual(audit["in_window_exact_duplicates"], 1)
        self.assertTrue(clean.event_ts.ge(clean.window_start_ts).all())
        self.assertTrue(clean.event_ts.lt(clean.window_end_ts).all())
        self.assertEqual(set(clean.platform), {"web"})
        self.assertEqual(set(clean.search_query), {"big car"})

    def test_shuffle_duplicate_and_future_event_invariance(self):
        meta, events = example_data()
        clean, _ = prepare_events(events, meta)
        expected = build_features(clean, meta)
        future = events.iloc[[2]].copy()
        future.event_ts = pd.Timestamp("2030-01-01")
        changed = pd.concat([events, events.iloc[[1, 3]], future], ignore_index=True)
        changed = changed.sample(frac=1, random_state=123)
        actual_events, _ = prepare_events(changed, meta)
        pd.testing.assert_frame_equal(expected, build_features(actual_events, meta))

    def test_no_events_cookie_and_id_not_in_features(self):
        meta, events = example_data()
        clean, _ = prepare_events(events, meta)
        features = build_features(clean, meta)
        self.assertEqual(features.index.tolist(), ["a", "b"])
        self.assertEqual(features.loc["b", "n_events"], 0)
        self.assertEqual(features.loc["b", "has_events"], 0)
        self.assertEqual(features.loc["b", "gap_mean"], -1)
        self.assertEqual(features.loc["a", "n_events"], 5)
        self.assertNotIn("cookie_id", features.columns)
        self.assertNotIn("target", features.columns)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertTrue(all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes))
        mapping = {"a": "new_unrelated_id", "b": "other_unrelated_id"}
        renamed_meta, renamed_events = meta.copy(), events.copy()
        renamed_meta.cookie_id = renamed_meta.cookie_id.map(mapping)
        renamed_events.cookie_id = renamed_events.cookie_id.map(mapping)
        renamed_clean, _ = prepare_events(renamed_events, renamed_meta)
        actual = build_features(renamed_clean, renamed_meta)
        np.testing.assert_allclose(features.to_numpy(), actual.to_numpy())

    def test_user_agent_versions_are_not_categories(self):
        self.assertEqual(parse_user_agent("Chrome/119.0 Windows"), parse_user_agent("Chrome/130.0 Windows"))
        self.assertTrue(parse_user_agent("HeadlessChrome/120.0 Linux")[2])

    def test_unknown_cookie_is_not_silently_dropped(self):
        meta, events = example_data()
        events.loc[0, "cookie_id"] = "unknown"
        with self.assertRaises(ValueError):
            prepare_events(events, meta)


class MetricAndSubmissionTests(unittest.TestCase):
    def test_threshold_matches_official_metric_including_ties(self):
        for y, scores in (
            ([1, 0, 1, 1, 1], [1, .8, .6, .4, .2]),
            ([0, 1, 0, 1], [.5, .5, .5, .5]),
        ):
            point = operating_point(np.array(y), np.array(scores))
            self.assertEqual(point["precision"], precision_at_recall(y, scores))
            self.assertGreaterEqual(point["recall"], .7)
        rng = np.random.default_rng(12)
        for _ in range(30):
            y = np.r_[np.ones(10, dtype=int), np.zeros(50, dtype=int)]
            scores = rng.integers(0, 10, len(y)) / 10
            thresholds = np.unique(scores)
            expected = max(
                y[scores >= t].mean() for t in thresholds
                if y[scores >= t].sum() / y.sum() >= .7
            )
            self.assertAlmostEqual(precision_at_recall(y, scores), expected)

    def test_submission_covers_exactly_test_in_order(self):
        test = pd.DataFrame({"cookie_id": ["x", "y"]})
        submission = pd.DataFrame({"cookie_id": ["x", "y"], "score": [.1, .9]})
        validate_submission(submission, test)
        for invalid in (
            submission.iloc[[1, 0]], submission.iloc[[0]],
            submission.assign(score=[np.nan, .9]),
            submission.assign(score=[-0.1, .9]),
            submission.assign(cookie_id=["x", "x"]),
            submission.assign(extra=1),
        ):
            with self.assertRaises(ValueError):
                validate_submission(invalid, test)


if __name__ == "__main__":
    unittest.main()
