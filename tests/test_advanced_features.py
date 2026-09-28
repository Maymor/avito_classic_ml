"""Focused regression checks for new features and chronological evaluation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bot_solution.advanced_features import build_advanced_features
from bot_solution.features import EVENT_COLUMNS, prepare_events
from bot_solution.models import blend_predictions, component_weights
from bot_solution.validation import make_folds
from test_solution import example_data


class AdvancedFeatureTests(unittest.TestCase):
    def test_shuffle_duplicates_and_post_window_events_do_not_change_features(self):
        meta, raw = example_data()
        events, _ = prepare_events(raw, meta)
        expected = build_advanced_features(events, meta)
        future = raw.iloc[[2]].copy()
        future.event_ts = pd.Timestamp("2030-01-01")
        changed = pd.concat([raw, raw.iloc[[1, 3]], future], ignore_index=True)
        changed = changed.sample(frac=1, random_state=123)
        actual, _ = prepare_events(changed, meta)
        pd.testing.assert_frame_equal(expected, build_advanced_features(actual, meta))

    def test_empty_cookie_is_preserved_and_identifiers_are_not_features(self):
        meta, raw = example_data()
        events, _ = prepare_events(raw, meta)
        expected = build_advanced_features(events, meta)
        self.assertEqual(expected.index.tolist(), ["a", "b"])
        self.assertEqual(expected.loc["b", "v2_pointer_observation_count"], 0)
        self.assertEqual(expected.loc["b", "v2_cadence_views_mean"], -1)
        self.assertTrue(np.isfinite(expected.to_numpy()).all())
        self.assertNotIn("cookie_id", expected.columns)
        self.assertNotIn("target", expected.columns)
        renamed_meta, renamed_raw = meta.copy(), raw.copy()
        mapping = {"a": "unrelated_cookie", "b": "another_cookie"}
        renamed_meta.cookie_id = renamed_meta.cookie_id.map(mapping)
        renamed_raw.cookie_id = renamed_raw.cookie_id.map(mapping)
        renamed_raw.item_id = renamed_raw.item_id * 100 + 12345
        renamed, _ = prepare_events(renamed_raw, renamed_meta)
        np.testing.assert_allclose(expected.to_numpy(), build_advanced_features(renamed, renamed_meta).to_numpy())

    def test_other_cookies_and_their_targets_cannot_affect_a_cookie(self):
        meta, raw = example_data()
        events, _ = prepare_events(raw, meta)
        reference = build_advanced_features(events, meta).loc[["a"]]
        only_meta = meta.iloc[[0]].copy()
        only_events, _ = prepare_events(raw, only_meta)
        alone = build_advanced_features(only_events, only_meta)
        pd.testing.assert_frame_equal(reference, alone)
        # Even accidentally supplied labels must never become model inputs.
        labeled_meta = meta.assign(target=[1, 0])
        changed_labels = meta.assign(target=[0, 1])
        pd.testing.assert_frame_equal(
            build_advanced_features(events, labeled_meta),
            build_advanced_features(events, changed_labels),
        )

    def test_cadence_funnel_pagination_sessions_and_pointer_geometry(self):
        meta, _ = example_data()
        rows = []
        for i, (seconds, event, item, query, page) in enumerate([
            (0, "search_results_view", None, "car", 1),
            (10, "item_view", 1, None, None),
            (20, "item_view", 2, None, None),
            (30, "item_view", 1, None, None),
            (40, "photo_swipe", 1, None, None),
            (50, "contact_phone_show", 1, None, None),
            (2000, "search_results_view", None, "car", 2),
            (2001, "search_results_view", None, "car", 3),
            (2010, "search_results_view", None, "boat", 1),
        ]):
            row = dict.fromkeys(EVENT_COLUMNS, np.nan)
            row.update(cookie_id="a", event_ts=pd.Timestamp("2026-04-06") + pd.Timedelta(seconds=seconds),
                       eid=100, event_name=event, platform="web", user_agent="Chrome/119.0",
                       item_id=item, item_category="cars" if item else None,
                       item_location="moscow" if item else None,
                       search_query=query, search_page=page, pointer_x=10 * (i + 1),
                       pointer_y=20 * (i + 1))
            rows.append(row)
        events, _ = prepare_events(pd.DataFrame(rows, columns=EVENT_COLUMNS), meta)
        f = build_advanced_features(events, meta).loc["a"]
        self.assertEqual(f.v2_cadence_views_mean, 10)
        self.assertEqual(f.v2_cadence_views_cv, 0)
        self.assertEqual(f.v2_cadence_views_mode_share, 1)
        self.assertAlmostEqual(f.v2_views_unique_item_fraction, 2 / 3)
        self.assertEqual(f.v2_viewed_items_with_contact_fraction, .5)
        self.assertEqual(f.v2_viewed_items_with_photo_fraction, .5)
        self.assertEqual(f.v2_contact_latency_mean, 40)
        self.assertAlmostEqual(f.v2_session_300_dominant_fraction, 6 / 9)
        self.assertEqual(f.v2_session_300_span_mean, 30)
        self.assertEqual(f.v2_same_query_page_next_fraction, 1)
        self.assertEqual(f.v2_same_query_page_back_fraction, 0)  # New-query reset is not a backward page.
        self.assertAlmostEqual(f.v2_search_query_switch_fraction, 1 / 3)
        self.assertEqual(f.v2_pointer_observation_count, 9)
        self.assertAlmostEqual(f.v2_pointer_xy_correlation, 1)
        self.assertAlmostEqual(f.v2_pointer_spread_anisotropy, .5)

    def test_same_timestamp_is_not_an_ordered_transition(self):
        meta, raw = example_data()
        events, _ = prepare_events(raw, meta)
        f = build_advanced_features(events, meta).loc["a"]
        self.assertEqual(f.v2_after_item_view_item_view_fraction, 0)
        self.assertEqual(f.v2_after_item_view_photo_swipe_fraction, 1)


class EvaluationAndEnsembleTests(unittest.TestCase):
    def test_expanding_folds_use_only_completed_past_windows(self):
        start = pd.date_range("2026-04-06", "2026-04-19", freq="D")
        train = pd.DataFrame({"window_start_ts": start, "window_end_ts": start + pd.Timedelta(days=1)})
        folds = make_folds(train)
        self.assertEqual([(int(f.sum()), int(v.sum())) for f, v in folds], [(5, 3), (8, 3), (11, 3)])
        for fit, valid in folds:
            self.assertFalse((fit & valid).any())
            self.assertLessEqual(train.loc[fit, "window_end_ts"].max(), train.loc[valid, "window_start_ts"].min())
        train.loc[0, "window_end_ts"] = pd.Timestamp("2026-04-12")
        with self.assertRaises(ValueError):
            make_folds(train)

    def test_weighted_predictions_and_invalid_recipes(self):
        weights = component_weights({"a": .25, "b": .75})
        actual = blend_predictions({"a": np.array([np.nan, .2, .6]),
                                    "b": np.array([np.nan, .8, .2])}, weights)
        np.testing.assert_allclose(actual, [np.nan, .65, .3], equal_nan=True)
        self.assertEqual(component_weights(["a", "b"]), {"a": .5, "b": .5})
        for invalid in ([], ["a", "a"], {}, {"a": -1, "b": 2}, {"a": .7}, {"a": np.nan}):
            with self.assertRaises(ValueError):
                component_weights(invalid)


if __name__ == "__main__":
    unittest.main()
