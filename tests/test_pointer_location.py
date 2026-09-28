"""Проверяем границы окна и смысл новых признаков."""

import unittest

import pandas as pd

from utils.pointer_location import build_pointer_location_features


class PointerLocationFeaturesTest(unittest.TestCase):
    def test_window_zeros_missing_coordinates_and_location_tie(self):
        start = pd.Timestamp("2026-04-10")
        end = start + pd.Timedelta(days=1)
        meta = pd.DataFrame({
            "cookie_id": ["train_cookie", "test_cookie"],
            "cookie_created_at": [start, start],
            "window_start_ts": [start, start],
            "window_end_ts": [end, end],
        })
        train = meta.iloc[:1].assign(target=1)
        test = meta.iloc[1:]
        events = pd.DataFrame([
            {"cookie_id": "train_cookie", "event_ts": start,
             "pointer_x": 0, "pointer_y": 10, "item_location": "moskva"},
            {"cookie_id": "train_cookie", "event_ts": start + pd.Timedelta(minutes=1),
             "pointer_x": 10, "pointer_y": 0, "item_location": "spb"},
            {"cookie_id": "train_cookie", "event_ts": start + pd.Timedelta(minutes=2),
             "pointer_x": None, "pointer_y": None, "item_location": None},
            {"cookie_id": "train_cookie", "event_ts": end,
             "pointer_x": 0, "pointer_y": 0, "item_location": "spb"},
            {"cookie_id": "test_cookie", "event_ts": start,
             "pointer_x": None, "pointer_y": None, "item_location": None},
        ])

        actual = build_pointer_location_features(train, test, events).set_index("cookie_id")
        self.assertEqual(actual.loc["train_cookie", "counter_zeros"], 2)
        self.assertEqual(actual.loc["train_cookie", "pointer_zero_any"], 2)
        self.assertEqual(actual.loc["train_cookie", "pointer_zero_both"], 0)
        self.assertAlmostEqual(actual.loc["train_cookie", "pointer_x_std"], 7.0710678119)
        self.assertEqual(actual.loc["train_cookie", "top_location"], "moskva")
        self.assertTrue(pd.isna(actual.loc["test_cookie", "pointer_x_std"]))
        self.assertEqual(actual.loc["test_cookie", "counter_zeros"], 0)
        self.assertEqual(actual.loc["test_cookie", "top_location"], "unknown")


if __name__ == "__main__":
    unittest.main()
