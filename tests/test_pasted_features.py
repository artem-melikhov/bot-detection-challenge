"""Дополнительные признаки не видят события за суточным окном."""

import unittest

import pandas as pd

from utils.pasted_features import build_pasted_features


class PastedFeaturesTest(unittest.TestCase):
    def test_future_event_does_not_change_features(self):
        start = pd.Timestamp("2026-04-10")
        end = start + pd.Timedelta(days=1)
        meta = pd.DataFrame({
            "cookie_id": ["train_cookie", "test_cookie"],
            "window_start_ts": [start, start],
            "window_end_ts": [end, end],
        })
        train = meta.iloc[:1].assign(target=1)
        test = meta.iloc[1:]
        inside = pd.DataFrame([
            {"cookie_id": "train_cookie", "event_ts": start, "item_location": "moskva",
             "platform": "desktop", "seller_type": "private", "pointer_x": 0,
             "pointer_y": 5, "search_query": "iphone 14"},
            {"cookie_id": "test_cookie", "event_ts": start, "item_location": None,
             "platform": None, "seller_type": None, "pointer_x": None,
             "pointer_y": None, "search_query": None},
        ])
        future = pd.DataFrame([
            {"cookie_id": "train_cookie", "event_ts": end, "item_location": "spb",
             "platform": "android", "seller_type": "pro", "pointer_x": 100,
             "pointer_y": 100, "search_query": "no digits"},
        ])

        expected = build_pasted_features(train, test, inside)
        actual = build_pasted_features(train, test, pd.concat([inside, future]))

        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual.loc[0, "pointer_x_zeros"], 1.0)
        self.assertEqual(actual.loc[0, "query_has_digits"], 1.0)
        self.assertEqual(actual.loc[1, "top_location_cat"], "Unknown")


if __name__ == "__main__":
    unittest.main()
