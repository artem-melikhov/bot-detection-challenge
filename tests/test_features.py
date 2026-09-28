"""Проверяем границы окна на маленьком наборе событий."""

import unittest

import pandas as pd

from utils.features import build_features


class FeatureWindowTest(unittest.TestCase):
    def test_events_outside_cookie_window_do_not_change_features(self):
        start = pd.Timestamp("2026-04-10 00:00:00")
        end = start + pd.Timedelta(days=1)
        meta = pd.DataFrame({
            "cookie_id": ["train_cookie", "test_cookie"],
            "cookie_created_at": [start, start],
            "window_start_ts": [start, start],
            "window_end_ts": [end, end],
        })
        train = meta.iloc[:1].assign(target=0)
        test = meta.iloc[1:]

        def event(cookie_id, timestamp, name, eid):
            return {
                "cookie_id": cookie_id, "event_ts": timestamp, "eid": eid,
                "event_name": name, "platform": "desktop", "user_agent": "browser",
                "item_id": "item_1", "item_category": "category_1",
                "item_location": "location_1", "seller_type": "private",
                "search_query": "query_1", "search_page": 1,
                "pointer_x": 100, "pointer_y": 200,
            }

        inside = pd.DataFrame([
            event("train_cookie", start, "item_view", 1),
            event("test_cookie", start + pd.Timedelta(hours=1), "search_results_view", 2),
        ])
        all_events = pd.concat([
            inside,
            pd.DataFrame([
                event("train_cookie", end, "contact_phone_show", 3),
                event("test_cookie", start - pd.Timedelta(seconds=1), "photo_swipe", 4),
            ]),
        ], ignore_index=True)

        expected, expected_groups, _ = build_features(train, test, inside)
        actual, groups, n_inside = build_features(train, test, all_events)

        self.assertEqual(n_inside, 2)
        self.assertEqual(groups, expected_groups)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual.events.tolist(), [1, 1])
        self.assertNotIn("n_contact_phone_show", groups["base"])


if __name__ == "__main__":
    unittest.main()
