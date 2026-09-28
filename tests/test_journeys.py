"""Проверяем связь действий с объявлением и границы окна."""

import unittest

import pandas as pd

from utils.journeys import build_journey_features


class JourneyFeaturesTest(unittest.TestCase):
    def test_contacts_are_linked_only_within_the_cookie_window(self):
        start = pd.Timestamp("2026-04-10 00:00:00")
        end = start + pd.Timedelta(days=1)
        meta = pd.DataFrame({
            "cookie_id": ["train_cookie", "test_cookie"],
            "cookie_created_at": [start, start],
            "window_start_ts": [start, start],
            "window_end_ts": [end, end],
        })
        train = meta.iloc[:1].assign(target=1)
        test = meta.iloc[1:]

        def event(cookie_id, item_id, when, name):
            return {
                "cookie_id": cookie_id, "item_id": item_id,
                "event_ts": when, "event_name": name,
                "item_category": "phones", "item_location": "moscow",
                "seller_type": "private",
            }

        inside = pd.DataFrame([
            event("train_cookie", 1, start + pd.Timedelta(minutes=1), "item_view"),
            event("train_cookie", 1, start + pd.Timedelta(minutes=2), "photo_swipe"),
            event("train_cookie", 1, start + pd.Timedelta(minutes=3), "contact_phone_show"),
            event("train_cookie", 2, start + pd.Timedelta(minutes=4), "contact_chat_open"),
            event("test_cookie", 3, start + pd.Timedelta(minutes=2), "item_view"),
        ])
        all_events = pd.concat([
            inside,
            pd.DataFrame([
                event("train_cookie", 3, end, "contact_phone_show"),
                event("test_cookie", 3, start - pd.Timedelta(seconds=1), "contact_chat_open"),
            ]),
        ], ignore_index=True)

        expected, expected_groups = build_journey_features(train, test, inside)
        actual, groups = build_journey_features(train, test, all_events)

        self.assertEqual(groups, expected_groups)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual.viewed_item_count.tolist(), [1, 1])
        self.assertEqual(actual.contacted_item_count.iloc[0], 2)
        self.assertEqual(actual.contact_without_view_share.iloc[0], .5)
        self.assertEqual(actual.view_to_contact_median_sec.iloc[0], 120)


if __name__ == "__main__":
    unittest.main()
