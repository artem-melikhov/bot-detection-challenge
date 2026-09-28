"""Проверяем, что обучение не использует будущие сутки."""

import unittest

import pandas as pd

from utils.time_validation import walk_forward_splits


class WalkForwardSplitsTest(unittest.TestCase):
    def test_validation_starts_after_all_training_windows_end(self):
        starts = pd.date_range("2026-04-06", "2026-04-19", freq="D")
        meta = pd.DataFrame({
            "window_start_ts": starts,
            "window_end_ts": starts + pd.Timedelta(days=1),
        })

        splits = list(walk_forward_splits(meta))

        self.assertEqual(len(splits), 3)
        self.assertEqual([int(valid.sum()) for _, _, valid in splits], [3, 3, 3])
        for _, fit, valid in splits:
            self.assertFalse((fit & valid).any())
            self.assertLessEqual(
                meta.loc[fit, "window_end_ts"].max(),
                meta.loc[valid, "window_start_ts"].min(),
            )


if __name__ == "__main__":
    unittest.main()
