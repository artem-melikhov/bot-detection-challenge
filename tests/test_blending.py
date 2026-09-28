"""Проверки финального смешивания оценок."""

import unittest

import numpy as np

from utils.blending import mean_percentile_rank


class MeanPercentileRankTest(unittest.TestCase):
    def test_monotonic_rescaling_does_not_change_result(self):
        first = np.array([0.9, 0.1, 0.5])
        second = np.array([0.2, 0.8, 0.4])
        expected = mean_percentile_rank([first, second])

        np.testing.assert_array_equal(
            mean_percentile_rank([first**3, second * 100]), expected
        )

    def test_equal_scores_share_rank(self):
        scores = mean_percentile_rank([
            np.array([0.2, 0.2, 0.8]),
            np.array([0.5, 0.5, 0.9]),
        ])

        np.testing.assert_allclose(scores, [0.5, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
