import pandas as pd

from utils.stacking import meta_features, time_blocks


def test_meta_ranks_keep_ties_together():
    scores = pd.DataFrame({
        "base_plus": [0.2, 0.2, 0.9],
        "routes_plus": [0.1, 0.7, 0.3],
        "context_plus": [0.4, 0.6, 0.5],
    })
    result = meta_features(scores)
    assert result.base_plus_rank.iloc[0] == result.base_plus_rank.iloc[1]
    assert result.base_plus_rank.iloc[2] == 1


def test_blocks_never_look_into_the_future():
    days = pd.date_range("2026-04-06", "2026-04-19")
    meta = pd.DataFrame({
        "window_start_ts": days,
        "window_end_ts": days + pd.Timedelta(days=1),
    })
    blocks = list(time_blocks(meta))
    assert len(blocks) == 4
    for _, fit, valid in blocks:
        assert meta.loc[fit, "window_end_ts"].max() <= meta.loc[valid, "window_start_ts"].min()
