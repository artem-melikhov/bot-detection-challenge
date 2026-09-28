import numpy as np
import pandas as pd

from utils.peer_features import build_peer_features


def test_new_features_use_only_cookie_window_and_handle_zero_dt():
    train = pd.DataFrame({
        "cookie_id": ["a"],
        "window_start_ts": [pd.Timestamp("2026-04-06")],
        "window_end_ts": [pd.Timestamp("2026-04-07")],
    })
    test = pd.DataFrame({
        "cookie_id": ["b"],
        "window_start_ts": [pd.Timestamp("2026-04-06")],
        "window_end_ts": [pd.Timestamp("2026-04-07")],
    })
    events = pd.DataFrame({
        "cookie_id": ["a", "a", "a", "a"],
        "event_ts": pd.to_datetime([
            "2026-04-06 01:00", "2026-04-06 01:01",
            "2026-04-06 01:01", "2026-04-07 00:00",
        ]),
        "item_category": ["z", "a", None, "future"],
        "pointer_x": [0, 3, 6, 10000],
        "pointer_y": [0, 4, 8, 10000],
    })

    result = build_peer_features(train, test, events).set_index("cookie_id")
    assert result.loc["a", "top_category_cat"] == "a"
    assert result.loc["a", "night_activity_ratio"] == 1
    assert np.isclose(result.loc["a", "pointer_path_valid"], 5)
    assert np.isclose(result.loc["a", "pointer_speed_median"], 5 / 60)
    assert np.isclose(result.loc["a", "pointer_zero_dt_share"], .5)
    assert result.loc["a", "pointer_max_speed_raw"] > 1000
    assert result.loc["b", "top_category_cat"] == "unknown"
    assert result.loc["b", "pointer_total_dist_raw"] == 0
