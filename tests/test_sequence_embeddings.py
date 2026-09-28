import numpy as np
import pandas as pd

from utils.sequence_embeddings import (
    EMBEDDING_COLUMNS, TRIGRAM_COLUMNS, chain_tokens, embedding_features,
    event_sequences, pair_features, train_word2vec,
)


def test_order_changes_chain_features():
    first = ["item_view", "photo_swipe", "contact_phone_show"]
    second = ["contact_phone_show", "photo_swipe", "item_view"]
    assert chain_tokens(first) != chain_tokens(second)
    assert chain_tokens(first, 3) != chain_tokens(second, 3)
    features = pair_features([first, second])
    assert not np.array_equal(features.iloc[0].to_numpy(), features.iloc[1].to_numpy())
    assert np.allclose(features.sum(axis=1), 1)


def test_only_events_inside_own_cookie_window():
    train = pd.DataFrame({
        "cookie_id": ["a"], "window_start_ts": [pd.Timestamp("2026-04-06")],
        "window_end_ts": [pd.Timestamp("2026-04-07")],
    })
    test = pd.DataFrame({
        "cookie_id": ["b"], "window_start_ts": [pd.Timestamp("2026-04-07")],
        "window_end_ts": [pd.Timestamp("2026-04-08")],
    })
    events = pd.DataFrame({
        "cookie_id": ["a", "a", "a", "b"],
        "event_ts": pd.to_datetime([
            "2026-04-06 02:00", "2026-04-06 01:00", "2026-04-07 00:00",
            "2026-04-07 01:00",
        ]),
        "event_name": ["photo_swipe", "item_view", "login", "item_view"],
    })
    _, sequences = event_sequences(train, test, events)
    assert sequences == [["item_view", "photo_swipe"], ["item_view"]]


def test_word2vec_does_not_learn_from_validation_sequences():
    train = [["item_view", "photo_swipe", "item_view"]] * 4
    model = train_word2vec(train)
    assert "item_view>photo_swipe" in model.wv
    assert "login>contact_phone_show" not in model.wv
    features = embedding_features([train[0], ["login", "contact_phone_show"]], model)
    assert features.columns.tolist() == EMBEDDING_COLUMNS
    assert features.chain_w2v_coverage.tolist() == [1, 0]
    assert np.isfinite(features.to_numpy()).all()


def test_trigram_vectors_keep_three_step_direction():
    sequences = [["item_view", "photo_swipe", "contact_phone_show"]] * 4
    model = train_word2vec(sequences, length=3)
    features = embedding_features(sequences, model, length=3)
    assert features.columns.tolist() == TRIGRAM_COLUMNS
    assert features.trigram_w2v_coverage.tolist() == [1] * 4
