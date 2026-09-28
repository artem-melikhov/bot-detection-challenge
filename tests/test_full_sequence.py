import numpy as np

from utils.full_sequence import (
    MEAN_COLUMNS, POSITION_COLUMNS, RANDOM_COLUMNS, SHARE_COLUMNS,
    full_embedding_features, position_share_features, random_mean_features,
    split_sequence,
    train_full_word2vec,
)


def test_segments_use_every_event_in_order():
    sequence = ["search_results_view", "item_view", "photo_swipe",
                "seller_page_view", "contact_phone_show"]
    pieces = split_sequence(sequence)
    assert [event for piece in pieces for event in piece] == sequence
    assert pieces[0] != pieces[-1]


def test_reversed_route_has_same_total_but_different_position_features():
    forward = ["search_results_view", "item_view", "contact_phone_show"]
    backward = list(reversed(forward))
    model = train_full_word2vec([forward] * 10)
    features = full_embedding_features([forward, backward], model)
    assert features.columns.tolist() == MEAN_COLUMNS + POSITION_COLUMNS
    assert np.allclose(features.loc[0, MEAN_COLUMNS], features.loc[1, MEAN_COLUMNS])
    assert not np.allclose(
        features.loc[0, POSITION_COLUMNS], features.loc[1, POSITION_COLUMNS]
    )
    shares = position_share_features([forward, backward])
    assert shares.columns.tolist() == SHARE_COLUMNS
    assert not np.array_equal(shares.iloc[0].to_numpy(), shares.iloc[1].to_numpy())


def test_unseen_event_does_not_enter_training_vocabulary():
    model = train_full_word2vec([["item_view", "photo_swipe"]] * 10)
    assert "login" not in model.wv
    features = full_embedding_features([["login"]], model)
    assert np.allclose(features.to_numpy(), 0)


def test_random_control_is_fixed_and_loses_order_like_mean_word2vec():
    forward = ["item_view", "photo_swipe", "contact_phone_show"]
    backward = list(reversed(forward))
    first = random_mean_features([forward, backward])
    second = random_mean_features([forward, backward])
    assert first.columns.tolist() == RANDOM_COLUMNS
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    assert np.allclose(first.iloc[0], first.iloc[1])
