"""Дополнительные идеи из присланного скрипта, только внутри окна куки."""

from __future__ import annotations

import pandas as pd

from utils.features import events_in_windows


def _most_common(inside: pd.DataFrame, column: str, name: str) -> pd.Series:
    counts = (
        inside.dropna(subset=[column])
        .groupby(["cookie_id", column])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["cookie_id", "count", column], ascending=[True, False, True])
    )
    return counts.drop_duplicates("cookie_id").set_index("cookie_id")[column].rename(name)


def build_pasted_features(
    train: pd.DataFrame, test: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Воспроизводит группы признаков скрипта с явной обработкой пропусков."""
    inside, meta = events_in_windows(train, test, events)
    features = meta[["cookie_id"]].copy()
    for source, target in [
        ("item_location", "top_location_cat"),
        ("platform", "top_platform_cat"),
        ("seller_type", "top_seller_cat"),
    ]:
        features = features.join(_most_common(inside, source, target), on="cookie_id")
        features[target] = features[target].fillna("Unknown").astype(str)

    first_event = inside.groupby("cookie_id").event_ts.min()
    weekday = first_event.dt.dayofweek.rename("weekday")
    features = features.join(weekday, on="cookie_id")
    features["is_weekend"] = features.weekday.fillna(0).ge(5).astype(int)
    features["day_of_week_cat"] = features.weekday.fillna(0).astype(int).astype(str)
    features = features.drop(columns="weekday")

    pointer = inside.groupby("cookie_id").agg(
        pointer_x_std_script=("pointer_x", "std"),
        pointer_y_std_script=("pointer_y", "std"),
        pointer_x_zeros=("pointer_x", lambda s: s.eq(0).mean()),
    ).fillna(0)
    features = features.join(pointer, on="cookie_id")
    for column in pointer.columns:
        features[column] = features[column].fillna(0)

    queries = inside.dropna(subset=["search_query"])
    query_features = queries.groupby("cookie_id").agg(
        query_mean_len=("search_query", lambda s: s.str.len().mean()),
        query_has_digits=("search_query", lambda s: s.str.contains(r"\d").mean()),
    ).fillna(0)
    features = features.join(query_features, on="cookie_id")
    for column in query_features.columns:
        features[column] = features[column].fillna(0)
    return features
