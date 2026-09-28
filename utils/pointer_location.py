"""Координаты курсора и самый частый город объявлений за сутки куки."""

from __future__ import annotations

import pandas as pd

from utils.features import events_in_windows


def build_pointer_location_features(
    train: pd.DataFrame, test: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Считает признаки только по событиям внутри окна каждой куки."""
    inside, meta = events_in_windows(train, test, events)

    pointer = inside.loc[inside.pointer_x.notna() & inside.pointer_y.notna()].copy()
    pointer["zero_x"] = pointer.pointer_x.eq(0).astype(int)
    pointer["zero_y"] = pointer.pointer_y.eq(0).astype(int)
    pointer["zero_any"] = pointer.zero_x.add(pointer.zero_y).gt(0).astype(int)
    pointer["zero_both"] = pointer.zero_x.mul(pointer.zero_y)
    pointer["zero_coordinates"] = pointer.zero_x.add(pointer.zero_y)
    pointer_features = pointer.groupby("cookie_id").agg(
        pointer_x_std=("pointer_x", "std"),
        pointer_y_std=("pointer_y", "std"),
        counter_zeros=("zero_coordinates", "sum"),
        pointer_zero_any=("zero_any", "sum"),
        pointer_zero_both=("zero_both", "sum"),
    )

    # При равной частоте берём город по алфавиту. Так результат не зависит от
    # порядка строк в events.csv.gz.
    location_counts = (
        inside.dropna(subset=["item_location"])
        .groupby(["cookie_id", "item_location"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["cookie_id", "count", "item_location"],
            ascending=[True, False, True],
        )
    )
    top_location = (
        location_counts.drop_duplicates("cookie_id")
        .set_index("cookie_id")["item_location"]
        .rename("top_location")
    )

    features = meta[["cookie_id"]].join(pointer_features, on="cookie_id")
    features = features.join(top_location, on="cookie_id")
    for column in ["counter_zeros", "pointer_zero_any", "pointer_zero_both"]:
        features[column] = features[column].fillna(0).astype(int)
    features["top_location"] = features.top_location.fillna("unknown").astype(str)
    return features
