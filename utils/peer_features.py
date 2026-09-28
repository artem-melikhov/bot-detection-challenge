"""Идеи второго решения: категория, время активности и путь курсора.

Все расчёты идут только по событиям внутри суточного окна куки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.features import events_in_windows


CATEGORY_COLUMNS = ["top_category_cat"]
TIME_COLUMNS = ["night_activity_ratio", "peak_hour_sin", "peak_hour_cos"]
RAW_POINTER_COLUMNS = ["pointer_total_dist_raw", "pointer_mean_speed_raw", "pointer_max_speed_raw"]
CLEAN_POINTER_COLUMNS = [
    "pointer_path_valid", "pointer_step_median", "pointer_step_p90",
    "pointer_speed_median", "pointer_speed_p90", "pointer_stationary_share",
    "pointer_zero_dt_share", "pointer_long_gap_share",
]


def build_peer_features(
    train: pd.DataFrame, test: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Одна строка на куку в порядке train, затем test."""
    inside, meta = events_in_windows(train, test, events)
    result = meta[["cookie_id"]].copy()

    # Самая частая категория. При ничьей выбираем по алфавиту, чтобы результат
    # не зависел от порядка строк в исходном файле.
    counts = (
        inside.dropna(subset=["item_category"])
        .groupby(["cookie_id", "item_category"])
        .size().rename("n").reset_index()
        .sort_values(["cookie_id", "n", "item_category"], ascending=[True, False, True])
    )
    category = counts.drop_duplicates("cookie_id").set_index("cookie_id").item_category
    result["top_category_cat"] = result.cookie_id.map(category).fillna("unknown").astype(str)

    # Используем часы из event_ts как есть. Это время сервера, а не обязательно
    # местное время человека, поэтому "ночь" здесь лишь условное название.
    hours = inside.event_ts.dt.hour
    night = hours.lt(6).groupby(inside.cookie_id).mean()
    hour_counts = (
        inside.assign(hour=hours).groupby(["cookie_id", "hour"])
        .size().rename("n").reset_index()
        .sort_values(["cookie_id", "n", "hour"], ascending=[True, False, True])
    )
    peak = hour_counts.drop_duplicates("cookie_id").set_index("cookie_id").hour
    result["night_activity_ratio"] = result.cookie_id.map(night)
    peak_hour = result.cookie_id.map(peak)
    result["peak_hour_sin"] = np.sin(2 * np.pi * peak_hour / 24)
    result["peak_hour_cos"] = np.cos(2 * np.pi * peak_hour / 24)

    # Берём только события с обеими координатами. После этого соседние точки
    # могут быть далеко друг от друга по времени, что отдельно учитываем ниже.
    pointer = inside.dropna(subset=["pointer_x", "pointer_y"]).copy()
    by_cookie = pointer.groupby("cookie_id", sort=False)
    pointer["dt"] = by_cookie.event_ts.diff().dt.total_seconds()
    pointer["dx"] = by_cookie.pointer_x.diff()
    pointer["dy"] = by_cookie.pointer_y.diff()
    pointer["dist"] = np.hypot(pointer.dx, pointer.dy)
    pointer["raw_speed"] = pointer.dist / (pointer.dt + .001)

    # Повторяем три признака из чужого скрипта как контрольный вариант.
    raw = pointer.groupby("cookie_id").agg(
        pointer_total_dist_raw=("dist", "sum"),
        pointer_mean_speed_raw=("raw_speed", "mean"),
        pointer_max_speed_raw=("raw_speed", "max"),
    )
    # Для скорости оставляем соседние точки с положительным интервалом не
    # больше получаса. Нулевые интервалы и длинные паузы считаем отдельно.
    valid_pair = pointer.dt.gt(0) & pointer.dt.le(1800)
    clean = pointer.loc[valid_pair].copy()
    clean["speed"] = clean.dist / clean.dt
    clean_group = clean.groupby("cookie_id")
    robust = clean_group.agg(
        pointer_path_valid=("dist", "sum"),
        pointer_step_median=("dist", "median"),
        pointer_step_p90=("dist", lambda x: x.quantile(.9)),
        pointer_speed_median=("speed", "median"),
        pointer_speed_p90=("speed", lambda x: x.quantile(.9)),
        pointer_stationary_share=("dist", lambda x: x.eq(0).mean()),
    ).reindex(pointer.cookie_id.drop_duplicates())
    pair_count = pointer.dt.notna().groupby(pointer.cookie_id).sum().replace(0, np.nan)
    robust["pointer_zero_dt_share"] = (
        pointer.dt.eq(0).groupby(pointer.cookie_id).sum() / pair_count
    )
    robust["pointer_long_gap_share"] = (
        pointer.dt.gt(1800).groupby(pointer.cookie_id).sum() / pair_count
    )

    result = result.join(raw, on="cookie_id").join(robust, on="cookie_id")
    result[RAW_POINTER_COLUMNS] = result[RAW_POINTER_COLUMNS].fillna(0)
    return result
