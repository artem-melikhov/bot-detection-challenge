"""Признаки по событиям внутри суточного окна каждой куки."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _concentration(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    """Доля самого частого значения признака у каждой куки."""
    counts = frame.groupby(["cookie_id", column], dropna=True).size()
    totals = counts.groupby(level=0).sum()
    return (counts.groupby(level=0).max() / totals).rename(name)


def _entropy(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    """Энтропия распределения, делённая на максимум при данном числе значений."""
    counts = frame.groupby(["cookie_id", column], dropna=True).size()
    totals = counts.groupby(level=0).transform("sum")
    shares = counts / totals
    entropy = (-(shares * np.log(shares))).groupby(level=0).sum()
    varieties = counts.groupby(level=0).size()
    return (entropy / np.log(varieties.where(varieties.gt(1)))).fillna(0).rename(name)


def build_features(
    train: pd.DataFrame, test: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]], int]:
    """Собирает старые и новые признаки без событий за концом окна.

    Возвращает одну таблицу в порядке train, затем test, группы колонок и число
    событий внутри окон. Метки target в функцию не попадают.
    """
    meta = pd.concat(
        [train.assign(split="train"), test.assign(split="test")], ignore_index=True
    )
    if not meta.cookie_id.is_unique:
        raise ValueError("cookie_id должен быть уникальным в train и test")

    original_event_cols = events.columns.tolist()
    joined = events.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]],
        on="cookie_id", how="left", validate="many_to_one",
    )
    if joined.window_start_ts.isna().any():
        raise ValueError("У части событий не нашлось суточного окна")
    inside = joined.loc[
        joined.event_ts.ge(joined.window_start_ts)
        & joined.event_ts.lt(joined.window_end_ts)
    ].copy()
    inside = inside.sort_values(["cookie_id", "event_ts"], kind="mergesort")
    by_cookie = inside.groupby("cookie_id", sort=False)
    inside["gap_sec"] = by_cookie.event_ts.diff().dt.total_seconds()
    inside["minute"] = inside.event_ts.dt.floor("min")
    inside["hour"] = inside.event_ts.dt.hour
    inside["has_pointer"] = inside.pointer_x.notna() & inside.pointer_y.notna()
    inside["headless_ua"] = inside.user_agent.str.contains("HeadlessChrome", case=False, na=False)
    inside["automation_ua"] = inside.user_agent.str.contains(
        "bot|curl|python|selenium|playwright", case=False, regex=True, na=False
    )
    inside["platform_group"] = inside.platform.str.lower().replace(
        {"desktop": "web", "iphone": "ios"}
    )
    inside["same_item_again"] = inside.item_id.notna() & inside.item_id.eq(
        by_cookie.item_id.shift()
    )
    inside["exact_duplicate"] = inside.duplicated(subset=original_event_cols, keep=False)

    by_cookie = inside.groupby("cookie_id")
    aggregates = by_cookie.agg(
        events=("eid", "size"), items=("item_id", "nunique"),
        categories=("item_category", "nunique"), locations=("item_location", "nunique"),
        queries=("search_query", "nunique"), ua_count=("user_agent", "nunique"),
        platform_count=("platform_group", "nunique"), active_hours=("hour", "nunique"),
        first_event=("event_ts", "min"), last_event=("event_ts", "max"),
        median_gap_sec=("gap_sec", "median"),
        gap_p10_sec=("gap_sec", lambda s: s.quantile(.10)),
        gap_p90_sec=("gap_sec", lambda s: s.quantile(.90)),
        gap_under_5s=("gap_sec", lambda s: s.le(5).mean()),
        gap_under_30s=("gap_sec", lambda s: s.le(30).mean()),
        gap_zero=("gap_sec", lambda s: s.eq(0).mean()),
        pointer_share=("has_pointer", "mean"), pointer_x_unique=("pointer_x", "nunique"),
        headless_ua=("headless_ua", "max"), automation_ua=("automation_ua", "max"),
        same_item_share=("same_item_again", "mean"),
        duplicate_share=("exact_duplicate", "mean"),
        search_page_max=("search_page", "max"), search_page_mean=("search_page", "mean"),
    )
    aggregates["span_minutes"] = (
        aggregates.last_event - aggregates.first_event
    ).dt.total_seconds() / 60
    aggregates = aggregates.drop(columns=["first_event", "last_event"])
    aggregates["max_events_minute"] = (
        inside.groupby(["cookie_id", "minute"]).size().groupby(level=0).max()
    )
    category_counts = inside.groupby(["cookie_id", "item_category"]).size()
    category_total = by_cookie.item_category.count()
    aggregates["top_category_share"] = (
        category_counts.groupby(level=0).max() / category_total
    ).fillna(0)

    event_counts = pd.crosstab(inside.cookie_id, inside.event_name).add_prefix("n_")
    aggregates = aggregates.join(event_counts)
    for column in event_counts.columns:
        aggregates[column.replace("n_", "share_", 1)] = (
            aggregates[column] / aggregates.events
        )
    platform_counts = pd.crosstab(inside.cookie_id, inside.platform_group)
    platform_counts = platform_counts.reindex(columns=["web", "android", "ios"], fill_value=0)
    for platform in platform_counts.columns:
        aggregates[f"share_platform_{platform}"] = platform_counts[platform] / aggregates.events
    aggregates["items_per_event"] = aggregates["items"] / aggregates.events
    aggregates["queries_per_search"] = (
        aggregates.queries / (aggregates.n_search_results_view + 1)
    )
    aggregates["item_views_per_item"] = (
        aggregates.n_item_view / (aggregates["items"] + 1)
    )
    base_columns = aggregates.columns.tolist() + ["cookie_age_days"]

    # Ритм: насколько действия сбиты в короткие серии и есть ли длинные перерывы.
    rhythm = pd.DataFrame(index=aggregates.index)
    rhythm["active_minutes"] = inside.groupby("cookie_id").minute.nunique()
    rhythm["events_per_active_minute"] = aggregates.events / rhythm.active_minutes
    rhythm["peak_minute_share"] = aggregates.max_events_minute / aggregates.events
    rhythm["max_events_5min"] = (
        inside.groupby(["cookie_id", inside.event_ts.dt.floor("5min")])
        .size().groupby(level=0).max()
    )
    rhythm["gap_mean_sec"] = by_cookie.gap_sec.mean()
    rhythm["gap_std_sec"] = by_cookie.gap_sec.std()
    rhythm["gap_cv"] = rhythm.gap_std_sec / (rhythm.gap_mean_sec + 1)
    rhythm["gap_max_sec"] = by_cookie.gap_sec.max()
    rhythm["gap_over_10min_share"] = inside.gap_sec.gt(600).groupby(inside.cookie_id).mean()
    new_session = inside.gap_sec.isna() | inside.gap_sec.gt(1800)
    rhythm["sessions_30min"] = new_session.groupby(inside.cookie_id).sum()
    burst_id = (inside.gap_sec.isna() | inside.gap_sec.gt(30)).groupby(
        inside.cookie_id
    ).cumsum()
    rhythm["max_burst_30s"] = (
        inside.groupby(["cookie_id", burst_id]).size().groupby(level=0).max()
    )

    # Маршруты: смотрим соседние шаги, без предположений о будущем событии.
    routes = pd.DataFrame(index=aggregates.index)
    following = inside.groupby("cookie_id").event_name.shift(-1)
    current = inside.event_name
    contact = following.isin(["contact_phone_show", "contact_chat_open", "contact_message_sent"])
    transitions = {
        "search_to_item_rate": ((current.eq("search_results_view") & following.eq("item_view")), "n_search_results_view"),
        "item_to_photo_rate": ((current.eq("item_view") & following.eq("photo_swipe")), "n_item_view"),
        "item_to_seller_rate": ((current.eq("item_view") & following.eq("seller_page_view")), "n_item_view"),
        "item_to_contact_rate": ((current.eq("item_view") & contact), "n_item_view"),
    }
    for name, (mask, denominator) in transitions.items():
        routes[name] = mask.groupby(inside.cookie_id).sum() / (aggregates[denominator] + 1)
    routes["same_event_next_share"] = (
        current.eq(following).groupby(inside.cookie_id).mean()
    )
    last_item_view = inside.event_ts.where(current.eq("item_view")).groupby(
        inside.cookie_id
    ).ffill()
    contact_now = current.isin(["contact_phone_show", "contact_chat_open", "contact_message_sent"])
    recent_view = (inside.event_ts - last_item_view).dt.total_seconds().le(300)
    routes["contact_after_view_5min_share"] = (
        (contact_now & recent_view).groupby(inside.cookie_id).sum()
        / (contact_now.groupby(inside.cookie_id).sum() + 1)
    )
    run_id = current.ne(inside.groupby("cookie_id").event_name.shift()).groupby(
        inside.cookie_id
    ).cumsum()
    routes["max_item_view_run"] = (
        inside.loc[current.eq("item_view")].groupby(
            [inside.cookie_id[current.eq("item_view")], run_id[current.eq("item_view")]]
        ).size().groupby(level=0).max()
    ).reindex(routes.index, fill_value=0)

    # Разнообразие: множество разных объявлений и равномерность просмотров.
    breadth = pd.DataFrame(index=aggregates.index)
    item_views = inside.loc[current.eq("item_view")]
    item_view_counts = item_views.groupby("cookie_id").item_id.nunique()
    breadth["item_view_unique"] = item_view_counts
    breadth["item_view_repeat_share"] = (
        1 - item_view_counts / aggregates.n_item_view.replace(0, np.nan)
    ).fillna(0)
    breadth["top_item_share"] = _concentration(item_views, "item_id", "top_item_share")
    for column, label in [
        ("item_category", "category"),
        ("item_location", "location"),
        ("search_query", "query"),
    ]:
        breadth[f"{label}_entropy"] = _entropy(inside, column, f"{label}_entropy")
        if label != "category":  # этот показатель уже был в базовом наборе
            breadth[f"top_{label}_share"] = _concentration(
                inside, column, f"top_{label}_share"
            )
    breadth["categories_per_item"] = aggregates.categories / (aggregates["items"] + 1)

    new_groups = {"rhythm": rhythm, "routes": routes, "breadth": breadth}
    for frame in new_groups.values():
        aggregates = aggregates.join(frame)
    features = meta[["cookie_id", "split", "window_start_ts", "cookie_created_at"]].merge(
        aggregates.reset_index(), on="cookie_id", how="left", validate="one_to_one"
    )
    features["cookie_age_days"] = (
        features.window_start_ts - features.cookie_created_at
    ).dt.total_seconds() / 86400
    groups = {"base": base_columns}
    groups.update({name: frame.columns.tolist() for name, frame in new_groups.items()})
    return features, groups, len(inside)
