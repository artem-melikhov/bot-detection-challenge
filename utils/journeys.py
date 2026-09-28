"""Как кука переходит от объявления к фото, продавцу и контакту."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.features import events_in_windows


CONTACT_EVENTS = ("contact_phone_show", "contact_chat_open", "contact_message_sent")
ITEM_EVENTS = ("item_view", "photo_swipe", "favorite_add", "seller_page_view", *CONTACT_EVENTS)


def _count_by_cookie(mask: pd.Series) -> pd.Series:
    """Сколько пар кука-объявление удовлетворяют условию."""
    return mask.groupby(level=0).sum()


def _switch_share(views: pd.DataFrame, column: str) -> pd.Series:
    previous = views.groupby("cookie_id")[column].shift()
    comparable = views[column].notna() & previous.notna()
    changed = views.loc[comparable, column].ne(previous.loc[comparable])
    return changed.groupby(views.loc[comparable, "cookie_id"]).mean()


def _pro_share(events: pd.DataFrame) -> pd.Series:
    observed = events.seller_type.notna().groupby(events.cookie_id).sum()
    pro = events.seller_type.eq("pro").groupby(events.cookie_id).sum()
    return pro / observed.replace(0, np.nan)


def build_journey_features(
    train: pd.DataFrame, test: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Строит три группы признаков из событий внутри окна каждой куки."""
    inside, meta = events_in_windows(train, test, events)
    item_events = inside.loc[inside.item_id.notna()]
    pair_keys = ["cookie_id", "item_id"]
    pair_counts = pd.crosstab(
        [item_events.cookie_id, item_events.item_id], item_events.event_name
    ).reindex(columns=ITEM_EVENTS, fill_value=0)
    pair_counts.index.names = pair_keys
    view = pair_counts.item_view.gt(0)
    photo = pair_counts.photo_swipe.gt(0)
    favorite = pair_counts.favorite_add.gt(0)
    seller = pair_counts.seller_page_view.gt(0)
    contact_count = pair_counts[list(CONTACT_EVENTS)].sum(axis=1)
    contact = contact_count.gt(0)

    # Воронка: сколько разных объявлений прошли каждый шаг.
    funnel = pd.DataFrame(index=pair_counts.index.get_level_values(0).unique())
    funnel["viewed_item_count"] = _count_by_cookie(view)
    funnel["photo_item_count"] = _count_by_cookie(photo)
    funnel["favorite_item_count"] = _count_by_cookie(favorite)
    funnel["seller_item_count"] = _count_by_cookie(seller)
    funnel["contacted_item_count"] = _count_by_cookie(contact)
    funnel["photo_on_viewed_share"] = _count_by_cookie(photo & view) / funnel.viewed_item_count.replace(0, np.nan)
    funnel["favorite_on_viewed_share"] = _count_by_cookie(favorite & view) / funnel.viewed_item_count.replace(0, np.nan)
    funnel["contact_on_viewed_share"] = _count_by_cookie(contact & view) / funnel.viewed_item_count.replace(0, np.nan)
    funnel["contact_without_view_share"] = _count_by_cookie(contact & ~view) / funnel.contacted_item_count.replace(0, np.nan)
    funnel["contacts_per_contacted_item"] = contact_count.groupby(level=0).sum() / funnel.contacted_item_count.replace(0, np.nan)
    pair_size = pair_counts.sum(axis=1)
    funnel["max_actions_per_item"] = pair_size.groupby(level=0).max()
    funnel["multi_action_item_share"] = pair_size.gt(1).groupby(level=0).mean()

    # Время между первыми действиями с тем же объявлением.
    first_view = item_events.loc[item_events.event_name.eq("item_view")].groupby(pair_keys).event_ts.min()
    first_photo = item_events.loc[item_events.event_name.eq("photo_swipe")].groupby(pair_keys).event_ts.min()
    first_contact = item_events.loc[item_events.event_name.isin(CONTACT_EVENTS)].groupby(pair_keys).event_ts.min()
    view_photo_gap = (first_photo - first_view).dt.total_seconds()
    view_contact_gap = (first_contact - first_view).dt.total_seconds()
    seen_both_contact = first_view.notna() & first_contact.notna()
    seen_both_photo = first_view.notna() & first_photo.notna()

    timing = pd.DataFrame(index=funnel.index)
    timing["view_to_contact_median_sec"] = view_contact_gap.where(view_contact_gap.ge(0)).groupby(level=0).median()
    timing["view_to_photo_median_sec"] = view_photo_gap.where(view_photo_gap.ge(0)).groupby(level=0).median()
    timing["contact_within_5min_share"] = (
        _count_by_cookie(view_contact_gap.between(0, 300))
        / _count_by_cookie(seen_both_contact).replace(0, np.nan)
    )
    timing["photo_within_2min_share"] = (
        _count_by_cookie(view_photo_gap.between(0, 120))
        / _count_by_cookie(seen_both_photo).replace(0, np.nan)
    )
    timing["contact_before_view_share"] = (
        _count_by_cookie(view_contact_gap.lt(0))
        / _count_by_cookie(seen_both_contact).replace(0, np.nan)
    )

    views = inside.loc[inside.event_name.eq("item_view")].copy()
    view_gap = views.groupby("cookie_id").event_ts.diff().dt.total_seconds()
    view_gap_count = view_gap.notna().groupby(views.cookie_id).sum().replace(0, np.nan)
    timing["item_view_gap_median_sec"] = view_gap.groupby(views.cookie_id).median()
    timing["item_view_gap_under_10s_share"] = view_gap.le(10).groupby(views.cookie_id).sum() / view_gap_count
    timing["item_view_gap_cv"] = view_gap.groupby(views.cookie_id).std() / (view_gap.groupby(views.cookie_id).mean() + 1)
    timing["max_item_views_5min"] = (
        views.groupby(["cookie_id", views.event_ts.dt.floor("5min")]).size().groupby(level=0).max()
    )
    photo_events = inside.loc[inside.event_name.eq("photo_swipe")]
    photo_gap = photo_events.groupby(pair_keys).event_ts.diff().dt.total_seconds()
    timing["same_item_photo_gap_median_sec"] = photo_gap.groupby(photo_events.cookie_id).median()

    # Контекст: меняется ли тематика просмотров и совпадает ли она с последним поиском.
    context = pd.DataFrame(index=funnel.index)
    context["category_switch_share"] = _switch_share(views, "item_category")
    context["location_switch_share"] = _switch_share(views, "item_location")
    context["seller_switch_share"] = _switch_share(views, "seller_type")
    context["pro_view_share"] = _pro_share(views)
    context["pro_contact_share"] = _pro_share(inside.loc[inside.event_name.isin(CONTACT_EVENTS)])
    previous_item = views.groupby("cookie_id").item_id.shift()
    return_after_switch = views.duplicated(["cookie_id", "item_id"]) & views.item_id.ne(previous_item)
    context["return_to_item_share"] = return_after_switch.groupby(views.cookie_id).mean()

    search = inside.event_name.eq("search_results_view")
    last_search_time = inside.event_ts.where(search).groupby(inside.cookie_id).ffill()
    recent_search = (inside.event_ts - last_search_time).dt.total_seconds().between(0, 600)
    for column, name in [("item_category", "category"), ("item_location", "location")]:
        last_value = inside[column].where(search).groupby(inside.cookie_id).ffill()
        eligible = inside.event_name.eq("item_view") & recent_search & inside[column].notna() & last_value.notna()
        matches = inside.loc[eligible, column].eq(last_value.loc[eligible])
        context[f"search_view_{name}_match_share"] = matches.groupby(inside.loc[eligible, "cookie_id"]).mean()

    groups = {"funnel": funnel, "timing": timing, "context": context}
    features = meta[["cookie_id", "split"]].copy()
    for frame in groups.values():
        features = features.merge(frame.reset_index().rename(columns={"index": "cookie_id"}),
                                  on="cookie_id", how="left", validate="one_to_one")
    return features, {name: frame.columns.tolist() for name, frame in groups.items()}
