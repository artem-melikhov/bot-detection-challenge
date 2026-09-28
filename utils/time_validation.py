"""Проверка на будущих днях без разделения одних суток между частями."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


FOLDS = (
    ("11–13 апреля", "2026-04-11", "2026-04-14"),
    ("14–16 апреля", "2026-04-14", "2026-04-17"),
    ("17–19 апреля", "2026-04-17", "2026-04-20"),
)


def walk_forward_splits(
    meta: pd.DataFrame,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Для каждого отрезка обучаемся лишь на уже закончившихся окнах."""
    for name, start_text, end_text in FOLDS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        fit = meta.window_end_ts.le(start).to_numpy()
        valid = meta.window_start_ts.ge(start).to_numpy() & meta.window_start_ts.lt(end).to_numpy()
        if not fit.any() or not valid.any():
            raise ValueError(f"Пустая обучающая или проверочная часть: {name}")
        if meta.loc[fit, "window_end_ts"].max() > meta.loc[valid, "window_start_ts"].min():
            raise ValueError(f"Временные окна пересеклись: {name}")
        yield name, fit, valid
