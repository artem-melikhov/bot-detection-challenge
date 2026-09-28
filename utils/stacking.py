"""Временные блоки и признаки для бустинга поверх трёх моделей."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Первый блок нужен, чтобы верхней модели хватило обучающих примеров.
# Для него базовые модели видят лишь 6–8 апреля.
OOF_BLOCKS = (
    ("9–10 апреля", "2026-04-09", "2026-04-11"),
    ("11–13 апреля", "2026-04-11", "2026-04-14"),
    ("14–16 апреля", "2026-04-14", "2026-04-17"),
    ("17–19 апреля", "2026-04-17", "2026-04-20"),
)

MODEL_NAMES = ("base_plus", "routes_plus", "context_plus")


def time_blocks(meta: pd.DataFrame):
    """Обучающая часть заканчивается до начала проверяемого блока."""
    for name, start_text, end_text in OOF_BLOCKS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        fit = meta.window_end_ts.le(start).to_numpy()
        valid = meta.window_start_ts.ge(start).to_numpy() & meta.window_start_ts.lt(end).to_numpy()
        if not fit.any() or not valid.any():
            raise ValueError(f"Пустой временной блок: {name}")
        if meta.loc[fit, "window_end_ts"].max() > meta.loc[valid, "window_start_ts"].min():
            raise ValueError(f"Временные окна пересеклись: {name}")
        yield name, fit, valid


def meta_features(scores: pd.DataFrame) -> pd.DataFrame:
    """Три исходных score и их места внутри одного будущего блока."""
    raw = scores.loc[:, MODEL_NAMES].reset_index(drop=True).copy()
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("У базовых моделей есть пропущенные или бесконечные score")
    ranks = raw.rank(pct=True).add_suffix("_rank")
    return pd.concat([raw, ranks], axis=1)
