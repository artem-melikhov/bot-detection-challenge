"""Объединение оценок моделей по их месту среди тестовых кук."""

import numpy as np
import pandas as pd


def mean_percentile_rank(scores: list[np.ndarray]) -> np.ndarray:
    """Средний процентиль трёх моделей; равные оценки получают одинаковый ранг."""
    if not scores:
        raise ValueError("Нужна хотя бы одна модель")

    values = np.column_stack(scores)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Оценки должны быть конечными числами одной длины")

    # rank(pct=True) ставит самую высокую оценку ближе к 1.
    return pd.DataFrame(values).rank(pct=True).mean(axis=1).to_numpy()
