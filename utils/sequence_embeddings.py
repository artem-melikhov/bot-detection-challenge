"""Порядок действий куки: частоты переходов и Word2Vec по коротким цепочкам."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from utils.features import events_in_windows


EVENT_NAMES = (
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "favorite_add", "contact_phone_show", "contact_chat_open",
    "contact_message_sent", "login",
)
PAIR_COLUMNS = [f"pair_{left}_{right}" for left in EVENT_NAMES for right in EVENT_NAMES]
EMBEDDING_COLUMNS = [f"chain_w2v_{i:02d}" for i in range(16)] + ["chain_w2v_coverage"]
TRIGRAM_COLUMNS = [f"trigram_w2v_{i:02d}" for i in range(16)] + ["trigram_w2v_coverage"]


def event_sequences(
    train: pd.DataFrame, test: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Возвращает последовательности внутри окна в порядке train, затем test."""
    inside, meta = events_in_windows(train, test, events)
    grouped = inside.groupby("cookie_id", sort=False).event_name.agg(list)
    sequences = [grouped.get(cookie_id, []) for cookie_id in meta.cookie_id]
    return meta, sequences


def chain_tokens(sequence: Sequence[str], length: int = 2) -> list[str]:
    """Имена соседних пар или троек. Направление перехода сохраняется."""
    if length not in (2, 3):
        raise ValueError("Поддерживаются только пары и тройки")
    return [">".join(sequence[i:i + length]) for i in range(len(sequence) - length + 1)]


def pair_features(sequences: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Доля каждого перехода от всех переходов данной куки."""
    vocabulary = {
        f"{left}>{right}": i * len(EVENT_NAMES) + j
        for i, left in enumerate(EVENT_NAMES)
        for j, right in enumerate(EVENT_NAMES)
    }
    values = np.zeros((len(sequences), len(PAIR_COLUMNS)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        pairs = chain_tokens(sequence)
        for pair in pairs:
            index = vocabulary.get(pair)
            if index is not None:
                values[row, index] += 1
        if pairs:
            values[row] /= len(pairs)
    return pd.DataFrame(values, columns=PAIR_COLUMNS)


def train_word2vec(sequences: Sequence[Sequence[str]], length: int = 2) -> Word2Vec | None:
    """Учится без меток и только на разрешённых куках обучающей части."""
    sentences = [chain_tokens(sequence, length) for sequence in sequences]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return None
    return Word2Vec(
        sentences=sentences, vector_size=16, window=3, min_count=1,
        sg=1, negative=5, epochs=20, workers=1, seed=42,
    )


def embedding_features(
    sequences: Sequence[Sequence[str]], model: Word2Vec | None, length: int = 2
) -> pd.DataFrame:
    """Средний вектор переходов и доля переходов из словаря обучения."""
    columns = EMBEDDING_COLUMNS if length == 2 else TRIGRAM_COLUMNS
    values = np.zeros((len(sequences), len(columns)), dtype=np.float32)
    if model is None:
        return pd.DataFrame(values, columns=columns)
    for row, sequence in enumerate(sequences):
        chains = chain_tokens(sequence, length)
        vectors = [model.wv[chain] for chain in chains if chain in model.wv]
        if vectors:
            values[row, :-1] = np.mean(vectors, axis=0)
            values[row, -1] = len(vectors) / len(chains)
    return pd.DataFrame(values, columns=columns)
