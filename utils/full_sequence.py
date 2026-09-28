"""Word2Vec по полной истории событий и признаки начала, середины, конца."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from utils.sequence_embeddings import EVENT_NAMES


VECTOR_SIZE = 8
PARTS = ("start", "middle", "end")
MEAN_COLUMNS = [f"whole_w2v_mean_{i:02d}" for i in range(VECTOR_SIZE)]
RANDOM_COLUMNS = [f"whole_random_mean_{i:02d}" for i in range(VECTOR_SIZE)]
POSITION_COLUMNS = [
    f"whole_w2v_{part}_{i:02d}" for part in PARTS for i in range(VECTOR_SIZE)
]
SHARE_COLUMNS = [
    f"whole_share_{part}_{event}" for part in PARTS for event in EVENT_NAMES
]


def split_sequence(sequence: Sequence[str]) -> tuple[Sequence[str], Sequence[str], Sequence[str]]:
    """Делит всю историю на три соседних участка, не меняя порядок."""
    indices = np.array_split(np.arange(len(sequence)), 3)
    return tuple([sequence[i] for i in chunk] for chunk in indices)


def train_full_word2vec(sequences: Sequence[Sequence[str]]) -> Word2Vec:
    """Каждая кука идёт в Word2Vec как отдельное предложение целиком."""
    sentences = [list(sequence) for sequence in sequences if sequence]
    if not sentences:
        raise ValueError("Нет событий для обучения Word2Vec")
    return Word2Vec(
        sentences=sentences, vector_size=VECTOR_SIZE, window=5,
        min_count=1, sg=1, negative=5, epochs=20,
        workers=1, seed=42,
    )


def full_embedding_features(
    sequences: Sequence[Sequence[str]], model: Word2Vec
) -> pd.DataFrame:
    """Один средний вектор и три вектора по участкам той же цепочки."""
    columns = MEAN_COLUMNS + POSITION_COLUMNS
    values = np.zeros((len(sequences), len(columns)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        vectors = [model.wv[event] for event in sequence if event in model.wv]
        if vectors:
            values[row, :VECTOR_SIZE] = np.mean(vectors, axis=0)
        for part_number, part in enumerate(split_sequence(sequence)):
            part_vectors = [model.wv[event] for event in part if event in model.wv]
            if part_vectors:
                start = VECTOR_SIZE * (part_number + 1)
                values[row, start:start + VECTOR_SIZE] = np.mean(part_vectors, axis=0)
    return pd.DataFrame(values, columns=columns)


def position_share_features(sequences: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Честный контроль: доли типов событий в тех же трёх участках."""
    vocabulary = {event: i for i, event in enumerate(EVENT_NAMES)}
    values = np.zeros((len(sequences), len(SHARE_COLUMNS)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for part_number, part in enumerate(split_sequence(sequence)):
            if not part:
                continue
            for event in part:
                index = vocabulary.get(event)
                if index is not None:
                    values[row, part_number * len(EVENT_NAMES) + index] += 1
            start = part_number * len(EVENT_NAMES)
            values[row, start:start + len(EVENT_NAMES)] /= len(part)
    return pd.DataFrame(values, columns=SHARE_COLUMNS)


def random_mean_features(sequences: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Контроль: те же средние, но векторы событий не обучались."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((len(EVENT_NAMES), VECTOR_SIZE)).astype(np.float32)
    vocabulary = {event: i for i, event in enumerate(EVENT_NAMES)}
    values = np.zeros((len(sequences), VECTOR_SIZE), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        known = [vocabulary[event] for event in sequence if event in vocabulary]
        if known:
            values[row] = vectors[known].mean(axis=0)
    return pd.DataFrame(values, columns=RANDOM_COLUMNS)
