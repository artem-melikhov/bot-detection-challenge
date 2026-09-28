"""Проверяет, помогают ли полные переходы и Word2Vec на будущих днях."""

from pathlib import Path

import numpy as np
import pandas as pd

from run_solution import load_model_data, make_model
from utils.blending import mean_percentile_rank
from utils.metric import precision_at_recall
from utils.sequence_embeddings import (
    EMBEDDING_COLUMNS, PAIR_COLUMNS, TRIGRAM_COLUMNS, embedding_features, event_sequences,
    pair_features, train_word2vec,
)
from utils.time_validation import walk_forward_splits


ROOT = Path(__file__).resolve().parent


def main() -> None:
    train, test, fit, predict, groups, y = load_model_data(ROOT)
    events = pd.read_csv(ROOT / "data/events.csv.gz", parse_dates=["event_ts"])
    meta, sequences = event_sequences(train, test, events)
    assert meta.cookie_id.equals(pd.concat([train.cookie_id, test.cookie_id], ignore_index=True))
    pairs = pair_features(sequences)
    fit = pd.concat([fit, pairs.iloc[:len(train)]], axis=1)
    predict = pd.concat([predict, pairs.iloc[len(train):].reset_index(drop=True)], axis=1)
    old_path = ROOT / "output/final_validation_scores.csv"
    old = pd.read_csv(old_path) if old_path.exists() else None
    parts = []
    for fold, fit_mask, valid_mask in walk_forward_splits(train):
        valid_ids = train.loc[valid_mask, "cookie_id"].to_numpy()
        if old is not None:
            previous = old.loc[old.fold.eq(fold)].set_index("cookie_id").reindex(valid_ids)
            if previous.rank_mean.isna().any() or not np.array_equal(previous.target, y[valid_mask]):
                raise ValueError(f"Сохранённая проверка не совпала с фолдом {fold}")
            baseline = previous.rank_mean.to_numpy()
        else:
            # Для чистого запуска заново считаем нынешний ансамбль.
            members = []
            for columns in groups.values():
                model = make_model()
                model.fit(fit.loc[fit_mask, columns], y[fit_mask])
                members.append(model.predict_proba(fit.loc[valid_mask, columns])[:, 1])
            baseline = mean_percentile_rank(members)

        # Словарь Word2Vec для фолда не видит события будущих дней.
        w2v = train_word2vec([sequences[i] for i in np.flatnonzero(fit_mask)])
        fold_embeddings = embedding_features(sequences[:len(train)], w2v)
        w2v_trigram = train_word2vec(
            [sequences[i] for i in np.flatnonzero(fit_mask)], length=3
        )
        fold_trigrams = embedding_features(
            sequences[:len(train)], w2v_trigram, length=3
        )
        fold_fit = pd.concat([fit, fold_embeddings, fold_trigrams], axis=1)
        scores = {"current_rank": baseline}
        context = groups["context_plus"]
        for name, extra in {
            "pairs": PAIR_COLUMNS,
            "word2vec": EMBEDDING_COLUMNS,
            "trigram_word2vec": TRIGRAM_COLUMNS,
            "pairs_word2vec": PAIR_COLUMNS + EMBEDDING_COLUMNS,
        }.items():
            model = make_model()
            model.fit(fold_fit.loc[fit_mask, context + extra], y[fit_mask])
            scores[name] = model.predict_proba(fold_fit.loc[valid_mask, context + extra])[:, 1]

        # Проверяем и исходный формат из трёх CatBoost, добавляя к каждой те же признаки.
        for name, extra in {"ensemble_pairs": PAIR_COLUMNS,
                            "ensemble_word2vec": EMBEDDING_COLUMNS,
                            "ensemble_trigram": TRIGRAM_COLUMNS}.items():
            member_scores = []
            for columns in groups.values():
                model = make_model()
                model.fit(fold_fit.loc[fit_mask, columns + extra], y[fit_mask])
                member_scores.append(
                    model.predict_proba(fold_fit.loc[valid_mask, columns + extra])[:, 1]
                )
            scores[name] = mean_percentile_rank(member_scores)
        part = pd.DataFrame({"cookie_id": valid_ids, "fold": fold,
                             "target": y[valid_mask], **scores})
        parts.append(part)
        print(f"{fold}: " + ", ".join(
            f"{name}={precision_at_recall(part.target, part[name]):.3f}"
            for name in scores
        ), flush=True)

    validation = pd.concat(parts, ignore_index=True)
    rows = []
    for name in validation.columns.difference(["cookie_id", "fold", "target"]):
        fold_scores = [precision_at_recall(part.target, part[name]) for part in parts]
        rows.append({
            "model": name,
            "fold_1": fold_scores[0], "fold_2": fold_scores[1],
            "fold_3": fold_scores[2],
            "mean_fold": float(np.mean(fold_scores)),
            "pooled": precision_at_recall(validation.target, validation[name]),
        })
    summary = pd.DataFrame(rows).sort_values("mean_fold", ascending=False)
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    validation.to_csv(output / "sequence_validation_scores.csv", index=False,
                      float_format="%.17g")
    summary.to_csv(output / "sequence_summary.csv", index=False,
                   float_format="%.17g")
    print(summary.to_string(index=False), flush=True)

    # Кандидаты отделены от действующего submission. Обучаем Word2Vec уже на всём train.
    w2v = train_word2vec(sequences[:len(train)])
    embeddings = embedding_features(sequences, w2v)
    w2v_trigram = train_word2vec(sequences[:len(train)], length=3)
    trigrams = embedding_features(sequences, w2v_trigram, length=3)
    fit = pd.concat([fit, embeddings.iloc[:len(train)], trigrams.iloc[:len(train)]], axis=1)
    predict = pd.concat([
        predict, embeddings.iloc[len(train):].reset_index(drop=True),
        trigrams.iloc[len(train):].reset_index(drop=True),
    ], axis=1)
    best = summary.loc[summary.model.ne("current_rank"), "model"].iloc[0]
    if best.startswith("ensemble_"):
        extra = {"ensemble_pairs": PAIR_COLUMNS,
                 "ensemble_word2vec": EMBEDDING_COLUMNS,
                 "ensemble_trigram": TRIGRAM_COLUMNS}[best]
        member_scores = []
        for columns in groups.values():
            model = make_model()
            model.fit(fit[columns + extra], y)
            member_scores.append(model.predict_proba(predict[columns + extra])[:, 1])
        test_score = mean_percentile_rank(member_scores)
    else:
        extra = {"pairs": PAIR_COLUMNS, "word2vec": EMBEDDING_COLUMNS,
                 "trigram_word2vec": TRIGRAM_COLUMNS,
                 "pairs_word2vec": PAIR_COLUMNS + EMBEDDING_COLUMNS}[best]
        model = make_model()
        model.fit(fit[groups["context_plus"] + extra], y)
        test_score = model.predict_proba(predict[groups["context_plus"] + extra])[:, 1]
    candidate = pd.DataFrame({"cookie_id": test.cookie_id, "score": test_score})
    assert candidate.cookie_id.is_unique and candidate.score.between(0, 1).all()
    candidate.to_csv(output / "sequence_candidate.csv", index=False, float_format="%.17g")
    print(f"Лучший новый вариант: {best}; кандидат сохранён отдельно.", flush=True)


if __name__ == "__main__":
    main()
