"""Сравнивает Word2Vec по полной цепочке с простыми долями событий."""

from pathlib import Path

import numpy as np
import pandas as pd

from run_solution import load_model_data, make_model
from utils.blending import mean_percentile_rank
from utils.full_sequence import (
    MEAN_COLUMNS, POSITION_COLUMNS, RANDOM_COLUMNS, SHARE_COLUMNS,
    full_embedding_features, position_share_features, random_mean_features,
    train_full_word2vec,
)
from utils.metric import precision_at_recall
from utils.sequence_embeddings import event_sequences
from utils.time_validation import walk_forward_splits


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "full_mean": MEAN_COLUMNS,
    "full_position": MEAN_COLUMNS + POSITION_COLUMNS,
    "position_shares": SHARE_COLUMNS,
    "random_mean": RANDOM_COLUMNS,
}


def make_baseline(
    train: pd.DataFrame, fit: pd.DataFrame, groups: dict[str, list[str]],
    y: np.ndarray, fold: str, fit_mask: np.ndarray, valid_mask: np.ndarray,
    old: pd.DataFrame | None,
) -> np.ndarray:
    """Берёт сохранённый ответ или пересчитывает его на чистом запуске."""
    ids = train.loc[valid_mask, "cookie_id"].to_numpy()
    if old is not None:
        previous = old.loc[old.fold.eq(fold)].set_index("cookie_id").reindex(ids)
        if previous.rank_mean.isna().any() or not np.array_equal(previous.target, y[valid_mask]):
            raise ValueError(f"Сохранённая проверка не совпала с фолдом {fold}")
        return previous.rank_mean.to_numpy()
    scores = []
    for columns in groups.values():
        model = make_model()
        model.fit(fit.loc[fit_mask, columns], y[fit_mask])
        scores.append(model.predict_proba(fit.loc[valid_mask, columns])[:, 1])
    return mean_percentile_rank(scores)


def main() -> None:
    train, test, fit, predict, groups, y = load_model_data(ROOT)
    events = pd.read_csv(ROOT / "data/events.csv.gz", parse_dates=["event_ts"])
    meta, sequences = event_sequences(train, test, events)
    assert meta.cookie_id.equals(pd.concat([train.cookie_id, test.cookie_id], ignore_index=True))
    shares = position_share_features(sequences)
    random = random_mean_features(sequences)
    fit = pd.concat([fit, shares.iloc[:len(train)], random.iloc[:len(train)]], axis=1)
    predict = pd.concat([
        predict, shares.iloc[len(train):].reset_index(drop=True),
        random.iloc[len(train):].reset_index(drop=True),
    ], axis=1)
    old_path = ROOT / "output/final_validation_scores.csv"
    old = pd.read_csv(old_path) if old_path.exists() else None
    parts = []
    for fold, fit_mask, valid_mask in walk_forward_splits(train):
        baseline = make_baseline(train, fit, groups, y, fold, fit_mask, valid_mask, old)
        past_sequences = [sequences[i] for i in np.flatnonzero(fit_mask)]
        w2v = train_full_word2vec(past_sequences)
        embeddings = full_embedding_features(sequences[:len(train)], w2v)
        fold_fit = pd.concat([fit, embeddings], axis=1)
        scores = {"current_rank": baseline}

        # Одна модель показывает, что даёт группа признаков без смешивания ответов.
        for name, extra in VARIANTS.items():
            columns = groups["context_plus"] + extra
            model = make_model()
            model.fit(fold_fit.loc[fit_mask, columns], y[fit_mask])
            scores[name] = model.predict_proba(fold_fit.loc[valid_mask, columns])[:, 1]

        # Проверяем те же признаки и в формате нынешнего ответа из трёх моделей.
        for name, extra in VARIANTS.items():
            member_scores = []
            for columns in groups.values():
                model = make_model()
                model.fit(fold_fit.loc[fit_mask, columns + extra], y[fit_mask])
                member_scores.append(
                    model.predict_proba(fold_fit.loc[valid_mask, columns + extra])[:, 1]
                )
            scores[f"ensemble_{name}"] = mean_percentile_rank(member_scores)

        part = pd.DataFrame({
            "cookie_id": train.loc[valid_mask, "cookie_id"].to_numpy(),
            "fold": fold, "target": y[valid_mask], **scores,
        })
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
    validation.to_csv(output / "full_sequence_validation_scores.csv", index=False,
                      float_format="%.17g")
    summary.to_csv(output / "full_sequence_summary.csv", index=False,
                   float_format="%.17g")
    print(summary.to_string(index=False), flush=True)

    # Кандидат отдельный. Действующий submission.csv этот опыт не меняет.
    best = summary.loc[summary.model.ne("current_rank"), "model"].iloc[0]
    w2v = train_full_word2vec(sequences[:len(train)])
    embeddings = full_embedding_features(sequences, w2v)
    fit = pd.concat([fit, embeddings.iloc[:len(train)]], axis=1)
    predict = pd.concat([predict, embeddings.iloc[len(train):].reset_index(drop=True)], axis=1)
    extra = VARIANTS[best.removeprefix("ensemble_")]
    if best.startswith("ensemble_"):
        member_scores = []
        for columns in groups.values():
            model = make_model()
            model.fit(fit[columns + extra], y)
            member_scores.append(model.predict_proba(predict[columns + extra])[:, 1])
        test_score = mean_percentile_rank(member_scores)
    else:
        model = make_model()
        columns = groups["context_plus"] + extra
        model.fit(fit[columns], y)
        test_score = model.predict_proba(predict[columns])[:, 1]
    candidate = pd.DataFrame({"cookie_id": test.cookie_id, "score": test_score})
    assert candidate.cookie_id.is_unique and candidate.score.between(0, 1).all()
    candidate.to_csv(output / "full_sequence_candidate.csv", index=False,
                     float_format="%.17g")
    print(f"Лучший новый вариант: {best}; кандидат сохранён отдельно.", flush=True)


if __name__ == "__main__":
    main()
