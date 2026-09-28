"""Сравнивает новые признаки на тех же трёх временных фолдах."""

from pathlib import Path

import numpy as np
import pandas as pd

from run_solution import load_model_data, make_model
from utils.blending import mean_percentile_rank
from utils.metric import precision_at_recall
from utils.peer_features import (
    CATEGORY_COLUMNS, CLEAN_POINTER_COLUMNS, RAW_POINTER_COLUMNS, TIME_COLUMNS,
    build_peer_features,
)
from utils.time_validation import walk_forward_splits


ROOT = Path(__file__).resolve().parent


def main() -> None:
    train, test, fit, predict, model_columns, y = load_model_data(ROOT)
    events = pd.read_csv(ROOT / "data" / "events.csv.gz", parse_dates=["event_ts"])
    extra = build_peer_features(train, test, events)
    fit = fit.merge(extra.iloc[:len(train)], on="cookie_id", how="left", validate="one_to_one")
    predict = predict.merge(extra.iloc[len(train):], on="cookie_id", how="left", validate="one_to_one")
    assert fit.cookie_id.equals(train.cookie_id)
    assert predict.cookie_id.equals(test.cookie_id)

    context = model_columns["context_plus"]
    variants = {
        "category": context + CATEGORY_COLUMNS,
        "time": context + TIME_COLUMNS,
        "pointer_raw": context + RAW_POINTER_COLUMNS,
        "pointer_clean": context + CLEAN_POINTER_COLUMNS,
        "category_time": context + CATEGORY_COLUMNS + TIME_COLUMNS,
        "category_pointer": context + CATEGORY_COLUMNS + CLEAN_POINTER_COLUMNS,
        "all_clean": context + CATEGORY_COLUMNS + TIME_COLUMNS + CLEAN_POINTER_COLUMNS,
    }
    ensemble_variants = {
        "ensemble_category": CATEGORY_COLUMNS,
        "ensemble_time": TIME_COLUMNS,
        "ensemble_pointer_raw": RAW_POINTER_COLUMNS,
        "ensemble_pointer_clean": CLEAN_POINTER_COLUMNS,
        "ensemble_all_clean": CATEGORY_COLUMNS + TIME_COLUMNS + CLEAN_POINTER_COLUMNS,
    }
    old_path = ROOT / "output" / "final_validation_scores.csv"
    old = pd.read_csv(old_path) if old_path.exists() else None
    parts = []
    for fold_name, fit_mask, valid_mask in walk_forward_splits(train):
        ids = train.loc[valid_mask, "cookie_id"].to_numpy()
        if old is not None:
            previous = old.loc[old.fold.eq(fold_name)].set_index("cookie_id").reindex(ids)
            if previous.rank_mean.isna().any() or not np.array_equal(previous.target, y[valid_mask]):
                raise ValueError(f"Старая проверка не совпадает с новым фолдом: {fold_name}")
            scores = {"current_rank": previous.rank_mean.to_numpy(),
                      "current_context": previous.context_plus.to_numpy()}
        else:
            # Сохранённая таблица ускоряет сравнение, но для запуска не нужна.
            base_scores = {}
            for name, columns in model_columns.items():
                model = make_model()
                model.fit(fit.loc[fit_mask, columns], y[fit_mask])
                base_scores[name] = model.predict_proba(fit.loc[valid_mask, columns])[:, 1]
            scores = {
                "current_rank": mean_percentile_rank(list(base_scores.values())),
                "current_context": base_scores["context_plus"],
            }
        for name, columns in variants.items():
            model = make_model(("top_category_cat",) if "top_category_cat" in columns else ())
            model.fit(fit.loc[fit_mask, columns], y[fit_mask])
            scores[name] = model.predict_proba(fit.loc[valid_mask, columns])[:, 1]

        for ensemble_name, extra_columns in ensemble_variants.items():
            ensemble_scores = []
            for columns in model_columns.values():
                new_columns = columns + extra_columns
                categorical = ("top_category_cat",) if "top_category_cat" in extra_columns else ()
                model = make_model(categorical)
                model.fit(fit.loc[fit_mask, new_columns], y[fit_mask])
                ensemble_scores.append(model.predict_proba(fit.loc[valid_mask, new_columns])[:, 1])
            scores[ensemble_name] = mean_percentile_rank(ensemble_scores)

        part = pd.DataFrame({"cookie_id": ids, "fold": fold_name,
                             "target": y[valid_mask], **scores})
        parts.append(part)
        print(f"{fold_name}: " + ", ".join(
            f"{name} {precision_at_recall(part.target, part[name]):.3f}"
            for name in ("current_rank", "all_clean", "ensemble_all_clean")
        ), flush=True)

    validation = pd.concat(parts, ignore_index=True)
    rows = []
    for name in validation.columns.difference(["cookie_id", "fold", "target"]):
        by_fold = [precision_at_recall(p.target, p[name]) for p in parts]
        rows.append({
            "model": name,
            **{fold_name: score for (fold_name, _, _), score in zip(walk_forward_splits(train), by_fold)},
            "mean_fold": float(np.mean(by_fold)),
            "pooled": precision_at_recall(validation.target, validation[name]),
        })
    summary = pd.DataFrame(rows).sort_values("mean_fold", ascending=False)
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    validation.to_csv(output / "peer_feature_validation_scores.csv", index=False,
                      float_format="%.17g")
    summary.to_csv(output / "peer_feature_summary.csv", index=False,
                   float_format="%.17g")
    print("\nСреднее по трём фолдам и единая метрика на всех OOF-куках:")
    print(summary[["model", "mean_fold", "pooled"]].to_string(index=False), flush=True)

    # Отдельный кандидат. Основной submission.csv эксперимент не трогает.
    best_name = summary.loc[summary.model.ne("current_rank") &
                            summary.model.ne("current_context"), "model"].iloc[0]
    if best_name in ensemble_variants:
        extra_columns = ensemble_variants[best_name]
        test_scores = []
        for columns in model_columns.values():
            categorical = ("top_category_cat",) if "top_category_cat" in extra_columns else ()
            model = make_model(categorical)
            model.fit(fit[columns + extra_columns], y)
            test_scores.append(model.predict_proba(predict[columns + extra_columns])[:, 1])
        test_score = mean_percentile_rank(test_scores)
    else:
        columns = variants[best_name]
        model = make_model(("top_category_cat",) if "top_category_cat" in columns else ())
        model.fit(fit[columns], y)
        test_score = model.predict_proba(predict[columns])[:, 1]
    answer = pd.DataFrame({"cookie_id": test.cookie_id, "score": test_score})
    assert answer.cookie_id.equals(test.cookie_id)
    assert answer.cookie_id.is_unique and answer.score.between(0, 1).all()
    answer.to_csv(output / "peer_best_candidate.csv", index=False, float_format="%.17g")
    print(f"Кандидат: {best_name}, файл {output / 'peer_best_candidate.csv'}")


if __name__ == "__main__":
    main()
