"""Optuna, честные предсказания по времени и CatBoost поверх трёх CatBoost.

Запуск с --tune заново ищет параметры и сохраняет их в stacking_params.json.
Обычный запуск берёт эти параметры и повторяет проверку и файл с ответом.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier

from run_solution import RANDOM_STATE, load_model_data
from utils.blending import mean_percentile_rank
from utils.metric import precision_at_recall
from utils.stacking import MODEL_NAMES, meta_features, time_blocks


ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "stacking_params.json"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_base(params: dict) -> CatBoostClassifier:
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        cat_features=["top_location"],
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )


def make_meta(params: dict) -> CatBoostClassifier:
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )


def score_block(model, fit, predict, y, fit_mask, valid_mask, columns):
    model.fit(fit.loc[fit_mask, columns], y[fit_mask])
    return model.predict_proba(predict.loc[valid_mask, columns])[:, 1]


def tune_base(fit, y, blocks, columns_by_model, trials: int) -> dict:
    """Подбор только на 9–13 апреля. Поздние дни остаются за рамками поиска."""
    chosen = {}
    for name in MODEL_NAMES:
        columns = columns_by_model[name]

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 300, 900, step=100),
                "depth": trial.suggest_int("depth", 4, 7),
                "learning_rate": trial.suggest_float("learning_rate", .025, .11, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2, 18, log=True),
            }
            scores = []
            for _, fit_mask, valid_mask in blocks[:2]:
                pred = score_block(
                    make_base(params), fit, fit, y, fit_mask, valid_mask, columns
                )
                scores.append(precision_at_recall(y[valid_mask], pred))
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        # Старые настройки входят в поиск: иначе можно проиграть просто потому,
        # что проверенный вариант даже не попал в сетку вариантов Optuna.
        study.enqueue_trial({
            "iterations": 600, "depth": 6,
            "learning_rate": .05, "l2_leaf_reg": 5,
        })
        study.optimize(objective, n_trials=trials)
        chosen[name] = study.best_params
        print(f"{name}: подбор P@R70 = {study.best_value:.3f}; {study.best_params}", flush=True)
    return chosen


def build_oof(fit, y, blocks, columns_by_model, base_params):
    """Предсказания для блока делает модель, видевшая лишь прошлые дни."""
    oof = []
    for block_name, fit_mask, valid_mask in blocks:
        scores = {}
        for name in MODEL_NAMES:
            scores[name] = score_block(
                make_base(base_params[name]), fit, fit, y, fit_mask, valid_mask,
                columns_by_model[name],
            )
        block = pd.DataFrame(scores)
        block.insert(0, "row_index", np.flatnonzero(valid_mask))
        block.insert(0, "block", block_name)
        block["target"] = y[valid_mask]
        oof.append(block)
    return oof


def combined_meta(blocks):
    return pd.concat([meta_features(block.loc[:, MODEL_NAMES]) for block in blocks],
                     ignore_index=True)


def tune_meta(oof, trials: int) -> dict:
    """На каждом шаге верхняя модель видит лишь более ранние OOF блоки."""
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 150, 500, step=50),
            "depth": trial.suggest_int("depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", .02, .11, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 20, log=True),
        }
        scores = []
        for valid_index in (1, 2):
            earlier = oof[:valid_index]
            model = make_meta(params)
            model.fit(combined_meta(earlier),
                      np.concatenate([part.target.to_numpy() for part in earlier]))
            pred = model.predict_proba(meta_features(oof[valid_index]))[:, 1]
            scores.append(precision_at_recall(oof[valid_index].target, pred))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=trials)
    print(f"meta: подбор P@R70 = {study.best_value:.3f}; {study.best_params}", flush=True)
    return study.best_params


def main(tune: bool, base_trials: int, meta_trials: int) -> None:
    train, test, fit, predict, columns_by_model, y = load_model_data(ROOT)
    blocks = list(time_blocks(train))

    if tune:
        base_params = tune_base(fit, y, blocks, columns_by_model, base_trials)
        oof = build_oof(fit, y, blocks, columns_by_model, base_params)
        meta_params = tune_meta(oof, meta_trials)
        params = {"random_state": RANDOM_STATE, "base": base_params, "meta": meta_params,
                  "base_trials": base_trials, "meta_trials": meta_trials}
        PARAMS_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n")
    else:
        params = json.loads(PARAMS_PATH.read_text())
        base_params, meta_params = params["base"], params["meta"]
        oof = build_oof(fit, y, blocks, columns_by_model, base_params)

    # Только этот блок не участвовал ни в подборе базовых моделей, ни в подборе meta.
    holdout = oof[-1]
    meta = make_meta(meta_params)
    meta.fit(combined_meta(oof[:-1]),
             np.concatenate([part.target.to_numpy() for part in oof[:-1]]))
    holdout_stack = meta.predict_proba(meta_features(holdout))[:, 1]
    holdout_rank = mean_percentile_rank([holdout[name].to_numpy() for name in MODEL_NAMES])
    summary_rows = [
        {"model": "Новые базовые, среднее рангов", "p_at_r70": precision_at_recall(holdout.target, holdout_rank)},
        {"model": "CatBoost поверх трёх", "p_at_r70": precision_at_recall(holdout.target, holdout_stack)},
    ]
    old_path = ROOT / "output" / "final_validation_scores.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        expected_ids = train.cookie_id.iloc[holdout.row_index].to_numpy()
        old_holdout = old.loc[old.fold.eq("17–19 апреля")].set_index("cookie_id")
        old_holdout = old_holdout.reindex(expected_ids)
        if old_holdout.rank_mean.isna().any() or not np.array_equal(
            old_holdout.target.to_numpy(), holdout.target.to_numpy()
        ):
            raise ValueError("Сохранённая старая проверка не совпадает с отложенными куками")
        summary_rows.insert(0, {
            "model": "Старое среднее рангов",
            "p_at_r70": precision_at_recall(old_holdout.target, old_holdout.rank_mean),
        })
    summary = pd.DataFrame(summary_rows)
    print("17–19 апреля, отложенные дни:\n" + summary.to_string(index=False), flush=True)

    # Верхняя модель учится на всех доступных OOF, включая 17–19 апреля.
    # Базовые модели после этого обучаются на всём размеченном train.
    final_meta = make_meta(meta_params)
    final_meta.fit(combined_meta(oof),
                   np.concatenate([part.target.to_numpy() for part in oof]))
    test_scores = {}
    for name in MODEL_NAMES:
        model = make_base(base_params[name])
        model.fit(fit[columns_by_model[name]], y)
        test_scores[name] = model.predict_proba(predict[columns_by_model[name]])[:, 1]
    answer = pd.DataFrame({
        "cookie_id": test.cookie_id,
        "score": final_meta.predict_proba(meta_features(pd.DataFrame(test_scores)))[:, 1],
    })
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    assert answer.cookie_id.is_unique
    assert answer.cookie_id.equals(test.cookie_id)
    assert set(answer.cookie_id) == set(sample.cookie_id)
    assert np.isfinite(answer.score).all() and answer.score.between(0, 1).all()

    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    summary.to_csv(output / "stacking_comparison.csv", index=False)
    details = holdout[["row_index", "target", *MODEL_NAMES]].copy()
    details.insert(0, "cookie_id", train.cookie_id.iloc[holdout.row_index].to_numpy())
    details["rank_mean"] = holdout_rank
    details["stacked_score"] = holdout_stack
    details.to_csv(output / "stacking_holdout_scores.csv", index=False, float_format="%.17g")
    answer.to_csv(output / "stacked_submission.csv", index=False, float_format="%.17g")
    print(f"Готово: {output / 'stacked_submission.csv'} ({len(answer)} кук)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true", help="заново подобрать параметры через Optuna")
    parser.add_argument("--base-trials", type=int, default=16)
    parser.add_argument("--meta-trials", type=int, default=24)
    args = parser.parse_args()
    main(args.tune, args.base_trials, args.meta_trials)
