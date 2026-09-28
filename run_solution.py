"""Проверяет ансамбль на будущих днях и создаёт submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from utils.blending import mean_percentile_rank
from utils.features import build_features
from utils.journeys import build_journey_features
from utils.pointer_location import build_pointer_location_features
from utils.metric import precision_at_recall
from utils.time_validation import walk_forward_splits


RANDOM_STATE = 42
DATE_COLUMNS = ["cookie_created_at", "window_start_ts", "window_end_ts"]


def make_model(extra_cat_features: tuple[str, ...] = ()) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=.05,
        l2_leaf_reg=5,
        loss_function="Logloss",
        verbose=False,
        random_seed=RANDOM_STATE,
        thread_count=4,
        allow_writing_files=False,
        cat_features=["top_location", *extra_cat_features],
    )


def load_model_data(root: Path):
    """Один раз собирает признаки для трёх вариантов CatBoost."""
    data = root / "data"
    train = pd.read_csv(data / "train.csv", parse_dates=DATE_COLUMNS)
    test = pd.read_csv(data / "test.csv", parse_dates=DATE_COLUMNS)
    events = pd.read_csv(data / "events.csv.gz", parse_dates=["event_ts"])

    features, groups, _ = build_features(train, test, events)
    journeys, journey_groups = build_journey_features(train, test, events)
    pointer = build_pointer_location_features(train, test, events)
    features = features.merge(
        journeys.drop(columns="split"), on="cookie_id", how="left", validate="one_to_one"
    ).merge(
        pointer, on="cookie_id", how="left", validate="one_to_one"
    )
    fit = features.iloc[:len(train)].reset_index(drop=True)
    predict = features.iloc[len(train):].reset_index(drop=True)
    assert fit.cookie_id.equals(train.cookie_id)
    assert predict.cookie_id.equals(test.cookie_id)

    base = [column for column in groups["base"]
            if column not in {"headless_ua", "automation_ua", "ua_count"}]
    routes = base + groups["routes"]
    context = routes + [column for column in journey_groups["context"]
                        if column != "seller_switch_share"]
    mouse_and_location = ["pointer_x_std", "pointer_y_std", "counter_zeros", "top_location"]
    columns_by_model = {
        "base_plus": base + mouse_and_location,
        "routes_plus": routes + mouse_and_location,
        "context_plus": context + mouse_and_location,
    }
    y = train.target.to_numpy()
    return train, test, fit, predict, columns_by_model, y


def main() -> None:
    root = Path(__file__).resolve().parent
    train, test, fit, predict, columns_by_model, y = load_model_data(root)
    validation_parts = []
    for fold_name, fit_mask, valid_mask in walk_forward_splits(train):
        fold_scores = {}
        for name, columns in columns_by_model.items():
            model = make_model()
            model.fit(fit.loc[fit_mask, columns], y[fit_mask])
            fold_scores[name] = model.predict_proba(fit.loc[valid_mask, columns])[:, 1]
        fold_scores["rank_mean"] = mean_percentile_rank(list(fold_scores.values()))
        fold_result = pd.DataFrame({
            "cookie_id": train.loc[valid_mask, "cookie_id"].to_numpy(),
            "fold": fold_name,
            "target": y[valid_mask],
            **fold_scores,
        })
        validation_parts.append(fold_result)
        print(f"{fold_name}: P@R70 = {precision_at_recall(y[valid_mask], fold_scores['rank_mean']):.3f}")

    output = root / "output"
    output.mkdir(exist_ok=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    validation.to_csv(output / "final_validation_scores.csv", index=False, float_format="%.17g")

    scores = []
    for columns in columns_by_model.values():
        model = make_model()
        model.fit(fit[columns], y)
        scores.append(model.predict_proba(predict[columns])[:, 1])

    answer = pd.DataFrame({
        "cookie_id": test.cookie_id,
        "score": mean_percentile_rank(scores),
    })
    sample = pd.read_csv(root / "data" / "sample_submission.csv")
    assert answer.cookie_id.is_unique
    assert set(answer.cookie_id) == set(sample.cookie_id)
    assert np.isfinite(answer.score).all() and answer.score.between(0, 1).all()

    path = output / "submission.csv"
    answer.to_csv(path, index=False, float_format="%.17g")
    print(f"Готово: {path} ({len(answer)} кук)")


if __name__ == "__main__":
    main()
