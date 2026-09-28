"""Обучает финальные модели из исходных данных и создаёт submission.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from utils.blending import mean_percentile_rank
from utils.features import build_features
from utils.journeys import build_journey_features


RANDOM_STATE = 42
DATE_COLUMNS = ["cookie_created_at", "window_start_ts", "window_end_ts"]


def make_model() -> CatBoostClassifier:
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
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    data = root / "data"
    train = pd.read_csv(data / "train.csv", parse_dates=DATE_COLUMNS)
    test = pd.read_csv(data / "test.csv", parse_dates=DATE_COLUMNS)
    events = pd.read_csv(data / "events.csv.gz", parse_dates=["event_ts"])

    features, groups, _ = build_features(train, test, events)
    journeys, journey_groups = build_journey_features(train, test, events)
    features = features.merge(
        journeys.drop(columns="split"), on="cookie_id", how="left", validate="one_to_one"
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
    scores = []
    for columns in [base, routes, context]:
        model = make_model()
        model.fit(fit[columns], train.target.to_numpy())
        scores.append(model.predict_proba(predict[columns])[:, 1])

    answer = pd.DataFrame({
        "cookie_id": test.cookie_id,
        "score": mean_percentile_rank(scores),
    })
    sample = pd.read_csv(data / "sample_submission.csv")
    assert answer.cookie_id.is_unique
    assert set(answer.cookie_id) == set(sample.cookie_id)
    assert np.isfinite(answer.score).all() and answer.score.between(0, 1).all()

    path = root / "output" / "submission.csv"
    path.parent.mkdir(exist_ok=True)
    answer.to_csv(path, index=False, float_format="%.10f")
    print(f"Готово: {path} ({len(answer)} кук)")


if __name__ == "__main__":
    main()
