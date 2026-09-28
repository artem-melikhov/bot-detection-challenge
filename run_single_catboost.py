"""Обучает одну CatBoost без Word2Vec и создаёт файл для проверки."""

from pathlib import Path

import numpy as np
import pandas as pd

from run_solution import load_model_data, make_model


def main() -> None:
    root = Path(__file__).resolve().parent
    train, test, fit, predict, groups, y = load_model_data(root)
    columns = groups["context_plus"]

    model = make_model()
    model.fit(fit[columns], y)
    score = model.predict_proba(predict[columns])[:, 1]

    answer = pd.DataFrame({"cookie_id": test.cookie_id, "score": score})
    sample = pd.read_csv(root / "data" / "sample_submission.csv")
    assert len(answer) == len(sample) == len(test)
    assert answer.cookie_id.is_unique
    assert set(answer.cookie_id) == set(sample.cookie_id)
    assert np.isfinite(answer.score).all() and answer.score.between(0, 1).all()

    output = root / "output"
    output.mkdir(exist_ok=True)
    submission = output / "submission.csv"
    answer.to_csv(submission, index=False, float_format="%.17g")
    print(f"Готово: {submission} ({len(answer)} кук)")
    print(f"Признаки: {len(columns)}; обучение: {len(train)} кук")


if __name__ == "__main__":
    main()
