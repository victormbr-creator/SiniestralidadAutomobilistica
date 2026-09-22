"""Tests for temporal isolation and predictor exclusions."""

import pandas as pd

from siniestros_viales.evaluate_models import calculate_metrics
from siniestros_viales.train_models import select_predictors, temporal_split


def test_temporal_split_and_leakage_exclusions() -> None:
    data = pd.DataFrame({"year": [2018, 2022, 2023, 2024], "event_id": [1, 2, 3, 4], "deaths": [0, 1, 0, 0], "injuries": [0, 0, 1, 0], "fatal": [0, 1, 0, 0], "victimas": [0, 1, 1, 0], "hour": [1, 2, 3, 4]})
    train, test = temporal_split(data)
    assert train["year"].tolist() == [2018, 2022]
    assert test["year"].tolist() == [2023, 2024]
    assert select_predictors(train) == ["hour"]


def test_metrics_perfect_predictions() -> None:
    result = calculate_metrics(pd.Series([0, 1]), [0.1, 0.9])
    assert result["f1"] == 1
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
