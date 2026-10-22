import numpy as np
import pandas as pd
from siniestros_riesgo.coverage import greedy_maximum_coverage, risk_beyond_threshold
from siniestros_riesgo.evaluation import capture_at_share, count_metrics


def test_count_metrics_and_capture() -> None:
    actual = np.array([10, 0, 0, 0])
    predicted = np.array([4, 3, 2, 1])
    assert capture_at_share(actual, predicted, .25) == 1
    assert count_metrics(actual, predicted)["mae"] >= 0


def test_coverage_functions() -> None:
    travel = pd.DataFrame({"cell_id": [1, 2], "predicted_risk": [3.0, 1.0], "nearest_existing_minutes": [20, 5]})
    assert risk_beyond_threshold(travel, 15) == .75
    incidence = pd.DataFrame({"A": [True, False], "B": [False, True]}, index=[1, 2])
    result = greedy_maximum_coverage(incidence, pd.Series([3.0, 1.0], index=[1, 2]), p_max=2)
    assert result.iloc[-1].risk_covered == 4
