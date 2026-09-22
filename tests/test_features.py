"""Tests for optional feature behavior."""

import pandas as pd

from siniestros_viales.features import add_features


def test_features_handle_missing_optional_columns() -> None:
    result = add_features(pd.DataFrame({"victimas": [0, 1]}))
    assert result.columns.tolist() == ["victimas"]


def test_time_features() -> None:
    frame = pd.DataFrame({"event_date": ["2023-01-07"], "event_time": ["21:30"]})
    result = add_features(frame)
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "weekend"]
    assert result.loc[0, "night"]
    assert result.loc[0, "time_band"] == "noche"
