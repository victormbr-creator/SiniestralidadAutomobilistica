import pandas as pd
from siniestros_riesgo.panel import balanced_municipalities, build_monthly_panel, freeze_after_year, freeze_post_2022_features


def sample_events() -> pd.DataFrame:
    rows = []
    for year in (2021, 2022, 2023):
        rows.append({"geo_id": "01001", "state_code": "01", "year": year, "month": 1, "event_id": f"a{year}", "victimas": 1, "deaths_total": 0, "report_status": "observed"})
    rows.append({"geo_id": "02001", "state_code": "02", "year": 2021, "month": 1, "event_id": "b", "victimas": 1, "deaths_total": 0, "report_status": "observed"})
    return pd.DataFrame(rows)


def test_balanced_sample_and_completion() -> None:
    data = sample_events()
    assert balanced_municipalities(data, [2021, 2022, 2023]) == {"01001"}
    panel = build_monthly_panel(data, [2021, 2022, 2023])
    assert len(panel) == 36
    assert panel.victim_crashes.sum() == 3


def test_frozen_holdout_does_not_use_2023_outcome() -> None:
    panel = build_monthly_panel(sample_events(), [2021, 2022, 2023])
    frozen = freeze_post_2022_features(panel)
    value = frozen.loc[(frozen.year == 2023) & (frozen.month == 2), "lag_1"].iloc[0]
    assert value == panel.loc[panel.year <= 2022].groupby("geo_id").victim_crashes.tail(12).mean()


def test_frozen_holdout_accepts_a_new_cutoff() -> None:
    data = sample_events()
    extra = data[data.year.eq(2023)].copy()
    extra["year"] = 2024
    extra["event_id"] = "a2024"
    panel = build_monthly_panel(pd.concat([data, extra], ignore_index=True), [2021, 2022, 2023, 2024])
    frozen = freeze_after_year(panel, 2023)
    expected = panel.loc[panel.year.eq(2023), "victim_crashes"].mean()
    assert frozen.loc[(frozen.year == 2024) & (frozen.month == 2), "lag_1"].iloc[0] == expected
