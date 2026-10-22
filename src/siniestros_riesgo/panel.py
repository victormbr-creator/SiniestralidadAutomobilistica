"""Balanced municipality-month panel and leakage-safe features."""

import pandas as pd


def balanced_municipalities(events: pd.DataFrame, years: list[int]) -> set[str]:
    """Return municipalities with evidence of reporting in every required year."""
    observed = events.loc[events["report_status"].isin(["observed", "certified_zero"]), ["geo_id", "year"]].drop_duplicates()
    counts = observed.groupby("geo_id")["year"].nunique()
    return set(counts[counts.eq(len(years))].index)


def build_monthly_panel(events: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Aggregate victim crashes and complete months only for balanced reporters."""
    keep = balanced_municipalities(events, years)
    if not keep:
        raise ValueError("Ningún municipio tiene evidencia de reporte en todos los años")
    filtered = events[events["geo_id"].isin(keep)].copy()
    aggregate = filtered.groupby(["geo_id", "state_code", "year", "month"], as_index=False).agg(
        all_crashes=("event_id", "nunique"), victim_crashes=("victimas", "sum"), fatalities=("deaths_total", "sum")
    )
    index = pd.MultiIndex.from_product([sorted(keep), years, range(1, 13)], names=["geo_id", "year", "month"]).to_frame(index=False)
    states = filtered[["geo_id", "state_code"]].drop_duplicates("geo_id")
    panel = index.merge(states, on="geo_id", how="left", validate="many_to_one").merge(aggregate, on=["geo_id", "state_code", "year", "month"], how="left", validate="one_to_one")
    panel[["all_crashes", "victim_crashes", "fatalities"]] = panel[["all_crashes", "victim_crashes", "fatalities"]].fillna(0).astype("int64")
    panel["any_victim_crash"] = panel["victim_crashes"].gt(0).astype("int8")
    panel["date"] = pd.to_datetime(dict(year=panel.year, month=panel.month, day=1))
    panel = panel.sort_values(["geo_id", "date"])
    group = panel.groupby("geo_id", sort=False)["victim_crashes"]
    panel["lag_1"] = group.shift(1)
    panel["mean_lag_12"] = group.transform(lambda series: series.shift(1).rolling(12, min_periods=1).mean())
    return panel


def freeze_after_year(panel: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    """Create a holdout view whose autoregressive features stop at a cutoff year."""
    result = panel.copy()
    cutoff = result["year"].le(cutoff_year)
    history = result.loc[cutoff].groupby("geo_id")["victim_crashes"].tail(12).groupby(result.loc[cutoff, "geo_id"]).mean()
    mask = result["year"].gt(cutoff_year)
    result.loc[mask, "mean_lag_12"] = result.loc[mask, "geo_id"].map(history)
    result.loc[mask, "lag_1"] = result.loc[mask, "geo_id"].map(history)
    return result


def freeze_post_2022_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the original 2023-2024 evaluation."""
    return freeze_after_year(panel, 2022)
