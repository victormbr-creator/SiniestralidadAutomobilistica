"""Capacity, projected pressure, and transparent facility-location helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return a weighted empirical quantile for nonnegative weights."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[valid], weights[valid]
    if not len(values) or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cumulative, quantile, side="left")])


def accessibility_metrics(
    distances_m: np.ndarray,
    weights: np.ndarray,
    boroughs: np.ndarray,
    threshold_m: float = 5_000,
) -> dict[str, float]:
    """Compute citywide efficiency and borough-tail equity metrics."""
    distances_m = np.asarray(distances_m, dtype=float)
    weights = np.asarray(weights, dtype=float)
    boroughs = np.asarray(boroughs)
    total = weights.sum()
    mean_km = float(np.average(distances_m, weights=weights) / 1_000)
    beyond = float(weights[distances_m > threshold_m].sum() / total)
    p90s = [
        weighted_quantile(distances_m[boroughs == borough], weights[boroughs == borough], 0.90)
        for borough in np.unique(boroughs)
    ]
    return {
        "weighted_mean_distance_km": mean_km,
        "risk_share_beyond_5km": beyond,
        "worst_borough_p90_km": float(np.nanmax(p90s) / 1_000),
    }


def projected_pressure(
    forecast: pd.DataFrame,
    capacity: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate parallel, interpretable demand-to-resource ratios by borough."""
    resource_columns = [
        "emergency_hospitals",
        "emergency_beds",
        "non_census_beds",
        "census_beds",
        "emergency_physicians",
        "contact_nurses",
        "critical_care_ambulances",
    ]
    out = forecast.merge(capacity, on="geo_id", how="left", validate="one_to_one")
    out[resource_columns] = out[resource_columns].fillna(0)
    demand = out["predicted_horizon_victim_crashes"]
    ratios = {
        "predicted_crashes_per_emergency_bed": "emergency_beds",
        "predicted_crashes_per_emergency_physician": "emergency_physicians",
        "predicted_crashes_per_contact_nurse": "contact_nurses",
    }
    for output, denominator in ratios.items():
        out[output] = demand / out[denominator].replace(0, np.nan)
    out["demand_share"] = demand / demand.sum()
    out["emergency_bed_share"] = out["emergency_beds"] / out["emergency_beds"].sum()
    out["pressure_gap_emergency_beds"] = out["demand_share"] - out["emergency_bed_share"]
    out["pressure_rank"] = (
        out["predicted_crashes_per_emergency_bed"]
        .rank(method="min", ascending=False, na_option="top")
        .astype(int)
    )
    return out.sort_values("pressure_rank").reset_index(drop=True)

