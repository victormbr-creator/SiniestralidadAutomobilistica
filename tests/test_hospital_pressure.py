import numpy as np
import pandas as pd

from siniestros_riesgo.hospital_pressure import (
    accessibility_metrics,
    distance_decay,
    projected_pressure,
    spatial_bed_pressure,
    weighted_quantile,
)


def test_weighted_quantile_and_accessibility() -> None:
    values = np.array([1.0, 2.0, 10.0])
    weights = np.array([1.0, 8.0, 1.0])
    assert weighted_quantile(values, weights, 0.5) == 2.0
    metrics = accessibility_metrics(values * 1_000, weights, np.array(["A", "A", "B"]))
    assert metrics["weighted_mean_distance_km"] == 2.7
    assert metrics["risk_share_beyond_5km"] == 0.1
    assert metrics["worst_borough_p90_km"] == 10.0


def test_projected_pressure_uses_observed_denominators() -> None:
    forecast = pd.DataFrame({
        "geo_id": ["A", "B"], "predicted_horizon_victim_crashes": [20.0, 10.0]
    })
    capacity = pd.DataFrame({
        "geo_id": ["A", "B"], "emergency_hospitals": [1, 1], "emergency_beds": [2, 5],
        "non_census_beds": [3, 6], "census_beds": [10, 20], "emergency_physicians": [1, 2],
        "contact_nurses": [4, 10], "critical_care_ambulances": [1, 1],
    })
    result = projected_pressure(forecast, capacity).set_index("geo_id")
    assert result.loc["A", "predicted_crashes_per_emergency_bed"] == 10
    assert result.loc["A", "pressure_rank"] == 1


def test_distance_decay_applies_catchment() -> None:
    weights = distance_decay(np.array([[0.0, 11_000.0]]), scale_m=3_000, catchment_m=10_000)
    assert weights[0, 0] == 1.0
    assert weights[0, 1] == 0.0


def test_spatial_bed_pressure_conserves_supply_and_uses_local_demand() -> None:
    distances = np.array([[0.0], [3_000.0]])
    demand = np.array([10.0, 0.0])
    accessible_beds, pressure, _ = spatial_bed_pressure(distances, demand, np.array([12.0]))
    assert np.isclose(accessible_beds.sum(), 12.0)
    assert pressure[0] > 0
    assert pressure[1] == 0
