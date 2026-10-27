"""Integrate SINERHIAS resources and locate additional emergency hospitals."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from siniestros_riesgo.config import FIGURES, PROCESSED, RAW, TABLES, setup_logging
from siniestros_riesgo.hospital_pressure import accessibility_metrics, projected_pressure


logger = setup_logging("build_hospital_pressure")
RESOURCE_PATH = RAW / "sinerhias" / "unidades_cdmx_2026.json"
NUMERIC_FIELDS = [
    "camas", "tot_camas_hospit", "tot_camas_no_hospi", "camasurg", "camterinte",
    "intercama", "total_ocup", "medcpuh", "medurgeuh", "medtrauuh", "medciruuh",
    "enfcpuh", "quirofanos", "tomografo", "ambequip", "ambcuin", "latitud", "longitud",
]


def load_resources() -> pd.DataFrame:
    data = pd.DataFrame(json.loads(RESOURCE_PATH.read_text(encoding="utf-8")))
    for column in NUMERIC_FIELDS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["resource_cutoff"] = np.where(
        data["institucion"].isin(["SSA", "IMB"]), "2026-05-15", "2025-12-31"
    )
    data.to_parquet(PROCESSED / "cdmx_sinerhias_resources_2026.parquet", index=False)
    data.to_csv(TABLES / "cdmx_sinerhias_resources_2026.csv", index=False)
    return data


def join_emergency_capacity(resources: pd.DataFrame) -> pd.DataFrame:
    hospitals = pd.read_parquet(PROCESSED / "cdmx_emergency_hospitals_sedesa_2025.parquet")
    keep = ["clues", "resource_cutoff"] + [c for c in NUMERIC_FIELDS if c not in {"latitud", "longitud"}]
    capacity = hospitals.merge(
        resources[keep], left_on="CLUES", right_on="clues", how="left", validate="one_to_one",
        indicator=True,
    )
    if not capacity["_merge"].eq("both").all():
        missing = capacity.loc[capacity["_merge"].ne("both"), "CLUES"].tolist()
        raise ValueError(f"Hospitales sin recursos SINERHIAS: {missing}")
    capacity = capacity.drop(columns=["_merge", "clues"])
    capacity = capacity.rename(columns={
        "camasurg": "emergency_beds",
        "tot_camas_no_hospi": "non_census_beds",
        "camas": "census_beds",
        "medurgeuh": "emergency_physicians",
        "medcpuh": "physicians_total",
        "enfcpuh": "contact_nurses",
        "total_ocup": "staff_total",
        "ambcuin": "critical_care_ambulances",
        "ambequip": "transfer_ambulances",
    })
    capacity.to_parquet(PROCESSED / "cdmx_emergency_hospital_capacity_2026.parquet", index=False)
    capacity.to_csv(TABLES / "cdmx_emergency_hospital_capacity_2026.csv", index=False)
    return capacity


def borough_pressure(capacity: pd.DataFrame) -> pd.DataFrame:
    aggregate = capacity.groupby("geo_id", as_index=False).agg(
        emergency_hospitals=("CLUES", "size"),
        emergency_beds=("emergency_beds", "sum"),
        non_census_beds=("non_census_beds", "sum"),
        census_beds=("census_beds", "sum"),
        emergency_physicians=("emergency_physicians", "sum"),
        contact_nurses=("contact_nurses", "sum"),
        critical_care_ambulances=("critical_care_ambulances", "sum"),
    )
    risk = pd.read_csv(TABLES / "cdmx_risk_capacity_priority_future.csv", dtype={"geo_id": "string"})
    forecast = risk[["geo_id", "nomgeo", "population", "predicted_horizon_victim_crashes"]].copy()
    pressure = projected_pressure(forecast, aggregate)
    pressure.to_csv(TABLES / "cdmx_borough_hospital_pressure_future.csv", index=False)
    return pressure


def demand_points(pressure: pd.DataFrame) -> gpd.GeoDataFrame:
    crashes = pd.read_parquet(PROCESSED / "cdmx_victim_crash_points_with_emergency_distance.parquet")
    crashes = crashes[
        crashes["victim_crash"] & crashes["coordinate_valid"] & crashes["inside_cdmx_boundary"]
    ].copy()
    crashes["severity_weight"] = pd.to_numeric(crashes["victims"], errors="coerce").fillna(1).clip(lower=1)
    totals = crashes.groupby("geo_id")["severity_weight"].transform("sum")
    predicted = pressure.set_index("geo_id")["predicted_horizon_victim_crashes"]
    crashes["predicted_demand_weight"] = crashes["severity_weight"] / totals * crashes["geo_id"].map(predicted)
    if crashes["predicted_demand_weight"].isna().any():
        raise ValueError("No fue posible distribuir el pronóstico entre puntos históricos")
    return gpd.GeoDataFrame(
        crashes,
        geometry=gpd.points_from_xy(crashes["LONGITUD"], crashes["LATITUD"]),
        crs=4326,
    ).to_crs(32614)


def assign_hospital_pressure(points: gpd.GeoDataFrame, capacity: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    hospital_gdf = gpd.GeoDataFrame(
        capacity.copy(), geometry=gpd.points_from_xy(capacity["longitud"], capacity["latitud"]), crs=4326,
    ).to_crs(32614)
    point_xy = np.column_stack([points.geometry.x, points.geometry.y])
    hospital_xy = np.column_stack([hospital_gdf.geometry.x, hospital_gdf.geometry.y])
    distances = cdist(point_xy, hospital_xy)
    nearest = distances.argmin(axis=1)
    base_distance = distances[np.arange(len(points)), nearest]
    assigned = pd.DataFrame({
        "CLUES": hospital_gdf.iloc[nearest]["CLUES"].to_numpy(),
        "predicted_demand_weight": points["predicted_demand_weight"].to_numpy(),
    }).groupby("CLUES", as_index=False)["predicted_demand_weight"].sum()
    out = capacity.merge(assigned, on="CLUES", how="left", validate="one_to_one")
    out["assigned_predicted_crashes"] = out.pop("predicted_demand_weight").fillna(0)
    for denominator, suffix in [
        ("emergency_beds", "emergency_bed"),
        ("emergency_physicians", "emergency_physician"),
        ("contact_nurses", "contact_nurse"),
    ]:
        out[f"assigned_crashes_per_{suffix}"] = (
            out["assigned_predicted_crashes"] / out[denominator].replace(0, np.nan)
        )
    out["assigned_demand_share"] = out["assigned_predicted_crashes"] / out["assigned_predicted_crashes"].sum()
    out["emergency_bed_share"] = out["emergency_beds"] / out["emergency_beds"].sum()
    out["pressure_gap_emergency_beds"] = out["assigned_demand_share"] - out["emergency_bed_share"]
    out = out.sort_values("assigned_crashes_per_emergency_bed", ascending=False, na_position="first")
    out.to_csv(TABLES / "cdmx_hospital_pressure_future.csv", index=False)
    return out, point_xy, base_distance


def _candidate_grid(boundary: gpd.GeoDataFrame, hospital_xy: np.ndarray) -> gpd.GeoDataFrame:
    metric = boundary.to_crs(32614)
    union = metric.geometry.union_all()
    xmin, ymin, xmax, ymax = union.bounds
    xs = np.arange(np.floor(xmin / 1_000) * 1_000 + 500, xmax, 1_000)
    ys = np.arange(np.floor(ymin / 1_000) * 1_000 + 500, ymax, 1_000)
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(np.repeat(xs, len(ys)), np.tile(ys, len(xs))), crs=32614
    )
    grid = grid[grid.geometry.within(union)].copy()
    xy = np.column_stack([grid.geometry.x, grid.geometry.y])
    grid = grid[cdist(xy, hospital_xy).min(axis=1) >= 2_000].copy()
    return gpd.sjoin(grid, metric[["cvegeo", "nomgeo", "geometry"]], predicate="within", how="left").drop(columns="index_right")


def _score_candidates(
    candidates: gpd.GeoDataFrame,
    point_xy: np.ndarray,
    current_distance: np.ndarray,
    weights: np.ndarray,
    boroughs: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for idx, candidate in candidates.iterrows():
        candidate_distance = np.hypot(point_xy[:, 0] - candidate.geometry.x, point_xy[:, 1] - candidate.geometry.y)
        proposed = np.minimum(current_distance, candidate_distance)
        metrics = accessibility_metrics(proposed, weights, boroughs)
        captured = float(weights[candidate_distance < current_distance].sum())
        rows.append({"candidate_index": idx, "captured_predicted_crashes": captured, **metrics})
    return pd.DataFrame(rows).set_index("candidate_index")


def locate_new_hospitals(
    points: gpd.GeoDataFrame,
    capacity: pd.DataFrame,
    point_xy: np.ndarray,
    base_distance: np.ndarray,
    boundary: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    hospitals = gpd.GeoDataFrame(
        capacity.copy(), geometry=gpd.points_from_xy(capacity["longitud"], capacity["latitud"]), crs=4326,
    ).to_crs(32614)
    hospital_xy = np.column_stack([hospitals.geometry.x, hospitals.geometry.y])
    candidates = _candidate_grid(boundary, hospital_xy)
    weights = points["predicted_demand_weight"].to_numpy()
    boroughs = points["geo_id"].astype(str).to_numpy()
    base_metrics = accessibility_metrics(base_distance, weights, boroughs)
    first_scores = _score_candidates(candidates, point_xy, base_distance, weights, boroughs)

    def select(scores: pd.DataFrame, objective: str) -> int:
        return int(scores.sort_values([objective, "weighted_mean_distance_km"]).index[0])

    efficiency_index = select(first_scores, "weighted_mean_distance_km")
    equity_index = select(first_scores, "worst_borough_p90_km")
    equity_point = candidates.loc[equity_index].geometry
    after_first = np.minimum(
        base_distance,
        np.hypot(point_xy[:, 0] - equity_point.x, point_xy[:, 1] - equity_point.y),
    )
    remaining = candidates[
        (candidates.index != equity_index)
        & (candidates.geometry.distance(equity_point) >= 2_000)
    ]
    second_scores = _score_candidates(remaining, point_xy, after_first, weights, boroughs)
    second_index = select(second_scores, "worst_borough_p90_km")

    median_beds = float(capacity["emergency_beds"].median())
    median_physicians = float(capacity["emergency_physicians"].median())
    median_nurses = float(capacity["contact_nurses"].median())
    rows = []
    specifications = [
        ("efficiency_1", efficiency_index, first_scores.loc[efficiency_index], base_metrics),
        ("equity_1", equity_index, first_scores.loc[equity_index], base_metrics),
        ("equity_2", second_index, second_scores.loc[second_index], accessibility_metrics(after_first, weights, boroughs)),
    ]
    for criterion, idx, score, before in specifications:
        point = gpd.GeoSeries([candidates.loc[idx].geometry], crs=32614).to_crs(4326).iloc[0]
        rows.append({
            "criterion": criterion,
            "recommended_order": {"equity_1": 1, "equity_2": 2}.get(criterion, np.nan),
            "geo_id": candidates.loc[idx, "cvegeo"],
            "alcaldia": candidates.loc[idx, "nomgeo"],
            "latitude": point.y,
            "longitude": point.x,
            "grid_resolution_m": 1_000,
            "minimum_existing_distance_m": 2_000,
            "planned_emergency_beds_scenario": median_beds,
            "planned_emergency_physicians_scenario": median_physicians,
            "planned_contact_nurses_scenario": median_nurses,
            "captured_predicted_crashes": score["captured_predicted_crashes"],
            "captured_crashes_per_planned_bed": score["captured_predicted_crashes"] / median_beds,
            "weighted_mean_distance_before_km": before["weighted_mean_distance_km"],
            "weighted_mean_distance_after_km": score["weighted_mean_distance_km"],
            "worst_borough_p90_before_km": before["worst_borough_p90_km"],
            "worst_borough_p90_after_km": score["worst_borough_p90_km"],
            "risk_share_beyond_5km_before": before["risk_share_beyond_5km"],
            "risk_share_beyond_5km_after": score["risk_share_beyond_5km"],
        })
    recommendations = pd.DataFrame(rows)
    recommendations.to_csv(TABLES / "cdmx_new_hospital_candidates.csv", index=False)
    return recommendations, candidates


def make_figures(
    pressure: pd.DataFrame,
    points: gpd.GeoDataFrame,
    capacity: pd.DataFrame,
    recommendations: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
) -> None:
    plot = pressure.sort_values("predicted_crashes_per_emergency_bed")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot["nomgeo"], plot["predicted_crashes_per_emergency_bed"], color="#B14A3B")
    ax.set(title="Presión vial proyectada sobre camas de urgencias", xlabel="Siniestros con víctimas predichos por cama de urgencias")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "cdmx_hospital_pressure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    base = boundary.to_crs(4326)
    hospital_gdf = gpd.GeoDataFrame(
        capacity, geometry=gpd.points_from_xy(capacity["longitud"], capacity["latitud"]), crs=4326,
    )
    rec = gpd.GeoDataFrame(
        recommendations, geometry=gpd.points_from_xy(recommendations["longitude"], recommendations["latitude"]), crs=4326,
    )
    pts = points.to_crs(4326)
    fig, ax = plt.subplots(figsize=(9, 9))
    base.plot(ax=ax, facecolor="#F4F0E8", edgecolor="#6F7782", linewidth=.7)
    pts.plot(ax=ax, color="#D36B45", markersize=2, alpha=.10)
    hospital_gdf.plot(ax=ax, color="#19647E", marker="^", markersize=22, label="Hospital actual con urgencias")
    eq = rec[rec["criterion"].str.startswith("equity")]
    eq.plot(ax=ax, color="#7B2CBF", marker="*", markersize=180, edgecolor="white", linewidth=.5, label="Secuencia recomendada")
    eff = rec[rec["criterion"].eq("efficiency_1")]
    eff.plot(ax=ax, color="#2A9D8F", marker="D", markersize=55, edgecolor="white", linewidth=.5, label="Alternativa de eficiencia")
    for row in eq.itertuples():
        ax.annotate(f"{int(row.recommended_order)}. {row.alcaldia}", (row.longitude, row.latitude), xytext=(5, 5), textcoords="offset points", fontsize=8, weight="bold")
    ax.set_title("Ubicaciones analíticas para hospitales adicionales\ncriterio de equidad territorial, cuadrícula de 1 km")
    ax.legend(loc="lower left", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "cdmx_new_hospital_locations.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    resources = load_resources()
    capacity = join_emergency_capacity(resources)
    pressure = borough_pressure(capacity)
    points = demand_points(pressure)
    hospital_pressure, point_xy, base_distance = assign_hospital_pressure(points, capacity)
    boundary = gpd.read_file("data/external/cdmx_alcaldias_inegi.geojson")
    recommendations, _ = locate_new_hospitals(points, capacity, point_xy, base_distance, boundary)
    make_figures(pressure, points, capacity, recommendations, boundary)
    logger.info(
        "Integrados %d establecimientos SINERHIAS, %d hospitales y %d ubicaciones candidatas",
        len(resources), len(hospital_pressure), len(recommendations),
    )


if __name__ == "__main__":
    main()
