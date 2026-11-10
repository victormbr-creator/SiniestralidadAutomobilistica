"""Integrate SINERHIAS resources and locate additional emergency hospitals."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import box

from siniestros_riesgo.config import FIGURES, PROCESSED, RAW, TABLES, setup_logging
from siniestros_riesgo.hospital_pressure import (
    accessibility_metrics,
    distance_decay,
    pressure_metrics,
    projected_pressure,
    spatial_bed_pressure,
    weighted_quantile,
)


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


def build_opportunity_grid(
    points: gpd.GeoDataFrame,
    capacity: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    resolution_m: int = 1_000,
) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate predicted crash demand and accessible emergency beds to a 1 km grid."""
    metric = boundary.to_crs(32614)
    union = metric.geometry.union_all()
    xmin, ymin, xmax, ymax = union.bounds
    x0 = np.floor(xmin / resolution_m) * resolution_m
    y0 = np.floor(ymin / resolution_m) * resolution_m
    xs = np.arange(x0, xmax, resolution_m)
    ys = np.arange(y0, ymax, resolution_m)
    rows = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            square = box(x, y, x + resolution_m, y + resolution_m)
            if square.intersects(union):
                rows.append({
                    "grid_id": f"g{ix:03d}_{iy:03d}",
                    "ix": ix,
                    "iy": iy,
                    "center_x": x + resolution_m / 2,
                    "center_y": y + resolution_m / 2,
                    "geometry": square.intersection(union),
                })
    grid = gpd.GeoDataFrame(rows, geometry="geometry", crs=32614)

    overlaps = gpd.overlay(
        grid[["grid_id", "geometry"]],
        metric[["cvegeo", "nomgeo", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    overlaps["overlap_area"] = overlaps.geometry.area
    borough = (
        overlaps.sort_values("overlap_area", ascending=False)
        .drop_duplicates("grid_id")[["grid_id", "cvegeo", "nomgeo"]]
    )
    grid = grid.merge(borough, on="grid_id", how="left", validate="one_to_one")

    point_ix = np.floor((points.geometry.x.to_numpy() - x0) / resolution_m).astype(int)
    point_iy = np.floor((points.geometry.y.to_numpy() - y0) / resolution_m).astype(int)
    point_grid = pd.DataFrame({
        "grid_id": [f"g{ix:03d}_{iy:03d}" for ix, iy in zip(point_ix, point_iy)],
        "predicted_demand": points["predicted_demand_weight"].to_numpy(),
        "observed_victim_crash_points": 1,
    })
    demand = point_grid.groupby("grid_id", as_index=False).agg(
        predicted_demand=("predicted_demand", "sum"),
        observed_victim_crash_points=("observed_victim_crash_points", "sum"),
    )
    grid = grid.merge(demand, on="grid_id", how="left", validate="one_to_one")
    grid[["predicted_demand", "observed_victim_crash_points"]] = grid[
        ["predicted_demand", "observed_victim_crash_points"]
    ].fillna(0)
    if not np.isclose(grid["predicted_demand"].sum(), points["predicted_demand_weight"].sum()):
        raise ValueError("La cuadrícula no conservó el total de demanda predicha")

    grid_xy = grid[["center_x", "center_y"]].to_numpy()
    hospitals = gpd.GeoDataFrame(
        capacity.copy(), geometry=gpd.points_from_xy(capacity["longitud"], capacity["latitud"]), crs=4326,
    ).to_crs(32614)
    hospital_xy = np.column_stack([hospitals.geometry.x, hospitals.geometry.y])
    distances = cdist(grid_xy, hospital_xy)
    accessible_beds, pressure, _ = spatial_bed_pressure(
        distances,
        grid["predicted_demand"].to_numpy(),
        capacity["emergency_beds"].fillna(0).to_numpy(dtype=float),
    )
    grid["accessible_emergency_bed_equivalents"] = accessible_beds
    grid["predicted_demand_per_accessible_emergency_bed"] = pressure
    finite = np.isfinite(pressure) & (grid["predicted_demand"].to_numpy() > 0)
    reference = weighted_quantile(
        pressure[finite], grid.loc[finite, "predicted_demand"].to_numpy(), 0.50
    )
    grid["relative_pressure_index"] = pressure / reference
    grid["nearest_hospital_distance_km"] = distances.min(axis=1) / 1_000
    centers_wgs84 = gpd.GeoSeries(
        gpd.points_from_xy(grid["center_x"], grid["center_y"]), crs=32614
    ).to_crs(4326)
    grid["center_longitude"] = centers_wgs84.x.to_numpy()
    grid["center_latitude"] = centers_wgs84.y.to_numpy()

    published = grid.to_crs(4326)
    published.to_parquet(PROCESSED / "cdmx_opportunity_grid_future.parquet", index=False)
    grid.drop(columns="geometry").to_csv(TABLES / "cdmx_opportunity_grid_future.csv", index=False)
    return grid, grid_xy, hospital_xy, distances


def _candidate_grid(
    opportunity_grid: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    hospital_xy: np.ndarray,
) -> gpd.GeoDataFrame:
    union = boundary.to_crs(32614).geometry.union_all()
    candidates = gpd.GeoDataFrame(
        opportunity_grid.drop(columns="geometry").copy(),
        geometry=gpd.points_from_xy(opportunity_grid["center_x"], opportunity_grid["center_y"]),
        crs=32614,
    )
    candidates = candidates[candidates.geometry.within(union)].copy()
    xy = np.column_stack([candidates.geometry.x, candidates.geometry.y])
    return candidates[cdist(xy, hospital_xy).min(axis=1) >= 2_000].copy()


def _score_pressure_candidates(
    candidates: gpd.GeoDataFrame,
    grid_xy: np.ndarray,
    demand: np.ndarray,
    base_supply: np.ndarray,
    current_distance: np.ndarray,
    boroughs: np.ndarray,
    planned_beds: float,
) -> pd.DataFrame:
    rows = []
    for idx, candidate in candidates.iterrows():
        distances = np.hypot(
            grid_xy[:, 0] - candidate.geometry.x,
            grid_xy[:, 1] - candidate.geometry.y,
        )
        kernel = distance_decay(distances[:, None])[:, 0]
        weighted_catchment_demand = float(kernel @ demand)
        if weighted_catchment_demand <= 0:
            continue
        kernel_total = float(kernel.sum())
        if kernel_total <= 0:
            continue
        proposed_supply = base_supply + kernel * (planned_beds / kernel_total)
        proposed_pressure = np.divide(
            demand,
            proposed_supply,
            out=np.full_like(proposed_supply, np.inf),
            where=proposed_supply > 0,
        )
        proposed_pressure[(demand == 0) & (proposed_supply > 0)] = 0.0
        pressure_summary = pressure_metrics(proposed_pressure, demand)
        proposed_distance = np.minimum(current_distance, distances)
        distance_summary = accessibility_metrics(proposed_distance, demand, boroughs)
        rows.append({
            "candidate_index": idx,
            "weighted_catchment_predicted_demand": weighted_catchment_demand,
            "predicted_demand_within_10km": float(demand[distances <= 10_000].sum()),
            **pressure_summary,
            **distance_summary,
        })
    return pd.DataFrame(rows).set_index("candidate_index")


def locate_new_hospitals(
    opportunity_grid: gpd.GeoDataFrame,
    capacity: pd.DataFrame,
    grid_xy: np.ndarray,
    hospital_xy: np.ndarray,
    grid_hospital_distances: np.ndarray,
    boundary: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    candidates = _candidate_grid(opportunity_grid, boundary, hospital_xy)
    demand = opportunity_grid["predicted_demand"].to_numpy(dtype=float)
    boroughs = opportunity_grid["cvegeo"].astype(str).to_numpy()
    beds = capacity["emergency_beds"].fillna(0).to_numpy(dtype=float)
    base_supply, base_pressure, _ = spatial_bed_pressure(
        grid_hospital_distances, demand, beds
    )
    base_pressure_metrics = pressure_metrics(base_pressure, demand)
    base_distance = grid_hospital_distances.min(axis=1)
    base_distance_metrics = accessibility_metrics(base_distance, demand, boroughs)
    median_beds = float(capacity["emergency_beds"].median())
    first_scores = _score_pressure_candidates(
        candidates, grid_xy, demand, base_supply, base_distance, boroughs, median_beds
    )

    def select(scores: pd.DataFrame) -> int:
        return int(scores.sort_values([
            "weighted_p90_pressure", "weighted_mean_pressure", "weighted_mean_distance_km"
        ]).index[0])

    first_index = select(first_scores)
    first_point = candidates.loc[first_index].geometry
    first_distance = np.hypot(
        grid_xy[:, 0] - first_point.x, grid_xy[:, 1] - first_point.y
    )
    first_kernel = distance_decay(first_distance[:, None])[:, 0]
    after_first_supply = base_supply + first_kernel * (median_beds / first_kernel.sum())
    pressure_after_first = np.divide(
        demand,
        after_first_supply,
        out=np.full_like(after_first_supply, np.inf),
        where=after_first_supply > 0,
    )
    pressure_after_first[(demand == 0) & (after_first_supply > 0)] = 0.0
    after_first_distance = np.minimum(base_distance, first_distance)
    remaining = candidates[
        (candidates.index != first_index)
        & (candidates.geometry.distance(first_point) >= 2_000)
    ]
    second_scores = _score_pressure_candidates(
        remaining,
        grid_xy,
        demand,
        after_first_supply,
        after_first_distance,
        boroughs,
        median_beds,
    )
    second_index = select(second_scores)

    median_physicians = float(capacity["emergency_physicians"].median())
    median_nurses = float(capacity["contact_nurses"].median())
    rows = []
    specifications = [
        ("pressure_access_1", 1, first_index, first_scores.loc[first_index], base_pressure_metrics, base_distance_metrics),
        (
            "pressure_access_2", 2, second_index, second_scores.loc[second_index],
            pressure_metrics(pressure_after_first, demand),
            accessibility_metrics(after_first_distance, demand, boroughs),
        ),
    ]
    for criterion, order, idx, score, pressure_before, distance_before in specifications:
        point = gpd.GeoSeries([candidates.loc[idx].geometry], crs=32614).to_crs(4326).iloc[0]
        rows.append({
            "criterion": criterion,
            "recommended_order": order,
            "grid_id": candidates.loc[idx, "grid_id"],
            "geo_id": candidates.loc[idx, "cvegeo"],
            "alcaldia": candidates.loc[idx, "nomgeo"],
            "latitude": point.y,
            "longitude": point.x,
            "grid_resolution_m": 1_000,
            "minimum_existing_distance_m": 2_000,
            "planned_emergency_beds_scenario": median_beds,
            "planned_emergency_physicians_scenario": median_physicians,
            "planned_contact_nurses_scenario": median_nurses,
            "weighted_catchment_predicted_demand": score["weighted_catchment_predicted_demand"],
            "predicted_demand_within_10km": score["predicted_demand_within_10km"],
            "weighted_p90_pressure_before": pressure_before["weighted_p90_pressure"],
            "weighted_p90_pressure_after": score["weighted_p90_pressure"],
            "weighted_p90_pressure_reduction_pct": 1 - score["weighted_p90_pressure"] / pressure_before["weighted_p90_pressure"],
            "weighted_mean_pressure_before": pressure_before["weighted_mean_pressure"],
            "weighted_mean_pressure_after": score["weighted_mean_pressure"],
            "weighted_mean_pressure_reduction_pct": 1 - score["weighted_mean_pressure"] / pressure_before["weighted_mean_pressure"],
            "weighted_mean_distance_before_km": distance_before["weighted_mean_distance_km"],
            "weighted_mean_distance_after_km": score["weighted_mean_distance_km"],
            "worst_borough_p90_before_km": distance_before["worst_borough_p90_km"],
            "worst_borough_p90_after_km": score["worst_borough_p90_km"],
            "risk_share_beyond_5km_before": distance_before["risk_share_beyond_5km"],
            "risk_share_beyond_5km_after": score["risk_share_beyond_5km"],
        })
    recommendations = pd.DataFrame(rows)
    recommendations.to_csv(TABLES / "cdmx_new_hospital_candidates.csv", index=False)
    return recommendations, candidates


def make_figures(
    pressure: pd.DataFrame,
    opportunity_grid: gpd.GeoDataFrame,
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

    base = boundary.to_crs(32614)
    hospital_gdf = gpd.GeoDataFrame(
        capacity, geometry=gpd.points_from_xy(capacity["longitud"], capacity["latitud"]), crs=4326,
    ).to_crs(32614)
    rec = gpd.GeoDataFrame(
        recommendations, geometry=gpd.points_from_xy(recommendations["longitude"], recommendations["latitude"]), crs=4326,
    ).to_crs(32614)
    mapped = opportunity_grid.copy()
    mapped.loc[~np.isfinite(mapped["relative_pressure_index"]), "relative_pressure_index"] = np.nan
    finite = mapped["relative_pressure_index"].dropna().to_numpy()
    vmin = min(float(np.quantile(finite, .10)), .95)
    vmax = max(float(np.quantile(finite, .95)), 1.05)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(10, 10))
    mapped.plot(
        ax=ax,
        column="relative_pressure_index",
        cmap="RdYlGn_r",
        norm=norm,
        linewidth=.12,
        edgecolor="#FFFFFF80",
        legend=True,
        legend_kwds={
            "label": "Presión relativa: demanda predicha / camas de urgencias accesibles",
            "shrink": .72,
        },
        missing_kwds={"color": "#D8D8D8", "label": "Sin oferta accesible a 10 km"},
    )
    base.boundary.plot(ax=ax, color="#4D5966", linewidth=.55)
    hospital_gdf.plot(
        ax=ax,
        color="#173B57",
        marker="^",
        markersize=18 + np.sqrt(capacity["emergency_beds"].clip(lower=0).to_numpy()) * 4,
        edgecolor="white",
        linewidth=.35,
        label="Hospital actual; tamaño según camas de urgencias",
    )
    rec.plot(
        ax=ax,
        color="#6A1B9A",
        marker="*",
        markersize=220,
        edgecolor="white",
        linewidth=.7,
        label="Nueva clínica recomendada por presión",
    )
    for row in rec.itertuples():
        ax.annotate(
            f"{int(row.recommended_order)}. {row.alcaldia}",
            (row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )
    ax.set_title(
        "Opportunity Atlas de urgencias: demanda vial futura frente a camas accesibles\n"
        "cuadrícula de 1 km; verde = menor presión, rojo = mayor presión",
        fontsize=13,
        weight="bold",
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "cdmx_opportunity_atlas_pressure.png", dpi=240, bbox_inches="tight")
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
    hospital_pressure, _, _ = assign_hospital_pressure(points, capacity)
    boundary = gpd.read_file("data/external/cdmx_alcaldias_inegi.geojson")
    opportunity_grid, grid_xy, hospital_xy, grid_hospital_distances = build_opportunity_grid(
        points, capacity, boundary
    )
    recommendations, _ = locate_new_hospitals(
        opportunity_grid,
        capacity,
        grid_xy,
        hospital_xy,
        grid_hospital_distances,
        boundary,
    )
    make_figures(pressure, opportunity_grid, capacity, recommendations, boundary)
    logger.info(
        "Integrados %d establecimientos SINERHIAS, %d hospitales y %d ubicaciones candidatas",
        len(resources), len(hospital_pressure), len(recommendations),
    )


if __name__ == "__main__":
    main()
