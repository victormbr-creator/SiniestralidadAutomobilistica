"""Aggregate official DGIS emergency records for the 34 CDMX hospitals."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from siniestros_riesgo.config import FIGURES, PROCESSED, RAW, TABLES, setup_logging
from siniestros_riesgo.emergency_flow import aggregate_archive


logger = setup_logging("prepare_emergency_flow")
URGENCY_DIR = RAW / "sinba" / "urgencias"
CATALOG_ARCHIVE = URGENCY_DIR / "catalogos_urgencias_2023_2025.zip"
YEARS = range(2018, 2026)


def load_crosswalk(target_clues: set[str]) -> dict[str, str]:
    with zipfile.ZipFile(CATALOG_ARCHIVE) as archive:
        member = next(name for name in archive.namelist() if name.upper().endswith("CATCLUES.CSV"))
        with archive.open(member) as binary:
            catalog = pd.read_csv(io.TextIOWrapper(binary, encoding="latin1"), sep="|", dtype=str)
    selected = catalog[catalog["CLUES"].isin(target_clues)].copy()
    if set(selected["CLUES"]) != target_clues:
        missing = sorted(target_clues - set(selected["CLUES"]))
        raise ValueError(f"CLUES missing from official crosswalk: {missing}")
    mapping = dict(zip(selected["CLUES"], selected["CLUES"]))
    for _, row in selected.iterrows():
        previous = str(row["CLUES ANTERIOR"] or "").strip()
        if previous and previous.lower() != "nan":
            mapping[previous] = row["CLUES"]
    return mapping


def build_pressure(flow: pd.DataFrame, capacity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = flow[flow["year"].eq(2025)].groupby("CLUES", as_index=False).sum(numeric_only=True)
    annual = annual.drop(columns=["year", "month"], errors="ignore")
    hospital = capacity.merge(annual, on="CLUES", how="left", validate="one_to_one")
    count_columns = [
        column for column in annual.columns if column != "CLUES"
    ]
    hospital[count_columns] = hospital[count_columns].fillna(0)
    hospital["qualified_emergency_share"] = (
        hospital["qualified_emergencies"] / hospital["emergency_visits"].replace(0, np.nan)
    )
    hospital["mean_stay_hours"] = hospital["total_stay_hours"] / hospital["known_stay_records"].replace(0, np.nan)
    hospital["mean_bed_stay_hours"] = hospital["bed_stay_hours"] / hospital["bed_stay_records"].replace(0, np.nan)
    hospital["emergency_visits_per_bed"] = (
        hospital["emergency_visits"] / hospital["emergency_beds"].replace(0, np.nan)
    )
    hospital["emergency_visits_per_emergency_physician"] = (
        hospital["emergency_visits"] / hospital["emergency_physicians"].replace(0, np.nan)
    )
    hospital["emergency_visits_per_contact_nurse"] = (
        hospital["emergency_visits"] / hospital["contact_nurses"].replace(0, np.nan)
    )
    hospital["observed_emergency_bed_occupancy_proxy"] = (
        hospital["bed_stay_hours"] / (hospital["emergency_beds"].replace(0, np.nan) * 24 * 365)
    )
    hospital["observed_flow_pressure_rank"] = (
        hospital["emergency_visits_per_bed"].rank(method="min", ascending=False, na_option="bottom")
    ).astype("Int64")
    hospital = hospital.sort_values("observed_flow_pressure_rank")

    borough = hospital.groupby(["geo_id", "alcaldia"], as_index=False).agg(
        emergency_hospitals=("CLUES", "size"),
        emergency_visits=("emergency_visits", "sum"),
        qualified_emergencies=("qualified_emergencies", "sum"),
        accident_poisoning_violence_visits=("accident_poisoning_violence_visits", "sum"),
        hospital_admissions=("hospital_admissions", "sum"),
        transfers=("transfers", "sum"),
        deaths_in_emergency=("deaths_in_emergency", "sum"),
        bed_stay_hours=("bed_stay_hours", "sum"),
        emergency_beds=("emergency_beds", "sum"),
        emergency_physicians=("emergency_physicians", "sum"),
        contact_nurses=("contact_nurses", "sum"),
    )
    borough["emergency_visits_per_bed"] = borough["emergency_visits"] / borough["emergency_beds"].replace(0, np.nan)
    borough["qualified_emergency_share"] = (
        borough["qualified_emergencies"] / borough["emergency_visits"].replace(0, np.nan)
    )
    borough["observed_emergency_bed_occupancy_proxy"] = (
        borough["bed_stay_hours"] / (borough["emergency_beds"].replace(0, np.nan) * 24 * 365)
    )
    borough["observed_flow_pressure_rank"] = (
        borough["emergency_visits_per_bed"].rank(method="min", ascending=False, na_option="bottom")
    ).astype("Int64")
    return hospital, borough.sort_values("observed_flow_pressure_rank")


def add_projected_road_pressure(
    hospital: pd.DataFrame, borough: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep observed emergency flow and projected road demand as parallel screens."""
    future_hospital = pd.read_csv(TABLES / "cdmx_hospital_pressure_future.csv", dtype={"CLUES": "string"})
    hospital_columns = [
        "CLUES", "assigned_predicted_crashes", "assigned_crashes_per_emergency_bed",
        "assigned_crashes_per_emergency_physician", "assigned_crashes_per_contact_nurse",
    ]
    hospital = hospital.merge(
        future_hospital[hospital_columns], on="CLUES", how="left", validate="one_to_one"
    )
    hospital["observed_flow_percentile"] = hospital["emergency_visits_per_bed"].rank(pct=True)
    hospital["projected_road_pressure_percentile"] = hospital["assigned_crashes_per_emergency_bed"].rank(pct=True)
    hospital["dual_pressure_score"] = hospital[
        ["observed_flow_percentile", "projected_road_pressure_percentile"]
    ].mean(axis=1)
    hospital["dual_pressure_priority_rank"] = (
        hospital["dual_pressure_score"].rank(method="min", ascending=False)
    ).astype("Int64")

    future_borough = pd.read_csv(
        TABLES / "cdmx_borough_hospital_pressure_future.csv", dtype={"geo_id": "string"}
    )
    borough_columns = [
        "geo_id", "predicted_horizon_victim_crashes", "predicted_crashes_per_emergency_bed",
        "predicted_crashes_per_emergency_physician", "predicted_crashes_per_contact_nurse",
    ]
    borough = borough.merge(
        future_borough[borough_columns], on="geo_id", how="left", validate="one_to_one"
    )
    borough["observed_flow_percentile"] = borough["emergency_visits_per_bed"].rank(pct=True)
    borough["projected_road_pressure_percentile"] = borough["predicted_crashes_per_emergency_bed"].rank(pct=True)
    borough["dual_pressure_score"] = borough[
        ["observed_flow_percentile", "projected_road_pressure_percentile"]
    ].mean(axis=1)
    borough["dual_pressure_priority_rank"] = (
        borough["dual_pressure_score"].rank(method="min", ascending=False)
    ).astype("Int64")
    return hospital, borough


def make_figure(borough: pd.DataFrame) -> None:
    plot = borough.sort_values("emergency_visits_per_bed")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot["alcaldia"], plot["emergency_visits_per_bed"], color="#19647E")
    ax.set(
        title="Flujo observado de urgencias por cama, 2025",
        xlabel="Atenciones registradas en urgencias por cama de urgencias",
    )
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "cdmx_observed_emergency_flow_pressure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    capacity = pd.read_csv(
        TABLES / "cdmx_emergency_hospital_capacity_2026.csv",
        dtype={"CLUES": "string", "geo_id": "string"},
    )
    crosswalk = load_crosswalk(set(capacity["CLUES"]))
    logger.info("Crosswalk contains %d current and historical CLUES", len(crosswalk))
    frames = []
    coverage = []
    for year in YEARS:
        archive = URGENCY_DIR / f"Urgencias_{year}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"Missing official emergency archive: {archive}")
        frame, stats = aggregate_archive(archive, year, crosswalk)
        frames.append(frame)
        coverage.append(stats)
        logger.info(
            "%d: %s emergency records matched across %d hospitals",
            year, f"{stats['matched_emergency_records']:,}", stats["matched_current_hospitals"],
        )
    flow = pd.concat(frames, ignore_index=True)
    flow.to_parquet(PROCESSED / "cdmx_emergency_flow_monthly_2018_2025.parquet", index=False)
    flow.to_csv(TABLES / "cdmx_emergency_flow_monthly_2018_2025.csv", index=False)
    pd.DataFrame(coverage).to_csv(TABLES / "cdmx_emergency_flow_coverage.csv", index=False)
    hospital, borough = build_pressure(flow, capacity)
    hospital, borough = add_projected_road_pressure(hospital, borough)
    hospital.to_csv(TABLES / "cdmx_hospital_observed_emergency_pressure_2025.csv", index=False)
    borough.to_csv(TABLES / "cdmx_borough_observed_emergency_flow_2025.csv", index=False)
    make_figure(borough)
    logger.info("2025 matched network total: %s", f"{int(borough['emergency_visits'].sum()):,}")


if __name__ == "__main__":
    main()
