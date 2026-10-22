"""Generate reproducible ATUS coverage, quality tables and descriptive figures."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from siniestros_riesgo.config import FIGURES, PROCESSED, TABLES, PROJECT_CONFIG, load_yaml, setup_logging

logger = setup_logging("analyze_quality")
config = load_yaml(PROJECT_CONFIG)
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
frames = [pd.read_parquet(PROCESSED / f"atus_events_{year}.parquet") for year in config["years"]]
events = pd.concat(frames, ignore_index=True)
coverage = events.groupby(["geo_id", "state_code", "year"], as_index=False).agg(records=("event_id", "size"), victim_crashes=("victimas", "sum"), fatalities=("deaths_total", "sum"), injuries=("injuries_total", "sum"))
pivot = coverage.assign(reported=True).pivot(index="geo_id", columns="year", values="reported").fillna(False)
balanced = pivot.all(axis=1)
coverage["balanced_2018_2024"] = coverage["geo_id"].isin(balanced[balanced].index)
coverage.to_csv(TABLES / "coverage_municipality_year.csv", index=False)
totals = events.groupby("year", as_index=False).agg(records=("event_id", "size"), victim_crashes=("victimas", "sum"), deaths=("deaths_total", "sum"), injuries=("injuries_total", "sum"), municipalities=("geo_id", "nunique"))
totals["victim_crash_share"] = totals["victim_crashes"] / totals["records"]
totals.to_csv(TABLES / "annual_totals.csv", index=False)
quality_rows = []
for year, frame in zip(config["years"], frames):
    duplicate_subset = [c for c in frame.columns if c != "event_id"]
    quality_rows.append({"year": year, "rows": len(frame), "indistinguishable_rows_without_source_id": int(frame.duplicated(duplicate_subset).sum()), "invalid_month": int((~frame.month.between(1,12)).sum()), "invalid_geo_id": int((~frame.geo_id.str.fullmatch(r'\d{5}')).sum()), "missing_victim_fields": int(frame[["deaths_total","injuries_total"]].isna().any(axis=1).sum()), "certified_zero_rows": int(frame.report_status.eq("certified_zero").sum())})
pd.DataFrame(quality_rows).to_csv(TABLES / "quality_summary.csv", index=False)
summary = {"balanced_municipalities": int(balanced.sum()), "municipalities_observed_any_year": int(len(balanced)), "excluded_municipalities": int((~balanced).sum()), "population_share_outside_balanced_sample": None, "population_note": "No calculable: falta población municipal Censo 2020", "certificate_note": "La publicación anual de julio de 2026 eliminó certificados cero de 2018-2024."}
(TABLES / "sample_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(totals.year, totals.records, marker="o", label="Todos los siniestros")
ax.plot(totals.year, totals.victim_crashes, marker="o", label="Con víctimas")
ax.set(title="Registros ATUS por año", xlabel="Año", ylabel="Número de siniestros")
ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(FIGURES / "annual_crash_counts.png", dpi=220); plt.close(fig)
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(totals.year, totals.victim_crash_share * 100, marker="o", color="#9C2F2F")
ax.set(title="Proporción de siniestros con víctimas", xlabel="Año", ylabel="Porcentaje")
ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "annual_victim_share.png", dpi=220); plt.close(fig)
logger.info("Calidad analizada: %d registros, %d municipios balanceados", len(events), balanced.sum())
