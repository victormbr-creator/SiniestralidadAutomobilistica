"""Run annual Gi* after verified geometries and analysis variables exist."""

import geopandas as gpd
from siniestros_riesgo.config import PROCESSED, TABLES, setup_logging
from siniestros_riesgo.spatial import getis_ord_by_year, hotspot_persistence

logger = setup_logging("run_spatial")
path = PROCESSED / "municipal_spatial_analysis.gpkg"
if not path.exists():
    raise SystemExit("Falta data/processed/municipal_spatial_analysis.gpkg con tasa suavizada y geometrías verificadas")
data = gpd.read_file(path)
required = {"geo_id", "year", "eb_rate", "geometry"}
if missing := required - set(data.columns):
    raise SystemExit(f"Faltan variables espaciales: {sorted(missing)}")
results = getis_ord_by_year(data, "eb_rate", "geo_id")
TABLES.mkdir(parents=True, exist_ok=True)
results.to_csv(TABLES / "municipal_gistar.csv", index=False)
hotspot_persistence(results, "geo_id").to_csv(TABLES / "municipal_hotspot_persistence.csv", index=False)
logger.info("Diagnóstico espacial guardado")
