"""Clean configured ATUS editions and write immutable-derived Parquet files."""

from pathlib import Path
from siniestros_riesgo.annual import clean_annual_file
from siniestros_riesgo.atus import clean_atus, read_table
from siniestros_riesgo.config import PROJECT_CONFIG, PROCESSED, RAW, load_yaml, setup_logging

logger = setup_logging("prepare_atus")
config = load_yaml(PROJECT_CONFIG)
mapping = config["columns"]
PROCESSED.mkdir(parents=True, exist_ok=True)
for year in config["years"]:
    annual = RAW / "conjunto_de_datos_atus_anual_csv" / "conjunto_de_datos" / f"atus_anual_{year}.csv"
    if not annual.exists():
        raise SystemExit(f"Falta {annual}")
    cleaned = clean_annual_file(annual, year, config["annual_columns"])
    output = PROCESSED / f"atus_events_{year}.parquet"
    cleaned.to_parquet(output, index=False)
    logger.info("Escrito %s (%d registros)", output, len(cleaned))

# Mantener una colección georreferenciada separada; no reemplaza el panel nacional.
for year in range(2020, 2025):
    matches = sorted(path for path in RAW.glob(f"**/*{year}*.csv") if "conjunto_de_datos" in path.parts and "atus_anual" not in path.name.lower())
    if len(matches) == 1:
        cleaned = clean_atus(read_table(matches[0]), mapping, year)
        cleaned.to_parquet(PROCESSED / f"atus_georef_{year}.parquet", index=False)
        logger.info("Georreferenciación %d guardada (%d registros)", year, len(cleaned))
