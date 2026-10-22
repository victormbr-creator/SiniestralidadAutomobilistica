"""Evaluate CLUES coverage only with verified network travel times."""

import pandas as pd
from siniestros_riesgo.config import PROCESSED, TABLES, setup_logging
from siniestros_riesgo.coverage import risk_beyond_threshold

logger = setup_logging("run_coverage")
path = PROCESSED / "clues_network_travel_times.parquet"
if not path.exists():
    raise SystemExit("Falta clues_network_travel_times.parquet; no se convertirán distancias a minutos sin supuesto documentado")
data = pd.read_parquet(path)
rows = [{"threshold_minutes": threshold, "risk_beyond_share": risk_beyond_threshold(data, threshold), "scenario": "CLUES"} for threshold in (10, 15)]
TABLES.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(TABLES / "clues_risk_coverage.csv", index=False)
logger.info("Cobertura CLUES guardada")
