"""Build the balanced municipality-month panel and frozen holdout features."""

import pandas as pd
from siniestros_riesgo.config import PROJECT_CONFIG, PROCESSED, load_yaml, setup_logging
from siniestros_riesgo.panel import build_monthly_panel, freeze_post_2022_features

logger = setup_logging("build_panel")
config = load_yaml(PROJECT_CONFIG)
paths = [PROCESSED / f"atus_events_{year}.parquet" for year in config["years"]]
missing = [path.name for path in paths if not path.exists()]
if missing:
    raise SystemExit(f"Faltan derivados: {missing}")
events = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
panel = build_monthly_panel(events, config["years"])
panel.to_parquet(PROCESSED / "municipality_month_panel.parquet", index=False)
panel["month_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * panel["month"] / 12)
panel["month_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * panel["month"] / 12)
panel["time_index"] = (panel["year"] - min(config["years"])) * 12 + panel["month"] - 1
frozen = freeze_post_2022_features(panel)
frozen.to_parquet(PROCESSED / "municipality_month_panel_frozen.parquet", index=False)
panel.to_parquet(PROCESSED / "modeling_panel.parquet", index=False)
frozen.to_parquet(PROCESSED / "modeling_panel_frozen.parquet", index=False)
(PROCESSED / "approved_features.json").write_text('{"features":["lag_1","mean_lag_12","month_sin","month_cos","time_index","state_code"]}', encoding="utf-8")
logger.info("Panel creado: %d filas, %d municipios", len(panel), panel.geo_id.nunique())
