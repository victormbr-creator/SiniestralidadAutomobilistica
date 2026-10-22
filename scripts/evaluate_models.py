"""Evaluate persisted models on the untouched 2023-2024 holdout."""

import json
import joblib
import pandas as pd
from siniestros_riesgo.config import MODELS, PROCESSED, TABLES, setup_logging
from siniestros_riesgo.evaluation import count_metrics

logger = setup_logging("evaluate_models")
panel_path = PROCESSED / "modeling_panel_frozen.parquet"
manifest_path = MODELS / "manifest.json"
if not panel_path.exists() or not manifest_path.exists():
    raise SystemExit("Faltan panel congelado o manifiesto de modelos")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("selection_status", "").startswith("TODO"):
    raise SystemExit("No se permite abrir el test final antes de documentar y completar la selección interna")
test = pd.read_parquet(panel_path)
test = test[test.year.isin(manifest["test_years"])].copy()
rows = []
for model_path in sorted(MODELS.glob("*.joblib")):
    model = joblib.load(model_path)
    prediction = model.predict(test[manifest["features"]])
    rows.append({"model": model_path.stem, **count_metrics(test.victim_crashes.to_numpy(), prediction)})
TABLES.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(TABLES / "test_count_metrics.csv", index=False)
logger.info("Evaluación final guardada")
