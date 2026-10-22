"""Fit comparable models after analyst-approved features are documented."""

import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from siniestros_riesgo.config import MODELS, PROCESSED, PROJECT_CONFIG, load_yaml, set_seed, setup_logging
from siniestros_riesgo.models import TrailingMeanBaseline, count_models, preprocessor

logger = setup_logging("train_models")
config = load_yaml(PROJECT_CONFIG)
set_seed(config["seed"])
path = PROCESSED / "modeling_panel.parquet"
if not path.exists():
    raise SystemExit("Falta data/processed/modeling_panel.parquet; incorpore predictores y documente su temporalidad")
data = pd.read_parquet(path)
manifest_path = PROCESSED / "approved_features.json"
if not manifest_path.exists():
    raise SystemExit("Falta data/processed/approved_features.json con la lista revisada de predictores")
features = json.loads(manifest_path.read_text(encoding="utf-8"))["features"]
for forbidden in ("victim_crashes", "any_victim_crash", "fatalities"):
    if forbidden in features:
        raise SystemExit(f"Variable de fuga temporal/resultado en predictores: {forbidden}")
train = data[data.year.isin(config["train_years"])].copy()
test = data[data.year.isin(config["test_years"])].copy()
if train.empty or test.empty:
    raise SystemExit("El panel debe contener entrenamiento 2018-2022 y test 2023-2024")
MODELS.mkdir(parents=True, exist_ok=True)
joblib.dump(TrailingMeanBaseline().fit(train[features], train.victim_crashes), MODELS / "baseline.joblib")
for name, estimator in count_models(config["seed"]).items():
    pipeline = Pipeline([("preprocess", preprocessor(train, features)), ("model", estimator)])
    pipeline.fit(train[features], train.victim_crashes)
    joblib.dump(pipeline, MODELS / f"{name}.joblib")
(MODELS / "manifest.json").write_text(json.dumps({"features": features, "train_years": config["train_years"], "test_years": config["test_years"], "seed": config["seed"], "selection_status": "TODO: validación temporal y leave-one-state-out antes del test final"}, indent=2, ensure_ascii=False), encoding="utf-8")
logger.info("Modelos de conteo ajustados; binomial negativa requiere región y rezago espacial validados")
