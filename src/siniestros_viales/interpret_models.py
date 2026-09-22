"""Save coefficient, importance and optional XGBoost SHAP outputs."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES, MODELS, TABLES
from .utils import get_logger, read_json

LOG = get_logger(__name__)


def save_interpretations(model_dir: Path = MODELS, table_dir: Path = TABLES, figure_dir: Path = FIGURES, shap_sample: int = 200) -> None:
    """Export fitted feature effects; cap SHAP rows for a lightweight first review."""
    manifest = read_json(model_dir / "training_manifest.json")
    holdout = pd.read_parquet(model_dir / "test_holdout.parquet")
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logit", "random_forest", "xgboost"):
        pipeline = joblib.load(model_dir / f"{name}.joblib")
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        model = pipeline.named_steps["model"]
        values = model.coef_[0] if name == "logit" else model.feature_importances_
        label = "coefficient" if name == "logit" else "importance"
        pd.DataFrame({"feature": feature_names, label: values}).sort_values(label, ascending=False).to_csv(table_dir / f"{name}_importance.csv", index=False)
    if shap_sample > 0:
        import shap

        pipeline = joblib.load(model_dir / "xgboost.joblib")
        sample = holdout[manifest["predictors"]].sample(n=min(shap_sample, len(holdout)), random_state=manifest["seed"])
        transformed = pipeline.named_steps["preprocess"].transform(sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        explanation = explainer(transformed)
        shap.summary_plot(explanation.values, transformed, feature_names=pipeline.named_steps["preprocess"].get_feature_names_out(), show=False)
        plt.savefig(figure_dir / "xgboost_shap.png", dpi=150, bbox_inches="tight")
        plt.close()
    LOG.info("Saved interpretation outputs")
