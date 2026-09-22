"""Evaluate persisted models on the untouched temporal holdout."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .config import FIGURES, MODELS, TABLES
from .utils import get_logger, read_json, write_json

LOG = get_logger(__name__)


def calculate_metrics(y_true: pd.Series, probabilities: pd.Series | list[float], threshold: float = 0.5) -> dict:
    """Calculate binary discrimination and classification metrics."""
    predictions = [int(value >= threshold) for value in probabilities]
    results = {
        "roc_auc": roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else None,
        "average_precision": average_precision_score(y_true, probabilities) if y_true.nunique() == 2 else None,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }
    return results


def evaluate_saved(model_dir: Path = MODELS, table_dir: Path = TABLES, figure_dir: Path = FIGURES) -> dict[str, dict]:
    """Read saved test data, score each model, and save tables and calibration plots."""
    manifest = read_json(model_dir / "training_manifest.json")
    holdout = pd.read_parquet(model_dir / "test_holdout.parquet")
    target, columns = manifest["target"], manifest["predictors"]
    y_true = holdout[target].astype(int)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("logit", "random_forest", "xgboost"):
        pipeline = joblib.load(model_dir / f"{name}.joblib")
        probabilities = pipeline.predict_proba(holdout[columns])[:, 1]
        results[name] = calculate_metrics(y_true, probabilities)
        pd.DataFrame({"year": holdout["year"], "actual": y_true, "probability": probabilities}).to_csv(table_dir / f"{name}_predictions.csv", index=False)
        if y_true.nunique() == 2:
            observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.plot(predicted, observed, marker="o")
            ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada", title=f"Calibración: {name}")
            fig.savefig(figure_dir / f"calibration_{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    write_json(table_dir / "metrics.json", results)
    pd.DataFrame({name: {key: value for key, value in metrics.items() if key != "confusion_matrix"} for name, metrics in results.items()}).T.to_csv(table_dir / "metrics.csv", index_label="model")
    LOG.info("Saved evaluation for %d models", len(results))
    return results
