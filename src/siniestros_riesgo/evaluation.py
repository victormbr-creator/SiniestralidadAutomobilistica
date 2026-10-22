"""Metrics, capture rates and clustered uncertainty."""

from collections.abc import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_poisson_deviance


def capture_at_share(actual: np.ndarray, predicted: np.ndarray, share: float) -> float:
    """Share of observed events captured by the highest predicted units."""
    if not 0 < share <= 1:
        raise ValueError("share debe estar en (0, 1]")
    total = actual.sum()
    if total <= 0:
        return float("nan")
    n = max(1, int(np.ceil(len(actual) * share)))
    selected = np.argsort(-predicted, kind="stable")[:n]
    return float(actual[selected].sum() / total)


def count_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Evaluate nonnegative count predictions."""
    prediction = np.clip(np.asarray(predicted, dtype=float), 1e-9, None)
    observed = np.asarray(actual, dtype=float)
    return {
        "mae": float(mean_absolute_error(observed, prediction)),
        "poisson_deviance": float(mean_poisson_deviance(observed, prediction)),
        "capture_top_5": capture_at_share(observed, prediction, 0.05),
        "capture_top_10": capture_at_share(observed, prediction, 0.10),
    }


def binary_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Evaluate probability rankings and calibration loss."""
    return {
        "pr_auc": float(average_precision_score(actual, probability)),
        "brier": float(brier_score_loss(actual, probability)),
    }


def block_bootstrap_difference(frame: pd.DataFrame, metric: Callable[[np.ndarray, np.ndarray], float], model_column: str, baseline_column: str, repetitions: int = 1000, seed: int = 42) -> dict[str, float]:
    """Resample municipalities with replacement and compare a model to baseline."""
    frame = frame.reset_index(drop=True)
    municipalities = frame["geo_id"].drop_duplicates().to_numpy()
    if len(municipalities) < 2:
        raise ValueError("Se requieren al menos dos municipios para bootstrap por bloques")
    rng = np.random.default_rng(seed)
    differences = []
    groups = {code: group.index.to_numpy() for code, group in frame.groupby("geo_id", sort=False)}
    actual_all = frame["victim_crashes"].to_numpy()
    model_all = frame[model_column].to_numpy()
    baseline_all = frame[baseline_column].to_numpy()
    for _ in range(repetitions):
        sampled = rng.choice(municipalities, size=len(municipalities), replace=True)
        indices = np.concatenate([groups[code] for code in sampled])
        actual = actual_all[indices]
        differences.append(metric(actual, model_all[indices]) - metric(actual, baseline_all[indices]))
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {"difference": float(np.mean(differences)), "ci_lower": float(lower), "ci_upper": float(upper), "repetitions": repetitions}
