"""Risk coverage and maximum-coverage location planning."""

import numpy as np
import pandas as pd


def risk_beyond_threshold(travel: pd.DataFrame, threshold_minutes: float) -> float:
    """Compute predicted risk beyond a verified travel-time threshold."""
    required = {"cell_id", "predicted_risk", "nearest_existing_minutes"}
    if missing := required - set(travel):
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    total = travel["predicted_risk"].sum()
    if total <= 0:
        return float("nan")
    return float(travel.loc[travel["nearest_existing_minutes"] > threshold_minutes, "predicted_risk"].sum() / total)


def greedy_maximum_coverage(incidence: pd.DataFrame, risk: pd.Series, p_max: int = 10) -> pd.DataFrame:
    """Greedy transparent approximation for p=1..p_max candidate bases."""
    if not incidence.index.equals(risk.index):
        raise ValueError("La matriz de cobertura y el riesgo deben compartir índice de celdas")
    covered = np.zeros(len(incidence), dtype=bool)
    remaining = list(incidence.columns)
    rows = []
    selected: list[str] = []
    for p in range(1, min(p_max, len(remaining)) + 1):
        gains = {candidate: float(risk[incidence[candidate].astype(bool) & ~covered].sum()) for candidate in remaining}
        choice = max(gains, key=gains.get)
        selected.append(str(choice))
        covered |= incidence[choice].astype(bool).to_numpy()
        remaining.remove(choice)
        rows.append({"p": p, "selected_candidates": "|".join(selected), "risk_covered": float(risk[covered].sum()), "method": "greedy_approximation"})
    return pd.DataFrame(rows)
