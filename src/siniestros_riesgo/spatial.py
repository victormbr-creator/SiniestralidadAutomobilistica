"""Spatial weights, Gi* inference and persistence summaries."""

import geopandas as gpd
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def build_weights(geodata: gpd.GeoDataFrame, k: int = 8):
    """Use Queen contiguity, falling back to k-nearest neighbors for islands."""
    from libpysal.weights import KNN, Queen
    weights = Queen.from_dataframe(geodata, use_index=True)
    if weights.islands:
        weights = KNN.from_dataframe(geodata, k=min(k, max(1, len(geodata) - 1)), use_index=True)
    weights.transform = "R"
    return weights


def getis_ord_by_year(geodata: gpd.GeoDataFrame, value: str, id_column: str, seed: int = 42) -> pd.DataFrame:
    """Calculate annual Gi*, BH-adjusted p-values and hot/cold labels."""
    from esda.getisord import G_Local
    rows = []
    for year, subset in geodata.groupby("year"):
        subset = subset.reset_index(drop=True)
        if subset[value].isna().any():
            raise ValueError(f"Valores faltantes en {value}, año {year}")
        weights = build_weights(subset)
        result = G_Local(subset[value].to_numpy(), weights, star=True, permutations=999, seed=seed)
        adjusted = multipletests(result.p_sim, method="fdr_bh")[1]
        labels = np.where((adjusted < 0.05) & (result.Zs > 0), "hot", np.where((adjusted < 0.05) & (result.Zs < 0), "cold", "not_significant"))
        rows.append(pd.DataFrame({id_column: subset[id_column], "year": year, "gi_z": result.Zs, "p_raw": result.p_sim, "p_bh": adjusted, "cluster": labels}))
    return pd.concat(rows, ignore_index=True)


def hotspot_persistence(results: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Count annual significant hot spots by spatial unit."""
    return results.assign(is_hot=results["cluster"].eq("hot")).groupby(id_column, as_index=False).agg(hot_years=("is_hot", "sum"), years_observed=("year", "nunique"))
