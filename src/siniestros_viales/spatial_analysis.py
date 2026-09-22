"""Municipal aggregation and guarded spatial analysis hooks."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .utils import get_logger

LOG = get_logger(__name__)


def aggregate_municipal(frame: pd.DataFrame, population: pd.DataFrame | None = None, vehicles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate event counts; add exposure rates only with verified denominators."""
    keys = ["year", "state", "municipality"]
    if missing := set(keys + ["victimas", "fatal"]) - set(frame.columns):
        raise ValueError(f"Missing aggregation fields: {sorted(missing)}")
    result = frame.groupby(keys, dropna=False).agg(siniestros=("victimas", "size"), siniestros_con_victimas=("victimas", "sum"), siniestros_fatales=("fatal", "sum")).reset_index()
    result["proporcion_con_victimas"] = result["siniestros_con_victimas"] / result["siniestros"]
    for denominator, source, scale in (("population", population, 100_000), ("vehicles", vehicles, 10_000)):
        if source is None:
            continue
        needed = set(keys + [denominator])
        if needed - set(source.columns):
            raise ValueError(f"Denominator table needs {sorted(needed)}")
        if source.duplicated(keys).any():
            raise ValueError(f"Duplicate denominator keys: {denominator}")
        result = result.merge(source[keys + [denominator]], on=keys, how="left", validate="one_to_one")
        result[f"victim_crash_rate_per_{scale}_{denominator}"] = (result["siniestros_con_victimas"] / result[denominator].where(result[denominator] > 0)) * scale
    return result


def join_municipal_geometries(aggregates: pd.DataFrame, boundary_path: Path, boundary_keys: list[str]) -> gpd.GeoDataFrame:
    """Join verified municipal boundaries using explicit key columns."""
    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)
    boundaries = gpd.read_file(boundary_path)
    if any(key not in boundaries or key not in aggregates for key in boundary_keys):
        raise ValueError("Join keys must exist in both tables")
    return boundaries.merge(aggregates, on=boundary_keys, how="inner", validate="one_to_many")


def getis_ord_gi_star(geodata: gpd.GeoDataFrame, value_column: str, weights: object) -> object:
    """Compute Gi* after an analyst has explicitly selected spatial weights."""
    from esda.getisord import G_Local

    if value_column not in geodata:
        raise ValueError(f"Missing spatial variable: {value_column}")
    if geodata[value_column].isna().any():
        raise ValueError("Gi* input has missing values; resolve them before analysis")
    return G_Local(geodata[value_column].to_numpy(), weights, star=True)
