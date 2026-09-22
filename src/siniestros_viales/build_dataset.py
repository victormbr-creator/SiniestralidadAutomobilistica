"""Assemble clean years and report dataset quality."""

from pathlib import Path
from collections.abc import Sequence

import pandas as pd

from .config import IDENTIFIERS, PROCESSED, REPORTS
from .utils import get_logger, write_json

LOG = get_logger(__name__)


def build_dataset(paths: Sequence[Path], output: Path = PROCESSED / "atus_dataset.parquet") -> pd.DataFrame:
    """Concatenate years, check types/duplicates, and save Parquet plus reports."""
    if not paths:
        raise ValueError("No cleaned years supplied")
    frames = [pd.read_parquet(path) for path in paths]
    dataset = pd.concat(frames, ignore_index=True)
    required = {"year", "deaths", "injuries", "victimas", "fatal"}
    if missing := required - set(dataset.columns):
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")
    for column in ("year", "deaths", "injuries", "victimas", "fatal"):
        if not pd.api.types.is_integer_dtype(dataset[column]):
            raise TypeError(f"Expected integer column: {column}")
    if dataset["year"].isna().any():
        raise ValueError("Missing year values")
    duplicate_rows = int(dataset.duplicated().sum())
    duplicate_ids = int(dataset.duplicated(subset=["year", "event_id"]).sum()) if "event_id" in dataset else None
    if duplicate_rows or duplicate_ids:
        raise ValueError(f"Duplicates found: rows={duplicate_rows}, year/event_id={duplicate_ids}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    missingness = dataset.isna().agg(["sum", "mean"]).T.rename(columns={"sum": "missing_count", "mean": "missing_fraction"})
    missingness.to_csv(REPORTS / "missingness.csv", index_label="column")
    identifiers = [column for column in IDENTIFIERS if column in dataset]
    predictors = [column for column in dataset if column not in identifiers and column not in {"deaths", "injuries", "fatal", "victimas"}]
    write_json(REPORTS / "dataset_schema.json", {"identifiers": identifiers, "candidate_predictors": predictors, "dtypes": {name: str(dtype) for name, dtype in dataset.dtypes.items()}, "rows": len(dataset)})
    LOG.info("Wrote %s (%d rows)", output, len(dataset))
    return dataset
