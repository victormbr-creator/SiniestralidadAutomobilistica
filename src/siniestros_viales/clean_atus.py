"""Schema-aware ATUS cleaning without assumptions about source column names."""

import re
import unicodedata
from pathlib import Path
from collections.abc import Mapping

import pandas as pd

from .config import ATUS_MAPPING, INTERIM
from .utils import get_logger, read_json

LOG = get_logger(__name__)


def snake_case(name: str) -> str:
    """Normalize a column label to lowercase ASCII snake_case."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def load_mapping(path: Path = ATUS_MAPPING) -> dict[str, str]:
    """Load confirmed canonical-to-source mappings; ignore unset placeholders."""
    return {key: value for key, value in read_json(path)["mapping"].items() if value}


def read_atus(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet source; CSV uses default UTF-8 pending source review."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported ATUS format: {path.suffix}")


def clean_year(frame: pd.DataFrame, year: int, mapping: Mapping[str, str]) -> pd.DataFrame:
    """Normalize columns, validate deaths/injuries, and build binary targets."""
    if frame.empty:
        raise ValueError("ATUS input is empty")
    normalized = {column: snake_case(column) for column in frame.columns}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Column names collide after snake_case normalization")
    result = frame.rename(columns=normalized).copy()
    for canonical, source in mapping.items():
        source_normalized = snake_case(source)
        if source_normalized not in result.columns:
            raise ValueError(f"Mapped source column missing: {source}")
        if canonical != source_normalized and canonical in result.columns:
            raise ValueError(f"Mapping would overwrite existing column: {canonical}")
        result = result.rename(columns={source_normalized: canonical})
    missing = {"deaths", "injuries"} - set(result.columns)
    if missing:
        raise ValueError(f"Configure confirmed ATUS mapping for: {sorted(missing)}")
    for column in ("deaths", "injuries"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain nonnegative integer counts without missing values")
        result[column] = values.astype("int64")
    result["year"] = int(year)
    result["victimas"] = ((result["deaths"] + result["injuries"]) > 0).astype("int8")
    result["fatal"] = (result["deaths"] > 0).astype("int8")
    return result


def clean_files(files_by_year: Mapping[int, Path], mapping: Mapping[str, str], output_dir: Path = INTERIM) -> list[Path]:
    """Clean each configured year and write one Parquet file per year."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for year, path in sorted(files_by_year.items()):
        cleaned = clean_year(read_atus(path), year, mapping)
        output = output_dir / f"atus_{year}.parquet"
        cleaned.to_parquet(output, index=False)
        LOG.info("Wrote %s (%d rows)", output, len(cleaned))
        paths.append(output)
    return paths
