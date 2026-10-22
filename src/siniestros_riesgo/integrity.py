"""Input inventory and integrity checks."""

from dataclasses import dataclass
from pathlib import Path
import hashlib

import pandas as pd

from .config import RAW, load_yaml, PROJECT_CONFIG, SOURCES_CONFIG


@dataclass(frozen=True)
class ValidationResult:
    """Machine-readable validation outcome."""
    ok: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


def sha256(path: Path) -> str:
    """Calculate a file hash without modifying the source."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs() -> ValidationResult:
    """Require documented sources, confirmed mappings and ATUS files."""
    project = load_yaml(PROJECT_CONFIG)
    sources = load_yaml(SOURCES_CONFIG).get("sources", {})
    blockers: list[str] = []
    warnings: list[str] = []
    required = ("event_id", "year", "month", "state_code", "municipality_code", "deaths_total", "injuries_total")
    mapping = project.get("columns", {})
    unset = [name for name in required if not mapping.get(name)]
    if unset:
        blockers.append(f"Confirmar mapeo ATUS: {', '.join(unset)}")
    atus_files = sorted(RAW.glob("conjunto_de_datos_atus_anual_csv/conjunto_de_datos/atus_anual_*.csv"))
    if not atus_files:
        blockers.append("No hay archivos ATUS anual en data/raw.")
    if not (RAW / "infraestructura_salud_alcaldias_cdmx_2020.csv").exists():
        blockers.append("Falta infraestructura_salud_alcaldias_cdmx_2020.csv.")
    for source_name in ("atus", "census_2020", "road_network", "clues", "municipal_boundaries"):
        entry = sources.get(source_name, {})
        if not entry or (source_name != "atus" and not entry.get("url")):
            warnings.append(f"Fuente sin URL oficial verificada: {source_name}")
    return ValidationResult(not blockers, tuple(blockers), tuple(warnings))


def quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and uniqueness for every column."""
    return pd.DataFrame({
        "column": frame.columns,
        "dtype": [str(dtype) for dtype in frame.dtypes],
        "missing_count": frame.isna().sum().to_numpy(),
        "missing_share": frame.isna().mean().to_numpy(),
        "unique_count": [frame[column].nunique(dropna=True) for column in frame.columns],
    })
