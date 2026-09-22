"""Rutas y parámetros centrales del proyecto."""

from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
REPORTS = OUTPUTS / "reports"
SEED = 42
TRAIN_YEARS = tuple(range(2018, 2023))
TEST_YEARS = (2023, 2024)
TARGET = "victimas"
IDENTIFIERS = ("event_id", "year")
LEAKAGE_COLUMNS = ("deaths", "injuries", "fatal", "victimas")


def project_path(value: str | Path) -> Path:
    """Resolve a relative configuration path inside the repository."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


ATUS_MAPPING = project_path(os.getenv("SVMX_ATUS_MAPPING", "config/atus_columns.json"))
SOURCES = project_path(os.getenv("SVMX_SOURCES", "config/sources.json"))


@dataclass(frozen=True)
class YearSplit:
    """Fixed temporal holdout years."""

    train: tuple[int, ...] = TRAIN_YEARS
    test: tuple[int, ...] = TEST_YEARS
