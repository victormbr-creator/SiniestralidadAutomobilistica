"""Configuración central, rutas portables y semilla reproducible."""

from pathlib import Path
from typing import Any
import logging
import random

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
LOGS = OUTPUTS / "logs"
REPORTS = ROOT / "reports"
PROJECT_CONFIG = ROOT / "config" / "project.yml"
SOURCES_CONFIG = ROOT / "config" / "sources.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def setup_logging(command: str) -> logging.Logger:
    """Configure console and per-command file logs."""
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("siniestros_riesgo")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        file_handler = logging.FileHandler(LOGS / f"{command}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(console)
        logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    """Seed Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
