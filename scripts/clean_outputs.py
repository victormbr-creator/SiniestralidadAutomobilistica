"""Remove generated outputs while preserving raw and processed data."""

from siniestros_riesgo.config import FIGURES, TABLES, MODELS, LOGS

for directory in (FIGURES, TABLES, MODELS, LOGS):
    for path in directory.glob("*"):
        if path.is_file() and path.name != ".gitkeep":
            path.unlink()
