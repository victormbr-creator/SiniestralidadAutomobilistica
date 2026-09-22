"""Shared logging and file helpers."""

import json
import logging
import os
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a logger with one consistent root configuration."""
    logging.basicConfig(level=os.getenv("SVMX_LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON configuration file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write indented JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
