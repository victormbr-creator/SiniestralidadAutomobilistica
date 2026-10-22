"""ATUS cleaning with explicit, edition-verified mappings."""

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


def snake_case(value: str) -> str:
    """Normalize a source label while preserving its meaning."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def read_table(path: Path) -> pd.DataFrame:
    """Read a supported table without editing it."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        try:
            return pd.read_csv(path, dtype="string", low_memory=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, dtype="string", low_memory=False, encoding="latin1")
    raise ValueError(f"Formato no soportado: {path.name}")


def clean_atus(frame: pd.DataFrame, mapping: dict[str, str], edition_year: int) -> pd.DataFrame:
    """Map confirmed columns, validate codes/counts and construct outcomes."""
    if frame.empty:
        raise ValueError("El archivo ATUS está vacío")
    normalized = frame.rename(columns={column: snake_case(column) for column in frame.columns}).copy()
    rename: dict[str, str] = {}
    for canonical, original in mapping.items():
        if original:
            source = snake_case(original)
            if source not in normalized:
                raise ValueError(f"Columna confirmada ausente: {original}")
            rename[source] = canonical
    normalized = normalized.rename(columns=rename)
    required = {"event_id", "year", "month", "state_code", "municipality_code", "deaths_total", "injuries_total"}
    if missing := required - set(normalized):
        raise ValueError(f"Faltan columnas canónicas: {sorted(missing)}")
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
    normalized["month"] = pd.to_numeric(normalized["month"], errors="coerce").astype("Int64")
    if normalized["year"].isna().any() or not normalized["year"].eq(edition_year).all():
        raise ValueError(f"Año inválido o distinto de la edición {edition_year}")
    if normalized["month"].isna().any() or not normalized["month"].between(1, 12).all():
        raise ValueError("Mes inválido")
    normalized["state_code"] = normalized["state_code"].astype("string").str.strip().str.zfill(2)
    normalized["municipality_code"] = normalized["municipality_code"].astype("string").str.strip().str.zfill(3)
    normalized["geo_id"] = normalized["state_code"] + normalized["municipality_code"]
    if not normalized["geo_id"].str.fullmatch(r"\d{5}").all():
        raise ValueError("Claves geográficas inválidas")
    for column in ("deaths_total", "injuries_total"):
        values = pd.to_numeric(normalized[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{column} debe contener enteros no negativos")
        normalized[column] = values.astype("int64")
    normalized["victimas"] = ((normalized["deaths_total"] + normalized["injuries_total"]) > 0).astype("int8")
    if "longitude" in normalized and "latitude" in normalized:
        lon = pd.to_numeric(normalized["longitude"], errors="coerce")
        lat = pd.to_numeric(normalized["latitude"], errors="coerce")
        normalized["georeferenced"] = lon.between(-119, -86) & lat.between(14, 33)
        normalized["longitude"] = lon
        normalized["latitude"] = lat
    else:
        normalized["georeferenced"] = False
    if "day" in normalized:
        day = pd.to_numeric(normalized["day"], errors="coerce")
        normalized["certified_zero"] = day.eq(0)
        normalized["invalid_day"] = ~(day.between(1, 31) | day.eq(0) | day.eq(32))
    else:
        normalized["certified_zero"] = False
        normalized["invalid_day"] = False
    normalized["report_status"] = np.where(normalized["certified_zero"], "certified_zero", "observed")
    return normalized
