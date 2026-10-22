"""Cleaning for the official annual ATUS release without geocoordinates."""

from pathlib import Path
import pandas as pd


def clean_annual_file(path: Path, year: int, columns: dict) -> pd.DataFrame:
    """Read confirmed annual fields and create victim totals and temporal flags."""
    deaths = columns["death_fields"]
    injuries = columns["injury_fields"]
    scalar = [value for key, value in columns.items() if key not in {"death_fields", "injury_fields"}]
    usecols = list(dict.fromkeys(scalar + deaths + injuries))
    frame = pd.read_csv(path, usecols=usecols, dtype="string", encoding="utf-8-sig", low_memory=False)
    inverse = {value: key for key, value in columns.items() if key not in {"death_fields", "injury_fields"}}
    frame = frame.rename(columns=inverse)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce").astype("Int64")
    frame["day"] = pd.to_numeric(frame["day"], errors="coerce").astype("Int64")
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce").astype("Int64")
    if frame["year"].isna().any() or not frame["year"].eq(year).all():
        raise ValueError(f"Año inválido en {path.name}")
    if frame["month"].isna().any() or not frame["month"].between(1, 12).all():
        raise ValueError(f"Mes inválido en {path.name}")
    frame["state_code"] = frame["state_code"].str.zfill(2)
    frame["municipality_code"] = frame["municipality_code"].str.zfill(3)
    frame["geo_id"] = frame["geo_id"].str.zfill(5)
    expected = frame["state_code"] + frame["municipality_code"]
    if not frame["geo_id"].eq(expected).all():
        raise ValueError(f"CVEGEO no coincide con entidad+municipio en {path.name}")
    for field in deaths + injuries:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        if frame[field].isna().any() or (frame[field] < 0).any() or (frame[field] % 1 != 0).any():
            raise ValueError(f"Conteos inválidos en {field}, {path.name}")
    frame["deaths_total"] = frame[deaths].sum(axis=1).astype("int32")
    frame["injuries_total"] = frame[injuries].sum(axis=1).astype("int32")
    frame["victimas"] = ((frame["deaths_total"] + frame["injuries_total"]) > 0).astype("int8")
    frame["fatal"] = frame["deaths_total"].gt(0).astype("int8")
    frame["night"] = (frame["hour"].between(0, 5) | frame["hour"].between(20, 23)).astype("int8")
    frame["weekend"] = frame["weekday"].isin(["Sábado", "Domingo"]).astype("int8")
    frame["report_status"] = "observed"
    frame["event_id"] = frame["year"].astype(str) + "-" + pd.Series(range(len(frame)), index=frame.index).astype(str)
    return frame[["event_id", "geo_id", "state_code", "municipality_code", "year", "month", "day", "hour", "weekday", "night", "weekend", "status", "crash_class", "deaths_total", "injuries_total", "victimas", "fatal", "report_status"]]
