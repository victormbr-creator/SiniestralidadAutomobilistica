"""Parse and aggregate DGIS/SINBA emergency records without extracting raw files."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
import csv
import io
from pathlib import Path
import subprocess
from typing import BinaryIO, Iterator
import zipfile

import pandas as pd


COUNT_COLUMNS = {
    "emergency_visits": None,
    "qualified_emergencies": ("TIPOURGENCIA", "1"),
    "nonqualified_emergencies": ("TIPOURGENCIA", "2"),
    "unspecified_urgency": ("TIPOURGENCIA", "9"),
    "accident_poisoning_violence_visits": ("MOTATE", "1"),
    "observation_bed_visits": ("TIPOCAMA", "1"),
    "shock_bed_visits": ("TIPOCAMA", "2"),
    "no_bed_visits": ("TIPOCAMA", "3"),
    "hospital_admissions": ("ENVIADOA", "1"),
    "transfers": ("ENVIADOA", "3"),
    "deaths_in_emergency": ("ENVIADOA", "5"),
}


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    """Normalize historical column aliases and strip encoded whitespace."""
    normalized = {
        str(key or "").lstrip("\ufeff").upper(): str(value or "").strip()
        for key, value in row.items()
    }
    if "TIPOURGENCIA" not in normalized:
        normalized["TIPOURGENCIA"] = normalized.get("TIPOURGEN", "")
    if "MINTAREATE" not in normalized:
        normalized["MINTAREATE"] = normalized.get("MINTERATE", "")
    return normalized


def parse_stay_hours(row: dict[str, str]) -> float | None:
    """Return recorded emergency stay hours across historical schemas."""
    row = normalize_row(row)
    raw = row.get("HORASESTANCIA") or row.get("TIEMPO_ESTANCIA") or ""
    if ":" in raw:
        try:
            hours, minutes = (int(part) for part in raw.split(":", maxsplit=1))
        except ValueError:
            return None
        if hours == 999 or not 0 <= minutes < 60 or hours < 0:
            return None
        return hours + minutes / 60

    # The 2018 schema supplies dates and split clock fields. This preserves minutes
    # that are truncated in its integer HORASESTANCIA field.
    start_date = row.get("FECHAINGRESO", "").split(" ")[0]
    end_date = row.get("FECHAALTA", "").split(" ")[0]
    split_fields = ("HORAINIATE", "MININIATE", "HORATERATE", "MINTAREATE")
    if start_date and end_date and all(row.get(field, "") != "" for field in split_fields):
        try:
            start = datetime.fromisoformat(start_date).replace(
                hour=int(row["HORAINIATE"]), minute=int(row["MININIATE"])
            )
            end = datetime.fromisoformat(end_date).replace(
                hour=int(row["HORATERATE"]), minute=int(row["MINTAREATE"])
            )
            hours = (end - start).total_seconds() / 3600
            if 0 <= hours < 999:
                return hours
        except (ValueError, TypeError):
            pass
    try:
        hours = float(raw)
    except ValueError:
        return None
    return hours if 0 <= hours < 999 else None


def _urgency_member(archive: zipfile.ZipFile) -> str:
    matches = [
        name for name in archive.namelist()
        if Path(name).stem.upper() == "URGENCIAS" and Path(name).suffix.lower() in {".csv", ".txt"}
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one URGENCIAS table, found {matches}")
    return matches[0]


@contextmanager
def open_urgency_member(archive_path: Path) -> Iterator[tuple[BinaryIO, str]]:
    """Open the urgency member, including legacy Deflate64 archives."""
    with zipfile.ZipFile(archive_path) as archive:
        member = _urgency_member(archive)
        compression = archive.getinfo(member).compress_type
        if compression in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}:
            with archive.open(member) as binary:
                yield binary, member
            return
    process = subprocess.Popen(
        ["unzip", "-p", str(archive_path), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError(f"Unable to open {member} from {archive_path}")
    try:
        yield process.stdout, member
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"unzip failed for {archive_path}: {stderr.strip()}")


def aggregate_archive(
    archive_path: Path,
    year: int,
    clues_crosswalk: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stream one national archive and retain monthly aggregates for target CLUES."""
    aggregates: dict[tuple[str, int], defaultdict[str, float]] = {}
    national_rows = 0
    matched_rows = 0
    matched_clues: set[str] = set()
    with open_urgency_member(archive_path) as (binary, member):
        delimiter = "," if Path(member).suffix.lower() == ".csv" else "|"
        stream = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="")
        reader = csv.DictReader(stream, delimiter=delimiter)
        for raw_row in reader:
            national_rows += 1
            row = normalize_row(raw_row)
            current_clues = clues_crosswalk.get(row.get("CLUES", ""))
            if current_clues is None:
                continue
            try:
                month = int(float(row.get("MES_ESTADISTICO", "")))
            except ValueError:
                continue
            if not 1 <= month <= 12:
                continue
            matched_rows += 1
            matched_clues.add(current_clues)
            key = (current_clues, month)
            values = aggregates.setdefault(key, defaultdict(float))
            for output, condition in COUNT_COLUMNS.items():
                if condition is None or row.get(condition[0], "") == condition[1]:
                    values[output] += 1
            stay = parse_stay_hours(row)
            if stay is not None:
                values["known_stay_records"] += 1
                values["total_stay_hours"] += stay
                if row.get("TIPOCAMA", "") in {"1", "2"}:
                    values["bed_stay_records"] += 1
                    values["bed_stay_hours"] += stay

    rows = []
    for (clues, month), values in aggregates.items():
        row = {"CLUES": clues, "year": year, "month": month, **values}
        rows.append(row)
    result = pd.DataFrame(rows)
    numeric = list(COUNT_COLUMNS) + [
        "known_stay_records", "total_stay_hours", "bed_stay_records", "bed_stay_hours"
    ]
    for column in numeric:
        if column not in result:
            result[column] = 0
    result[numeric] = result[numeric].fillna(0)
    coverage = {
        "year": year,
        "national_emergency_records": national_rows,
        "matched_emergency_records": matched_rows,
        "matched_current_hospitals": len(matched_clues),
    }
    return result.sort_values(["CLUES", "year", "month"]), coverage
