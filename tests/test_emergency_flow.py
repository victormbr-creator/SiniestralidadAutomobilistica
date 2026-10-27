from pathlib import Path
import csv
import zipfile

from siniestros_riesgo.emergency_flow import aggregate_archive, parse_stay_hours


def test_parse_stay_hours_across_schemas() -> None:
    assert parse_stay_hours({"HORASESTANCIA": "02:30"}) == 2.5
    assert parse_stay_hours({"TIEMPO_ESTANCIA": "999:99"}) is None
    assert parse_stay_hours({
        "FECHAINGRESO": "2018-01-30 00:00:00.000", "FECHAALTA": "2018-01-30 00:00:00.000",
        "HORAINIATE": "19", "MININIATE": "35", "HORATERATE": "20", "MINTAREATE": "10",
        "HORASESTANCIA": "0",
    }) == 35 / 60


def test_aggregate_archive_filters_and_uses_alias(tmp_path: Path) -> None:
    archive_path = tmp_path / "Urgencias_2018.zip"
    rows = [
        {"CLUES": "OLD", "MES_ESTADISTICO": "1", "TIPOURGEN": "1", "MOTATE": "1", "TIPOCAMA": "1", "ENVIADOA": "1", "HORASESTANCIA": "02:00"},
        {"CLUES": "OTHER", "MES_ESTADISTICO": "1", "TIPOURGEN": "2", "MOTATE": "2", "TIPOCAMA": "3", "ENVIADOA": "4", "HORASESTANCIA": "01:00"},
    ]
    csv_path = tmp_path / "URGENCIAS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, arcname="URGENCIAS.csv")
    result, coverage = aggregate_archive(archive_path, 2018, {"OLD": "CURRENT"})
    assert coverage["national_emergency_records"] == 2
    assert coverage["matched_emergency_records"] == 1
    assert result.iloc[0]["qualified_emergencies"] == 1
    assert result.iloc[0]["bed_stay_hours"] == 2
