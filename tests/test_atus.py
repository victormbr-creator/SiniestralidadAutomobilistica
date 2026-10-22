import pandas as pd
import pytest
from siniestros_riesgo.atus import clean_atus


MAPPING = {"event_id": "folio", "year": "anio", "month": "mes", "state_code": "ent", "municipality_code": "mun", "deaths_total": "muertos", "injuries_total": "heridos"}


def test_targets_and_text_codes() -> None:
    raw = pd.DataFrame({"folio": ["a", "b"], "anio": [2020, 2020], "mes": [1, 2], "ent": [1, 9], "mun": [2, 15], "muertos": [0, 1], "heridos": [0, 0]})
    clean = clean_atus(raw, MAPPING, 2020)
    assert clean.geo_id.tolist() == ["01002", "09015"]
    assert clean.victimas.tolist() == [0, 1]


def test_invalid_month_fails() -> None:
    raw = pd.DataFrame({"folio": ["a"], "anio": [2020], "mes": [13], "ent": [1], "mun": [2], "muertos": [0], "heridos": [0]})
    with pytest.raises(ValueError, match="Mes inválido"):
        clean_atus(raw, MAPPING, 2020)
