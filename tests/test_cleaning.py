"""Tests for schema mapping and outcome construction."""

import pandas as pd
import pytest

from siniestros_viales.clean_atus import clean_year, snake_case


def test_snake_case_normalizes_accents() -> None:
    assert snake_case("Número de Víctimas") == "numero_de_victimas"


def test_clean_year_builds_targets() -> None:
    source = pd.DataFrame({"Muertos verificados": [0, 1, 0], "Heridos verificados": [0, 0, 2]})
    result = clean_year(source, 2023, {"deaths": "Muertos verificados", "injuries": "Heridos verificados"})
    assert result["victimas"].tolist() == [0, 1, 1]
    assert result["fatal"].tolist() == [0, 1, 0]
    assert result["year"].tolist() == [2023] * 3


def test_clean_year_requires_confirmed_counts() -> None:
    with pytest.raises(ValueError, match="mapping"):
        clean_year(pd.DataFrame({"otro": [1]}), 2023, {})
