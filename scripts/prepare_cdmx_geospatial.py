"""Prepare CDMX crash points and verified emergency hospital locations.

Raw inputs remain unchanged. The script filters national INEGI ATUS files to
CDMX, validates coordinates, links SEDESA's emergency service portfolio with
CLUES, and produces compact analysis-ready tables and a point map.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from siniestros_riesgo.config import FIGURES, PROCESSED, RAW, TABLES, setup_logging


logger = setup_logging("prepare_cdmx_geospatial")
PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

ATUS_COLUMNS = [
    "ID", "EDO", "MPIO", "ANIO", "MES", "DIA", "HORA", "MINUTOS",
    "DIASEMANA", "TIPACCID", "CLASE", "TOTMUERTOS", "TOTHERIDOS",
    "CALLE1", "CALLE2", "CARRETERA", "LONGITUD", "LATITUD",
]

# Names that changed materially between the SEDESA portfolio and CLUES.
CLUES_OVERRIDES = {
    "HG BELISARIO DOMINGUEZ": "DFIMB002266",
    "HMI XOCHIMILCO": "DFIMB001706",
    "HP ARAGON": "DFIMB000410",
    "CT VENUSTIANO CARRANZA": "DFIMB002085",
    "CT XOCHIMILCO": "DFIMB001694",
}


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().upper()
    replacements = {
        "HOSPITAL GENERAL": "HG",
        "HOSPITAL PEDIATRICO": "HP",
        "HOSPITAL MATERNO INFANTIL": "HMI",
        "CLINICA HOSPITAL": "CH",
        "CLINICA DE TOXICOLOGIA": "CT",
        "HOSPITAL ONCOLOGICO": "HO",
        "DOCTOR": "",
        "DRA": "",
        "DR": "",
        "CIUDAD DE MEXICO": "",
        "CDMX": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def _atus_csv(year: int) -> Path:
    candidates = list((RAW / "atus_georreferenciado" / f"atus_{year}").rglob("*.csv"))
    candidates = [path for path in candidates if "GEORREFERENCIADOS" in path.name.upper()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Se esperaba un CSV georreferenciado para {year}; encontrados: {candidates}")
    return candidates[0]


def prepare_crashes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the six national files in chunks and retain CDMX events."""
    frames: list[pd.DataFrame] = []
    for year in range(2019, 2025):
        path = _atus_csv(year)
        pieces = []
        for chunk in pd.read_csv(
            path,
            encoding="latin1",
            usecols=lambda column: column in ATUS_COLUMNS,
            chunksize=200_000,
            low_memory=False,
        ):
            entity = pd.to_numeric(chunk["EDO"], errors="coerce")
            pieces.append(chunk.loc[entity.eq(9)].copy())
        year_frame = pd.concat(pieces, ignore_index=True)
        if year_frame.empty:
            raise ValueError(f"ATUS georreferenciado {year} no contiene registros de CDMX")
        frames.append(year_frame)
        logger.info("ATUS %s: %s registros CDMX", year, f"{len(year_frame):,}")

    crashes = pd.concat(frames, ignore_index=True)
    numeric = [
        "EDO", "MPIO", "ANIO", "MES", "DIA", "HORA", "MINUTOS",
        "DIASEMANA", "TIPACCID", "CLASE", "TOTMUERTOS", "TOTHERIDOS",
        "LONGITUD", "LATITUD",
    ]
    for column in numeric:
        crashes[column] = pd.to_numeric(crashes[column], errors="coerce")
    crashes["geo_id"] = "09" + crashes["MPIO"].astype("Int64").astype("string").str.zfill(3)
    crashes["victims"] = crashes["TOTMUERTOS"].fillna(0) + crashes["TOTHERIDOS"].fillna(0)
    crashes["victim_crash"] = crashes["victims"].gt(0)
    crashes["coordinate_valid"] = crashes["LONGITUD"].between(-99.4, -98.9) & crashes["LATITUD"].between(19.0, 19.7)

    boroughs = gpd.read_file("data/external/cdmx_alcaldias_inegi.geojson").to_crs(4326)
    boundary = boroughs.geometry.union_all()
    points = gpd.GeoSeries(
        gpd.points_from_xy(crashes["LONGITUD"], crashes["LATITUD"]),
        crs=4326,
    )
    crashes["inside_cdmx_boundary"] = False
    valid = crashes["coordinate_valid"]
    crashes.loc[valid, "inside_cdmx_boundary"] = points.loc[valid].covered_by(boundary).to_numpy()

    catalog = pd.read_csv("data/external/inegi_municipal_catalog.csv", dtype={"cvegeo": "string"})
    catalog["cvegeo"] = catalog["cvegeo"].str.zfill(5)
    crashes = crashes.merge(
        catalog[["cvegeo", "nomgeo"]], left_on="geo_id", right_on="cvegeo",
        how="left", validate="many_to_one",
    ).drop(columns="cvegeo")

    coverage = (
        crashes.groupby("ANIO", as_index=False)
        .agg(
            registros_cdmx=("ID", "size"),
            coordenadas_validas=("coordinate_valid", "sum"),
            puntos_dentro_cdmx=("inside_cdmx_boundary", "sum"),
            siniestros_con_victimas=("victim_crash", "sum"),
        )
    )
    victim_with_coord = crashes["victim_crash"] & crashes["coordinate_valid"] & crashes["inside_cdmx_boundary"]
    extra = crashes.assign(victim_with_coord=victim_with_coord).groupby("ANIO")["victim_with_coord"].sum()
    coverage["siniestros_con_victimas_y_coordenadas"] = coverage["ANIO"].map(extra).astype(int)
    coverage["cobertura_coord_victimas_pct"] = (
        coverage["siniestros_con_victimas_y_coordenadas"]
        / coverage["siniestros_con_victimas"].replace(0, np.nan) * 100
    ).round(2)
    coverage.to_csv(TABLES / "cdmx_georeference_coverage.csv", index=False)
    crashes.to_parquet(PROCESSED / "cdmx_atus_georeferenced_2019_2024.parquet", index=False)
    return crashes, boroughs


def _read_clues(stamp: str) -> pd.DataFrame:
    path = RAW / "clues" / f"ESTABLECIMIENTO_SALUD_{stamp}.xlsx"
    facilities = pd.read_excel(path, sheet_name=f"CLUES_{stamp}")
    facilities = facilities[
        pd.to_numeric(facilities["CLAVE DE LA ENTIDAD"], errors="coerce").eq(9)
        & facilities["ESTATUS DE OPERACION"].astype(str).str.upper().eq("EN OPERACION")
    ].copy()
    facilities["LATITUD"] = pd.to_numeric(facilities["LATITUD"], errors="coerce")
    facilities["LONGITUD"] = pd.to_numeric(facilities["LONGITUD"], errors="coerce")
    facilities["coordinate_valid"] = facilities["LONGITUD"].between(-99.4, -98.9) & facilities["LATITUD"].between(19.0, 19.7)
    facilities["geo_id"] = "09" + facilities["CLAVE DEL MUNICIPIO"].astype("Int64").astype("string").str.zfill(3)
    keep = [
        "CLUES", "geo_id", "NOMBRE DE LA INSTITUCION", "MUNICIPIO",
        "NOMBRE TIPO ESTABLECIMIENTO", "NOMBRE DE TIPOLOGIA", "NOMBRE DE LA UNIDAD",
        "NOMBRE COMERCIAL", "TIPO DE VIALIDAD", "VIALIDAD", "NUMERO EXTERIOR",
        "TIPO DE ASENTAMIENTO", "ASENTAMIENTO", "CODIGO POSTAL", "NIVEL ATENCION",
        "LATITUD", "LONGITUD", "coordinate_valid",
    ]
    out = facilities[keep].copy()
    for column in out.columns:
        if out[column].dtype == "object":
            out[column] = out[column].astype("string")
    out.to_parquet(PROCESSED / f"cdmx_clues_facilities_{stamp}.parquet", index=False)
    out.to_csv(TABLES / f"cdmx_clues_facilities_{stamp}.csv", index=False)
    return facilities


def prepare_emergency_hospitals(facilities: pd.DataFrame) -> pd.DataFrame:
    """Join SEDESA's explicit emergency portfolio to current CLUES points."""
    portfolio_path = RAW / "clues" / "cartera_servicios_hospitales_segundo_nivel_sedesa_2025.xlsx"
    portfolio = pd.read_excel(portfolio_path, header=4)
    portfolio.columns = [str(column).strip() for column in portfolio.columns]
    emergency_columns = ["URGENCIAS GENERALES", "URGENCIAS PEDIATRÍA", "URGENCIAS GINECO-OBSTETRICIA"]
    portfolio = portfolio.dropna(subset=["UNIDAD MÉDICA"]).copy()
    for column in emergency_columns:
        portfolio[column] = portfolio[column].notna()
    portfolio["emergency_any"] = portfolio[emergency_columns].any(axis=1)
    portfolio = portfolio[portfolio["emergency_any"]].copy()

    candidates = facilities[
        facilities["NOMBRE DE LA INSTITUCION"].astype(str).str.contains("IMSS BIENESTAR", case=False, na=False)
        & facilities["coordinate_valid"]
    ].copy()
    matches = []
    for _, row in portfolio.iterrows():
        source_name = row["UNIDAD MÉDICA"]
        normalized = _normalize(source_name)
        if normalized in CLUES_OVERRIDES:
            clues = CLUES_OVERRIDES[normalized]
            match_score = 1.0
        else:
            scored = []
            for _, candidate in candidates.iterrows():
                names = [candidate["NOMBRE DE LA UNIDAD"], candidate["NOMBRE COMERCIAL"]]
                score = max(
                    SequenceMatcher(None, normalized, _normalize(name)).ratio()
                    for name in names if pd.notna(name)
                )
                scored.append((score, candidate["CLUES"]))
            match_score, clues = max(scored)
        matches.append({"UNIDAD MÉDICA": source_name, "CLUES": clues, "match_score": match_score})

    matched = portfolio.merge(pd.DataFrame(matches), on="UNIDAD MÉDICA", validate="one_to_one")
    if matched["CLUES"].duplicated().any() or matched["match_score"].lt(0.80).any():
        raise ValueError("La vinculación cartera SEDESA-CLUES requiere revisión manual")
    facility_columns = [
        "CLUES", "geo_id", "NOMBRE DE LA INSTITUCION", "MUNICIPIO", "NOMBRE DE LA UNIDAD",
        "NOMBRE COMERCIAL", "NOMBRE DE TIPOLOGIA", "TIPO DE VIALIDAD", "VIALIDAD",
        "NUMERO EXTERIOR", "TIPO DE ASENTAMIENTO", "ASENTAMIENTO", "CODIGO POSTAL",
        "LATITUD", "LONGITUD",
    ]
    emergency = matched.merge(facilities[facility_columns], on="CLUES", validate="one_to_one")
    emergency = emergency.rename(columns={
        "UNIDAD MÉDICA": "nombre_cartera_sedesa",
        "URGENCIAS GENERALES": "urgencias_generales",
        "URGENCIAS PEDIATRÍA": "urgencias_pediatria",
        "URGENCIAS GINECO-OBSTETRICIA": "urgencias_gineco_obstetricia",
        "NOMBRE DE LA UNIDAD": "nombre_clues",
        "NOMBRE DE LA INSTITUCION": "institucion",
        "MUNICIPIO": "alcaldia",
        "NOMBRE DE TIPOLOGIA": "tipologia",
        "LATITUD": "latitud",
        "LONGITUD": "longitud",
    })
    emergency = emergency[[
        "nombre_cartera_sedesa", "CLUES", "match_score", "geo_id", "institucion",
        "alcaldia", "nombre_clues", "NOMBRE COMERCIAL", "tipologia",
        "urgencias_generales", "urgencias_pediatria", "urgencias_gineco_obstetricia",
        "TIPO DE VIALIDAD", "VIALIDAD", "NUMERO EXTERIOR", "TIPO DE ASENTAMIENTO",
        "ASENTAMIENTO", "CODIGO POSTAL", "latitud", "longitud",
    ]]
    for column in emergency.columns:
        if emergency[column].dtype == "object":
            emergency[column] = emergency[column].astype("string")
    emergency.to_parquet(PROCESSED / "cdmx_emergency_hospitals_sedesa_2025.parquet", index=False)
    emergency.to_csv(TABLES / "cdmx_emergency_hospitals_sedesa_2025.csv", index=False)
    return emergency


def calculate_access(crashes: pd.DataFrame, hospitals: pd.DataFrame) -> pd.DataFrame:
    """Calculate straight-line nearest-hospital distances for victim crashes."""
    points = crashes[crashes["victim_crash"] & crashes["coordinate_valid"] & crashes["inside_cdmx_boundary"]].copy()
    point_gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["LONGITUD"], points["LATITUD"]), crs=4326).to_crs(32614)
    hospital_gdf = gpd.GeoDataFrame(
        hospitals.copy(), geometry=gpd.points_from_xy(hospitals["longitud"], hospitals["latitud"]), crs=4326,
    ).to_crs(32614)
    nearest = NearestNeighbors(n_neighbors=1).fit(np.column_stack([hospital_gdf.geometry.x, hospital_gdf.geometry.y]))
    distances, indexes = nearest.kneighbors(np.column_stack([point_gdf.geometry.x, point_gdf.geometry.y]))
    points["nearest_emergency_distance_km"] = distances[:, 0] / 1000
    points["nearest_emergency_clues"] = hospital_gdf.iloc[indexes[:, 0]]["CLUES"].to_numpy()
    points.to_parquet(PROCESSED / "cdmx_victim_crash_points_with_emergency_distance.parquet", index=False)

    access = points.groupby(["geo_id", "nomgeo"], as_index=False).agg(
        victim_crashes_georeferenced=("ID", "size"),
        median_nearest_emergency_km=("nearest_emergency_distance_km", "median"),
        p90_nearest_emergency_km=("nearest_emergency_distance_km", lambda values: values.quantile(0.90)),
        share_over_5km=("nearest_emergency_distance_km", lambda values: values.gt(5).mean()),
        share_over_10km=("nearest_emergency_distance_km", lambda values: values.gt(10).mean()),
    )
    hospital_counts = hospitals.groupby("geo_id").size().rename("verified_emergency_hospitals")
    access = access.merge(hospital_counts, on="geo_id", how="left")
    access["verified_emergency_hospitals"] = access["verified_emergency_hospitals"].fillna(0).astype(int)
    access.to_csv(TABLES / "cdmx_accident_emergency_access.csv", index=False)
    return points


def make_map(crashes: pd.DataFrame, hospitals: pd.DataFrame, boroughs: gpd.GeoDataFrame) -> None:
    points = crashes[crashes["victim_crash"] & crashes["coordinate_valid"] & crashes["inside_cdmx_boundary"]]
    fig, ax = plt.subplots(figsize=(9, 9))
    boroughs.plot(ax=ax, facecolor="#F4F1EA", edgecolor="#6E7781", linewidth=0.65)
    ax.scatter(points["LONGITUD"], points["LATITUD"], s=3, alpha=0.20, color="#C0392B", linewidths=0, label="Siniestro con víctimas, 2019–2024")
    ax.scatter(hospitals["longitud"], hospitals["latitud"], s=48, marker="^", color="#1167B1", edgecolor="white", linewidth=0.5, label="Hospital con urgencias verificadas")
    ax.set_title("Siniestros con víctimas y hospitales con urgencias en CDMX", fontsize=14, weight="bold")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    fig.text(0.5, 0.012, "Fuentes: INEGI ATUS georreferenciado; cartera SEDESA 2025; CLUES agosto 2026.", ha="center", fontsize=7, color="#4D5966")
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    fig.savefig(FIGURES / "cdmx_accident_points_emergency_hospitals.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    crashes, boroughs = prepare_crashes()
    _read_clues("202412")
    current_facilities = _read_clues("202608")
    hospitals = prepare_emergency_hospitals(current_facilities)
    calculate_access(crashes, hospitals)
    make_map(crashes, hospitals, boroughs)
    logger.info("Preparación terminada: %s siniestros CDMX, %s hospitales con urgencias", f"{len(crashes):,}", len(hospitals))


if __name__ == "__main__":
    main()
