"""Exploratory dashboard shell; renders outputs only when available."""

import json

import pandas as pd
import streamlit as st

from siniestros_viales.config import PROCESSED, TABLES

st.set_page_config(page_title="Siniestros viales MX", layout="wide")
st.title("Severidad de siniestros viales en México")
st.caption("Tablero exploratorio. Las cifras aparecerán tras completar y validar el pipeline.")

data_path = PROCESSED / "modeling_dataset.parquet"
if data_path.exists():
    data = pd.read_parquet(data_path)
    years = sorted(data["year"].dropna().unique().tolist())
    selected_years = st.multiselect("Año", years, default=years)
    filtered = data[data["year"].isin(selected_years)]
    for column, label in (("state", "Entidad"), ("municipality", "Municipio")):
        if column in filtered:
            choices = sorted(filtered[column].dropna().astype(str).unique().tolist())
            chosen = st.multiselect(label, choices)
            if chosen:
                filtered = filtered[filtered[column].astype(str).isin(chosen)]
    a, b, c = st.columns(3)
    a.metric("Siniestros", f"{len(filtered):,}")
    b.metric("Con víctimas", f"{filtered['victimas'].sum():,}")
    c.metric("Proporción con víctimas", f"{filtered['victimas'].mean():.1%}" if len(filtered) else "—")
    st.subheader("Frecuencia y severidad")
    st.info("TODO: incorporar agregados municipales con denominadores verificados.")
else:
    st.info("Aún no existe data/processed/modeling_dataset.parquet.")

st.subheader("Mapa")
st.info("TODO: integrar límites municipales verificados y tasas de severidad.")
st.subheader("Factores de riesgo")
st.info("TODO: mostrar coeficientes e importancias tras evaluación.")
st.subheader("Resultados de modelos")
metrics_path = TABLES / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.dataframe(pd.DataFrame(metrics).T)
else:
    st.info("Aún no hay resultados de modelos.")
