"""Definitive CDMX report: forecast, hospital pressure, and new locations."""

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import BaseDocTemplate, Frame, Image as RLImage, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

from .config import FIGURES, MODELS, REPORTS, TABLES


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5B6573"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "Riesgo vial y capacidad de urgencias - CDMX")
    canvas.drawRightString(letter[0] - 1.8 * cm, 1.1 * cm, f"Página {doc.page}")
    canvas.restoreState()


def _table(data, widths, font_size=8):
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEEF0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#A6B1BC")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_cdmx_report(output: Path = REPORTS / "resultados_proyecto.pdf") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleMX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#13315C"), alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="SectionMX", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#0B6E75"), spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle(name="NoteMX", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=colors.HexColor("#5B6573")))
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    frame = Frame(1.8 * cm, 1.7 * cm, letter[0] - 3.6 * cm, letter[1] - 3.4 * cm, id="main")
    doc = BaseDocTemplate(str(output), pagesize=letter, title="Riesgo vial y capacidad de atención por alcaldía, CDMX", author="Proyecto de políticas públicas", pageTemplates=[PageTemplate(id="main", frames=frame, onPage=_footer)])

    manifest = json.loads((MODELS / "cdmx_manifest.json").read_text(encoding="utf-8"))
    months_es = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")

    def month_label(value):
        period = pd.Period(value, freq="M")
        return f"{months_es[period.month - 1]} de {period.year}"

    forecast_label = f"{month_label(manifest['forecast_start'])} a {month_label(manifest['forecast_end'])}"
    metrics = pd.read_csv(TABLES / "cdmx_test_metrics.csv")
    validation = pd.read_csv(TABLES / "cdmx_temporal_validation.csv")
    capacity = pd.read_csv(TABLES / "cdmx_risk_capacity_priority_future.csv", dtype={"geo_id": "string"})
    forecast = pd.read_csv(TABLES / "cdmx_monthly_forecast_future.csv", dtype={"geo_id": "string"})
    geocoverage = pd.read_csv(TABLES / "cdmx_georeference_coverage.csv")
    access = pd.read_csv(TABLES / "cdmx_accident_emergency_access.csv", dtype={"geo_id": "string"})
    hospitals = pd.read_csv(TABLES / "cdmx_emergency_hospital_capacity_2026.csv", dtype={"geo_id": "string"})
    pressure = pd.read_csv(TABLES / "cdmx_borough_hospital_pressure_future.csv", dtype={"geo_id": "string"})
    recommendations = pd.read_csv(TABLES / "cdmx_new_hospital_candidates.csv", dtype={"geo_id": "string"})

    model_names = {"random_forest": "Random Forest Poisson", "xgboost_poisson": "XGBoost Poisson", "negative_binomial": "Binomial negativa", "baseline": "Promedio móvil"}
    selected = manifest["selected_forecast_model"]
    story = [
        Spacer(1, 1.7 * cm),
        Paragraph("¿Dónde habrá mayor peligro vial y dónde reforzar urgencias?", styles["TitleMX"]),
        Paragraph("Pronóstico por alcaldía, presión hospitalaria y ubicación de nuevas unidades en Ciudad de México", styles["Heading2"]),
        Spacer(1, .7 * cm),
        _table([
            ["Ámbito", "16 alcaldías de Ciudad de México"],
            ["Observación", f"enero de 2018 a {month_label(manifest['observed_through'])}"],
            ["Validación", "progresiva: 2021, 2022 y 2023"],
            ["Prueba final", "2024-2025; no selecciona el modelo"],
            ["Modelo futuro", model_names[selected]],
            ["Horizonte", forecast_label],
            ["Capacidad", f"{len(hospitals)} hospitales; recursos SINERHIAS 2026"],
        ], [4.2 * cm, 11.8 * cm]),
        Spacer(1, .8 * cm),
        Paragraph("El estudio es predictivo y de planeación. Las asociaciones no son efectos causales y las coordenadas propuestas requieren factibilidad técnica.", styles["NoteMX"]),
        PageBreak(),
    ]

    validation_summary = validation[validation["selected_option"]].groupby("model", as_index=False).mae.mean().sort_values("mae")
    vtable = [["Modelo", "MAE medio 2021-2023"]] + [[model_names.get(r.model, r.model), f"{r.mae:.3f}"] for r in validation_summary.itertuples()]
    mtable = [["Modelo", "MAE", "Devianza Poisson", "Captura top 10%"]] + [[model_names.get(r.model, r.model), f"{r.mae:.3f}", f"{r.poisson_deviance:.3f}", f"{r.capture_top_10:.1%}"] for r in metrics.sort_values("mae").itertuples()]
    test_best = metrics.sort_values("mae").iloc[0]
    story.extend([
        Paragraph("1. Entrenamiento, pandemia y selección", styles["SectionMX"]),
        Paragraph("La unidad es alcaldía-mes y el resultado es el conteo de siniestros con al menos una persona herida o fallecida. Los predictores son rezago mensual, promedio de doce meses, rezago de alcaldías contiguas, estacionalidad, tendencia e identidad territorial.", styles["BodyText"]),
        Paragraph("Se conserva 2020 como choque real de movilidad. La validación entrena 2018-2020 y valida 2021; después amplía el entrenamiento para validar 2022 y 2023. No se proyecta una etiqueta de pandemia porque no representa una condición futura conocida.", styles["BodyText"]),
        _table(vtable, [9.5 * cm, 6.1 * cm]),
        Paragraph(f"Tabla 1. {model_names[selected]} obtuvo el menor MAE medio y se eligió antes de abrir la prueba final.", styles["NoteMX"]),
        Spacer(1, 7),
        RLImage(str(FIGURES / "cdmx_model_mae.png"), width=15.3 * cm, height=8.9 * cm),
        Paragraph("Figura 1. Error absoluto medio en la prueba final 2024-2025.", styles["NoteMX"]),
        Spacer(1, 7),
        _table(mtable, [5.0 * cm, 2.6 * cm, 4.2 * cm, 3.7 * cm]),
        Paragraph(f"Tabla 2. En la prueba final, {model_names[test_best.model]} logró el menor MAE ({test_best.mae:.3f}). Ese resultado se reporta como diagnóstico y no reemplaza la regla de selección predefinida.", styles["NoteMX"]),
        PageBreak(),
    ])

    victim_points = int(geocoverage.siniestros_con_victimas_y_coordenadas.sum())
    victim_total = int(geocoverage.siniestros_con_victimas.sum())
    far = access.sort_values("p90_nearest_emergency_km", ascending=False).head(5)
    far_table = [["Alcaldía", "P90 km", ">5 km", "Hosp. urg."]] + [[r.nomgeo, f"{r.p90_nearest_emergency_km:.1f}", f"{r.share_over_5km:.1%}", str(int(r.verified_emergency_hospitals))] for r in far.itertuples()]
    story.extend([
        Paragraph("2. Granularidad puntual y red actual", styles["SectionMX"]),
        Paragraph(f"ATUS georreferenciado permite ubicar {victim_points:,} de {victim_total:,} siniestros con víctimas de 2019-2024 ({victim_points / victim_total:.1%}). Estos puntos distribuyen el pronóstico futuro dentro de cada alcaldía según la severidad histórica; no son predicciones de eventos individuales.", styles["BodyText"]),
        RLImage(str(FIGURES / "cdmx_accident_points_emergency_hospitals.png"), width=11.3 * cm, height=14.1 * cm),
        Paragraph("Figura 2. Puntos observados y hospitales SEDESA/IMSS-Bienestar con urgencias verificadas.", styles["NoteMX"]),
        Spacer(1, 7),
        _table(far_table, [7.2 * cm, 2.7 * cm, 2.7 * cm, 3.0 * cm]),
        Paragraph("Tabla 3. Cola de distancia directa al hospital más cercano. No equivale a tiempo de ambulancia.", styles["NoteMX"]),
        PageBreak(),
    ])

    bridge_note = f" Incluye {manifest['recursive_bridge_months']} meses puente desde el último dato observado." if manifest["recursive_bridge_months"] else ""
    top = capacity.sort_values("predicted_horizon_victim_crashes", ascending=False)
    rank_table = [["Rango", "Alcaldía", "Predicción", "Por 100 mil"]] + [[str(i + 1), r.nomgeo, f"{r.predicted_horizon_victim_crashes:,.0f}", f"{r.predicted_horizon_per_100k:,.1f}"] for i, r in enumerate(top.itertuples())]
    story.extend([
        Paragraph(f"3. Pronóstico futuro: {forecast_label}", styles["SectionMX"]),
        Paragraph(f"El modelo ganador se reajustó con toda la historia 2018-2025 y generó {manifest['forecast_horizon_months']} meses de pronóstico recursivo.{bridge_note}", styles["BodyText"]),
        RLImage(str(FIGURES / "cdmx_risk_map_future.png"), width=13.9 * cm, height=13.9 * cm),
        Paragraph("Figura 3. Conteo predicho de siniestros con víctimas por alcaldía.", styles["NoteMX"]),
        PageBreak(),
        Paragraph("Ranking de las 16 alcaldías", styles["SectionMX"]),
        _table(rank_table, [1.3 * cm, 7.0 * cm, 3.5 * cm, 3.7 * cm]),
        Paragraph("Tabla 4. La tasa usa población censal fija como contexto; no mide exposición por viaje.", styles["NoteMX"]),
        Spacer(1, 6),
        RLImage(str(FIGURES / "cdmx_forecast_ranking_future.png"), width=15.2 * cm, height=9.4 * cm),
        Paragraph("Figura 4. Comparación del conteo predicho entre alcaldías.", styles["NoteMX"]),
        PageBreak(),
    ])

    top_pressure = pressure.sort_values("pressure_rank").head(8)
    pressure_table = [["Rango", "Alcaldía", "Riesgo pred.", "Camas urg.", "Por cama", "Urgenciólogos"]] + [[str(int(r.pressure_rank)), r.nomgeo, f"{r.predicted_horizon_victim_crashes:,.0f}", f"{int(r.emergency_beds)}", f"{r.predicted_crashes_per_emergency_bed:.1f}", f"{int(r.emergency_physicians)}"] for r in top_pressure.itertuples()]
    story.extend([
        Paragraph("4. Presión hospitalaria medible", styles["SectionMX"]),
        Paragraph(f"Los {len(hospitals)} hospitales coincidieron uno a uno por CLUES con SINERHIAS. Para esta red IMSS-Bienestar, el corte de recursos es 15 de mayo de 2026. La capacidad incluye {int(hospitals.emergency_beds.sum())} camas de urgencias, {int(hospitals.emergency_physicians.sum())} médicos urgenciólogos y {int(hospitals.contact_nurses.sum()):,} enfermeras en contacto con pacientes.", styles["BodyText"]),
        Paragraph("La presión principal divide siniestros predichos entre camas de urgencias. También se publican razones por médico urgenciólogo y por personal de enfermería. Los denominadores no se mezclan y los ceros no se imputan.", styles["BodyText"]),
        RLImage(str(FIGURES / "cdmx_hospital_pressure.png"), width=15.2 * cm, height=10.1 * cm),
        Paragraph("Figura 5. Carga vial futura por cama de urgencias ubicada en cada alcaldía.", styles["NoteMX"]),
        Spacer(1, 7),
        _table(pressure_table, [1.2 * cm, 5.0 * cm, 2.5 * cm, 2.1 * cm, 2.2 * cm, 2.4 * cm], font_size=7.5),
        Paragraph("Tabla 5. Cuauhtémoc presenta la mayor razón proyectada. El indicador orienta planeación; no mide ocupación ni saturación clínica.", styles["NoteMX"]),
        PageBreak(),
    ])

    equity = recommendations[recommendations.criterion.str.startswith("equity")].sort_values("recommended_order")
    efficiency = recommendations[recommendations.criterion.eq("efficiency_1")].iloc[0]
    location_table = [["Orden", "Alcaldía", "Latitud", "Longitud", "P90 antes", "P90 después", ">5 km después"]] + [[str(int(r.recommended_order)), r.alcaldia, f"{r.latitude:.6f}", f"{r.longitude:.6f}", f"{r.worst_borough_p90_before_km:.2f}", f"{r.worst_borough_p90_after_km:.2f}", f"{r.risk_share_beyond_5km_after:.2%}"] for r in equity.itertuples()]
    story.extend([
        Paragraph("5. ¿Dónde ubicar el siguiente hospital?", styles["SectionMX"]),
        Paragraph("La regla principal minimiza el peor percentil 90 de distancia entre alcaldías; en caso de empate minimiza la distancia media ponderada. Tras añadir cada hospital se vuelve a optimizar. Los candidatos son centros de celdas de 1 km dentro de CDMX y a por lo menos 2 km de la red actual.", styles["BodyText"]),
        RLImage(str(FIGURES / "cdmx_new_hospital_locations.png"), width=14.7 * cm, height=14.7 * cm),
        Paragraph("Figura 6. Estrellas: secuencia recomendada por equidad. Rombo: alternativa de eficiencia global.", styles["NoteMX"]),
        PageBreak(),
        Paragraph("Secuencia y capacidad de referencia", styles["SectionMX"]),
        _table(location_table, [1.1 * cm, 3.5 * cm, 2.4 * cm, 2.5 * cm, 2.1 * cm, 2.2 * cm, 2.4 * cm], font_size=7.4),
        Paragraph(f"Tabla 6. Primero Milpa Alta y después Tlalpan. El peor P90 baja de {equity.iloc[0].worst_borough_p90_before_km:.2f} km a {equity.iloc[-1].worst_borough_p90_after_km:.2f} km. La alternativa de eficiencia está en {efficiency.alcaldia} ({efficiency.latitude:.6f}, {efficiency.longitude:.6f}) y reduce la distancia media de {efficiency.weighted_mean_distance_before_km:.2f} a {efficiency.weighted_mean_distance_after_km:.2f} km.", styles["NoteMX"]),
        Spacer(1, 8),
        Paragraph("El escenario para comparar una unidad nueva usa la mediana observada: 10 camas de urgencias, 2 médicos urgenciólogos y 160 enfermeras en contacto. Sirve para dimensionar la prueba de localización; no sustituye un programa médico-arquitectónico.", styles["BodyText"]),
        Paragraph("Las coordenadas son centros de cuadrícula, no predios disponibles. Antes de invertir deben verificarse suelo, accesos, tiempos viales, población flotante, costos, ambulancias y capacidad de IMSS, ISSSTE, sector privado y Cruz Roja.", styles["BodyText"]),
    ])

    story.extend([
        Paragraph("6. SINBA, límites y reproducibilidad", styles["SectionMX"]),
        Paragraph("El cubo SINBA de urgencias 2025 contiene una medida de atenciones y dimensiones por unidad médica, CLUES, tipo de urgencia y afección principal. El portal la publica mediante Microsoft Office Web Components y conexión nativa de Analysis Services; no ofrece una descarga CSV ni un endpoint HTTP reproducible por unidad. Se registraron sus metadatos, pero no se usaron cifras de ocupación.", styles["BodyText"]),
        Paragraph("Por ello, presión hospitalaria significa carga vial predicha por recurso disponible. No significa pacientes por cama, ocupación, tiempos de espera ni saturación. Cuando DGIS publique una extracción estable por CLUES, deberá agregarse atenciones anuales por hospital y calcular urgencias por cama y por turno.", styles["NoteMX"]),
        Paragraph("Limitaciones principales", styles["SectionMX"]),
        Paragraph("La ubicación usa distancia euclidiana y patrones puntuales 2019-2024; el pronóstico conserva resolución de alcaldía. ATUS excluye carreteras federales. La red hospitalaria analizada no es todo el sistema de salud. El horizonte contiene meses puente y su incertidumbre aumenta con cada paso recursivo.", styles["BodyText"]),
        Paragraph("Fuentes y archivos", styles["SectionMX"]),
        Paragraph("Fuentes oficiales: INEGI ATUS anual 2018-2025 y georreferenciado 2019-2024; Secretaría de Salud, CLUES agosto de 2026; SEDESA, cartera hospitalaria 2025; DGIS, SINERHIAS y SINBA; INEGI, Marco Geoestadístico.", styles["BodyText"]),
        Paragraph("Tablas reproducibles: cdmx_emergency_hospital_capacity_2026.csv, cdmx_borough_hospital_pressure_future.csv, cdmx_hospital_pressure_future.csv y cdmx_new_hospital_candidates.csv. La metodología completa está en docs/metodologia_actualizada.md.", styles["BodyText"]),
        Paragraph("Conclusión", styles["SectionMX"]),
        Paragraph(f"Para {forecast_label}, Cuauhtémoc encabeza la presión por cama de urgencias. Para cerrar la peor brecha territorial, la secuencia analítica propone primero Milpa Alta y después Tlalpan. Estas ubicaciones pasan a una fase de factibilidad vial, clínica, financiera y predial.", styles["BodyText"]),
        Paragraph(f"Se generaron {len(forecast)} predicciones alcaldía-mes y se integraron recursos oficiales para {len(hospitals)} hospitales.", styles["NoteMX"]),
    ])
    doc.build(story)
    return output
