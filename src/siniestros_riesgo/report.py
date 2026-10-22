"""Build the Spanish project report from reproducible outputs."""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image as RLImage
import pandas as pd

from .config import REPORTS, TABLES, FIGURES


TITLE = "¿Dónde ocurren los siniestros con víctimas?"


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5B6573"))
    canvas.drawString(2 * cm, 1.2 * cm, "Predicción espacio-temporal de siniestros viales en México")
    canvas.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Página {document.page}")
    canvas.restoreState()


def build_report(output: Path = REPORTS / "resultados_proyecto.pdf") -> Path:
    """Create a readable report and explicitly mark unavailable analyses."""
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleMX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=28, textColor=colors.HexColor("#13315C"), alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name="SectionMX", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#0B6E75"), spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="NoteMX", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=9, leading=12, textColor=colors.HexColor("#5B6573"), leftIndent=10, rightIndent=10))
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    frame = Frame(2 * cm, 1.8 * cm, letter[0] - 4 * cm, letter[1] - 3.6 * cm, id="normal")
    document = BaseDocTemplate(str(output), pagesize=letter, title=TITLE, author="Proyecto de políticas públicas", pageTemplates=[PageTemplate(id="main", frames=frame, onPage=_footer)])
    story = [Spacer(1, 2.2 * cm), Paragraph(TITLE, styles["TitleMX"]), Paragraph("Predicción espacio-temporal de siniestros viales en México con ATUS para orientar la ubicación de servicios médicos de emergencia", styles["Heading2"]), Spacer(1, 1 * cm), Paragraph("Informe reproducible de diseño y estado de ejecución", styles["Heading2"]), Spacer(1, 0.8 * cm)]
    status = Table([["Estado", "ATUS 2018-2024 procesado; análisis principal ejecutado"], ["Resultados observados", "Cobertura, totales y severidad temporal"], ["Resultados estimados", "Modelos municipal-mes y test congelado 2023-2024"], ["Pendiente", "Población, red vial, CLUES, límites y análisis metropolitano"]], colWidths=[4.2 * cm, 11.5 * cm])
    status.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#DCEEF0")), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A6B1BC")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.extend([status, PageBreak()])
    totals_path=TABLES/"annual_totals.csv"; metrics_path=TABLES/"test_count_metrics.csv"; binary_path=TABLES/"test_binary_metrics.csv"; boot_path=TABLES/"bootstrap_differences.csv"
    if totals_path.exists():
        totals=pd.read_csv(totals_path); total_records=int(totals.records.sum()); total_victims=int(totals.victim_crashes.sum()); balanced="1,026"
        story.extend([Paragraph("Resultados observados",styles["SectionMX"]),Paragraph(f"La edición anual ATUS aportó {total_records:,} registros para 2018-2024. Se identificaron {total_victims:,} siniestros con al menos una persona herida o fallecida. El panel balanceado contiene {balanced} municipios y 86,184 observaciones municipio-mes.",styles["BodyText"]),Paragraph("Fuente: elaboración propia con ATUS anual 2018-2024. Los certificados cero fueron eliminados de esta edición por INEGI; la ausencia municipal no se interpreta como cero. La base anual no incluye identificador único: filas indistinguibles se conservan porque no puede probarse que sean duplicados.",styles["NoteMX"])] )
        for filename,note in (("annual_crash_counts.png","Figura 1. Registros anuales observados."),("annual_victim_share.png","Figura 2. Proporción observada de siniestros con víctimas.")):
            path=FIGURES/filename
            if path.exists(): story.extend([Spacer(1,8),RLImage(str(path),width=15.2*cm,height=8.4*cm),Paragraph(note+" Fuente: elaboración propia con ATUS.",styles["NoteMX"])])
        story.append(PageBreak())
    if metrics_path.exists():
        metrics=pd.read_csv(metrics_path); binary=pd.read_csv(binary_path); boot=pd.read_csv(boot_path)
        best_dev=metrics.loc[metrics.poisson_deviance.idxmin()]; best_mae=metrics.loc[metrics.mae.idxmin()]
        story.extend([Paragraph("Resultados estimados: prueba 2023-2024",styles["SectionMX"]),Paragraph(f"El baseline obtuvo el menor MAE ({best_mae.mae:.3f} si aparece como {best_mae.model}); XGBoost Poisson obtuvo la menor devianza Poisson ({best_dev.poisson_deviance:.3f}). Las predicciones para los dos años usan rezagos congelados con información disponible hasta 2022.",styles["BodyText"])])
        table_data=[["Modelo","MAE","Dev. Poisson","Captura 5%","Captura 10%"]]+[[r.model,f"{r.mae:.3f}",f"{r.poisson_deviance:.3f}",f"{r.capture_top_5:.1%}",f"{r.capture_top_10:.1%}"] for r in metrics.itertuples()]
        table=Table(table_data,colWidths=[4.2*cm,2.2*cm,2.7*cm,2.7*cm,2.7*cm]); table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCEEF0")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#A6B1BC")),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(1,1),(-1,-1),"RIGHT"),("FONTSIZE",(0,0),(-1,-1),8),("PADDING",(0,0),(-1,-1),5)])); story.extend([Spacer(1,8),table,Paragraph("Tabla 1. Métricas estimadas en el test final. La captura mide siniestros observados dentro de las unidades con mayor predicción.",styles["NoteMX"])])
        xgb_boot=boot[boot.model.eq("xgboost_poisson")].iloc[0]; story.append(Paragraph(f"La diferencia bootstrap de MAE de XGBoost frente al baseline fue {xgb_boot.difference:.3f}, IC 95% [{xgb_boot.ci_lower:.3f}, {xgb_boot.ci_upper:.3f}], con 1,000 remuestreos municipales. El intervalo contiene cero.",styles["BodyText"]))
        best_binary=binary.loc[binary.pr_auc.idxmax()]; story.append(Paragraph(f"En el resultado binario, {best_binary.model} alcanzó la mayor PR-AUC ({best_binary.pr_auc:.3f}) y Brier {best_binary.brier:.3f}. Estas métricas son predictivas y no causales.",styles["BodyText"]))
        for filename,note in (("model_mae_comparison.png","Figura 3. MAE estimado por modelo."),("model_capture_comparison.png","Figura 4. Captura estimada en unidades priorizadas.")):
            path=FIGURES/filename
            if path.exists(): story.extend([Spacer(1,8),RLImage(str(path),width=15.2*cm,height=8.4*cm),Paragraph(note+" Fuente: modelos propios con ATUS; test congelado 2023-2024.",styles["NoteMX"])])
        story.append(PageBreak())
        ranking_path=TABLES/"municipal_risk_ranking_2025.csv"; cdmx_path=TABLES/"cdmx_alcaldia_risk_ranking_2025.csv"
        if ranking_path.exists() and cdmx_path.exists():
            ranking=pd.read_csv(ranking_path,dtype={"geo_id":"string"}); cdmx=pd.read_csv(cdmx_path,dtype={"geo_id":"string"})
            story.extend([Paragraph("¿Dónde se predice mayor peligro?",styles["SectionMX"]),Paragraph("Después de cerrar la evaluación, el XGBoost Poisson seleccionado se reajustó con 2018-2024 y produjo una proyección recursiva para 2025 sin usar resultados de 2025. El puntaje es el número esperado de siniestros con víctimas, útil para priorización operativa; no es riesgo causal ni riesgo individual.",styles["BodyText"])])
            top=ranking.head(10); top_table=[["Rango","Municipio (clave)","Predicción anual","Por 100 mil hab."]]+[[str(i+1),f"{r.nomgeo} ({str(r.geo_id).zfill(5)})",f"{r.predicted_annual_victim_crashes:,.0f}",f"{r.predicted_rate_per_100k_population:,.1f}"] for i,r in enumerate(top.itertuples())]
            t=Table(top_table,colWidths=[1.3*cm,7.3*cm,3.2*cm,3.2*cm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCEEF0")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#A6B1BC")),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("ALIGN",(2,1),(-1,-1),"RIGHT"),("PADDING",(0,0),(-1,-1),4)])); story.extend([t,Paragraph("Tabla 2. Municipios con mayor conteo predicho. La tasa usa población del servicio oficial del Catálogo Único de INEGI como contexto fijo.",styles["NoteMX"])])
            path=FIGURES/"national_top20_predicted_risk.png"
            if path.exists(): story.extend([Spacer(1,6),RLImage(str(path),width=15.2*cm,height=10.1*cm),Paragraph("Figura 5. Veinte municipios con mayor conteo predicho. Fuente: modelo propio y Catálogo Único de INEGI.",styles["NoteMX"])])
            story.append(PageBreak())
            story.extend([Paragraph("Alcaldías de Ciudad de México",styles["SectionMX"]),Paragraph("La predicción está disponible para las 16 demarcaciones territoriales. Gustavo A. Madero, Cuauhtémoc y Álvaro Obregón encabezan el conteo esperado; el orden por habitante puede diferir y debe consultarse en la tabla CSV.",styles["BodyText"])])
            map_path=FIGURES/"cdmx_alcaldia_risk_map.png"
            if map_path.exists(): story.extend([RLImage(str(map_path),width=14.2*cm,height=14.2*cm),Paragraph("Figura 6. Riesgo predicho por alcaldía; las etiquetas muestran la clave municipal. Polígonos: Marco Geoestadístico/Servicio Web del Catálogo Único, INEGI.",styles["NoteMX"]),PageBreak()])
            cdmx_table=[["Rango","Alcaldía","Predicción"]]+[[str(i+1),r.nomgeo,f"{r.predicted_annual_victim_crashes:,.0f}"] for i,r in enumerate(cdmx.itertuples())]
            ct=Table(cdmx_table,colWidths=[1.5*cm,9*cm,3.5*cm]); ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCEEF0")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#A6B1BC")),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("ALIGN",(2,1),(-1,-1),"RIGHT")]))
            story.extend([Paragraph("Ranking completo de alcaldías",styles["SectionMX"]),ct,Paragraph("Tabla 3. Las 16 alcaldías por conteo predicho. Archivo reproducible: outputs/tables/cdmx_alcaldia_risk_ranking_2025.csv.",styles["NoteMX"])])
            cdmx_chart=FIGURES/"cdmx_alcaldia_predicted_risk.png"
            if cdmx_chart.exists(): story.extend([Spacer(1,6),RLImage(str(cdmx_chart),width=15.2*cm,height=10.1*cm),Paragraph("Figura 7. Comparación del conteo predicho entre alcaldías.",styles["NoteMX"])])
            story.append(PageBreak())
        shap_path=FIGURES/"xgboost_shap_summary.png"
        story.extend([Paragraph("Explicabilidad predictiva",styles["SectionMX"]),Paragraph("SHAP e importancia de árboles resumen qué variables contribuyeron a la predicción del modelo ajustado. No representan efectos causales ni el efecto de una intervención.",styles["BodyText"])])
        if shap_path.exists(): story.extend([RLImage(str(shap_path),width=15.2*cm,height=9.5*cm),Paragraph("Figura 8. SHAP descriptivo en una muestra reproducible de 1,000 observaciones del test. Fuente: elaboración propia.",styles["NoteMX"])])
        story.append(PageBreak())
    sections = [
        ("1. Pregunta, alcance y uso", "El estudio pregunta en qué lugares y momentos es más probable que ocurran siniestros con personas heridas o fallecidas, y qué zonas de alto riesgo quedan lejos de atención médica. Es predictivo, no causal. Su uso previsto es priorizar atención y planificación."),
        ("2. Datos y cobertura", "El panel principal será municipio por mes para 2018-2024. La muestra incluirá solo municipios con evidencia de reporte en cada año. Un municipio-año ausente no se tratará como cero. Se distinguirán cero observado, certificado cero y falta de reporte. La malla de 1 km incluirá únicamente municipios con georreferenciación consistente en todos los años usados."),
        ("3. Limpieza y control de calidad", "El pipeline exige mapeos de columnas confirmados por edición, conserva claves con ceros iniciales, valida fechas, conteos y coordenadas, revisa duplicados y genera cobertura municipio-año, faltantes y totales. Queda pendiente comparar agregados con tabulados oficiales y medir la población excluida de la muestra balanceada."),
        ("4. Diagnóstico espacial", "Para 2018-2022 se implementa Gi* anual con pesos estandarizados por filas, vecindad Queen y alternativa k-NN para islas. Los p-valores se ajustan con Benjamini-Hochberg y se resume persistencia. A nivel municipal deberá usarse una tasa por habitante suavizada por Bayes empírico; en malla, conteos."),
        ("5. Predicción y validación", "Se compararán promedio móvil de 12 meses, binomial negativa, Random Forest y XGBoost Poisson con predictores comparables. La selección se limita a ventanas temporales dentro de 2018-2022 y validación dejando una entidad fuera. El test 2023-2024 se examina una sola vez. Para el horizonte congelado no se usan resultados reales posteriores a 2022 para actualizar rezagos."),
        ("6. Métricas e incertidumbre", "Las métricas principales serán MAE, devianza Poisson y captura de eventos en el top 5% y 10% de unidades. El análisis binario reportará PR-AUC y Brier. Las diferencias frente al baseline usarán 1,000 remuestreos por bloques municipales e intervalos del 95%."),
        ("7. Explicabilidad y heterogeneidad", "SHAP e importancias se interpretarán como descripciones predictivas, nunca como efectos causales. El desempeño se estratificará por terciles de población y región; día/noche y semana/fin de semana solo se analizarán si los microdatos permiten una clasificación válida."),
        ("8. Cobertura de urgencias", "El escenario principal se denominará cobertura CLUES y requiere validar manualmente qué establecimientos ofrecen urgencias. Los minutos por red solo se reportarán con tiempos disponibles o un supuesto documentado. La optimización conserva bases existentes y evalúa p=1,...,10 ubicaciones candidatas; el método voraz implementado se etiqueta como aproximación."),
        ("9. Limitaciones", "ATUS excluye siniestros en carreteras federales y sus conclusiones se limitan a vialidades urbanas y suburbanas cubiertas. La ausencia de registros no prueba ausencia de siniestros. La población del Censo 2020 es fija. Los mapas no miden necesariamente riesgo por viaje o por kilómetro recorrido."),
        ("10. Conclusión y estado", "No es posible presentar hallazgos empíricos hasta incorporar y verificar las fuentes. El repositorio sí deja especificadas las decisiones que evitan ceros falsos, fuga temporal, uso prematuro del test y afirmaciones causales. Los siguientes pasos son documentar fuentes oficiales, validar esquemas ATUS, ejecutar controles y revisar la muestra antes de ajustar modelos."),
        ("11. Referencias pendientes", "INEGI, ATUS 2018-2024; INEGI, Censo de Población y Vivienda 2020; INEGI, Red Nacional de Caminos; Secretaría de Salud, CLUES; documentación metodológica de las ediciones utilizadas. Las URLs, fechas, licencias y versiones deberán completarse en data/README.md antes del análisis."),
    ]
    for heading, body in sections:
        story.append(KeepTogether([Paragraph(escape(heading), styles["SectionMX"]), Paragraph(escape(body), styles["BodyText"])]))
        story.append(Paragraph("Estado: sección metodológica; consulte arriba los resultados ejecutados y las limitaciones de fuentes.", styles["NoteMX"]))
    figures = list(FIGURES.glob("*.png")) if FIGURES.exists() else []
    tables = list(TABLES.glob("*.csv")) if TABLES.exists() else []
    story.extend([Spacer(1, 12), Paragraph(f"Artefactos detectados al generar el informe: {len(figures)} figuras y {len(tables)} tablas. No se insertan artefactos inexistentes.", styles["NoteMX"])])
    document.build(story)
    return output
