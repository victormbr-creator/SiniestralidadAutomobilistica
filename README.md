# ¿En qué alcaldías habrá mayor peligro vial?

Pipeline reproducible para predecir siniestros viales con personas heridas o fallecidas en las **16 alcaldías de Ciudad de México** y comparar el riesgo esperado con la infraestructura SEDESA reportada. El estudio es predictivo, no causal.

## Estado actual

ATUS anual 2018-2024 fue procesado, pero el análisis sustantivo usa exclusivamente registros de Ciudad de México. La unidad predictiva es alcaldía-mes. El proyecto también incorpora ATUS georreferenciado 2019-2024, dos cortes CLUES y la cartera SEDESA 2025: 34 hospitales tienen al menos un servicio de urgencias explícitamente marcado y fueron vinculados con coordenadas CLUES.

## Unidad y diseño

- Panel de las 16 alcaldías por mes, 2018-2024.
- Conteo principal: siniestros con `muertos_total + heridos_total > 0`.
- Binario secundario: al menos un siniestro con víctimas en la unidad-mes.
- Entrenamiento 2018-2022; prueba final 2023-2024 con rezagos congelados.
- Predicción operativa 2025, reajustada con 2018-2024.
- Horizonte dinámico de un mes y horizonte congelado 2023-2024 con información disponible hasta 2022.
- Mapa puntual observado de 8,820 siniestros con víctimas, 2019-2024.
- Distancia directa desde cada siniestro al hospital con urgencias verificado más cercano.

## Instalación

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
make test
```

En macOS, XGBoost suele requerir `brew install libomp`. Las dependencias geoespaciales pueden requerir binarios del sistema.

## Ejecución

1. Complete `config/sources.yml`, `config/project.yml` y `data/schema_changes.csv` con documentación verificada.
2. Copie descargas originales a `data/raw/` sin modificarlas y complete `data/README.md`.
3. Ejecute `make validate` y corrija todos los bloqueos.
4. Ejecute `make prepare panel geospatial spatial train evaluate coverage report`, o `make all`.
5. Ejecute `make train report` y revise las tablas `cdmx_*`, las figuras CDMX y el PDF final.

Cada comando escribe logs en `outputs/logs/`. Los scripts se ejecutan desde la raíz y no utilizan rutas absolutas.

## Modelos y evaluación

Se comparan promedio móvil de 12 meses, binomial negativa, Random Forest y XGBoost Poisson. Los predictores incluyen rezagos propios y de alcaldías vecinas. Se calculan MAE, devianza Poisson, captura top 5%/10% y diferencias contra baseline mediante bootstrap por bloques de alcaldía.

## Límites sustantivos

ATUS excluye carreteras federales; la ausencia de registros no prueba ausencia de siniestros. La población censal 2020 es fija. El mapa puntual es histórico y la predicción conserva resolución de alcaldía. Las distancias a hospitales son euclidianas y no equivalen a tiempo de ambulancia. La red de urgencias cubre SEDESA/IMSS-Bienestar; no incluye todo IMSS, ISSSTE, sector privado, Cruz Roja ni bases de ambulancias.
