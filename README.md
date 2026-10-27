# ¿En qué alcaldías habrá mayor peligro vial?

Pipeline reproducible para predecir siniestros viales con personas heridas o fallecidas en las **16 alcaldías de Ciudad de México** y comparar el riesgo esperado con la infraestructura SEDESA reportada. El estudio es predictivo, no causal.

## Estado actual

ATUS anual 2018-2025 fue procesado, pero el análisis sustantivo usa exclusivamente registros de Ciudad de México. La unidad predictiva es alcaldía-mes. El proyecto incorpora ATUS georreferenciado 2019-2024, CLUES, la cartera SEDESA 2025, recursos SINERHIAS 2026 y el flujo oficial de urgencias DGIS/SINBA 2018-2025. Los 34 hospitales con urgencias fueron vinculados por CLUES con camas, personal, equipo y atenciones observadas.

## Unidad y diseño

- Panel de las 16 alcaldías por mes, 2018-2025.
- Conteo principal: siniestros con `muertos_total + heridos_total > 0`.
- Binario secundario: al menos un siniestro con víctimas en la unidad-mes.
- Validación temporal progresiva 2021-2023; prueba final 2024-2025 con rezagos congelados.
- Pronóstico móvil de los próximos 12 meses, reajustado con toda la historia observada hasta 2025.
- El mes inicial y la longitud del horizonte son configurables; el valor predeterminado comienza el mes siguiente a la ejecución.
- Mapa puntual observado de 8,820 siniestros con víctimas, 2019-2024.
- Distancia directa desde cada siniestro al hospital con urgencias verificado más cercano.
- Presión vial proyectada por cama de urgencias, médico urgenciólogo y enfermería.
- Flujo observado exclusivamente de urgencias: atenciones, urgencias calificadas, ingresos, traslados, defunciones y horas de estancia.
- Atenciones de urgencias por cama y ocupación aproximada basada en horas registradas de cama de observación o choque.
- Secuencia de hospitales adicionales mediante una cuadrícula de 1 km y criterio de equidad territorial.

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
4. Ejecute `make prepare panel geospatial spatial train pressure emergency-flow report`, o `make all`.
5. Revise las tablas `cdmx_*`, las figuras CDMX y el PDF final.

Cada comando escribe logs en `outputs/logs/`. Los scripts se ejecutan desde la raíz y no utilizan rutas absolutas.

Para fijar un horizonte reproducible:

```bash
python scripts/run_cdmx_analysis.py --forecast-start 2026-11 --horizon-months 12
python scripts/build_report.py
```

Si el horizonte empieza después del último mes observado, el pipeline genera meses puente de forma recursiva y registra su número en `outputs/models/cdmx_manifest.json`.

## Modelos y evaluación

Se comparan promedio móvil de 12 meses, binomial negativa, Random Forest y XGBoost Poisson. El modelo final se selecciona por el menor MAE medio en validación progresiva 2021-2023; la prueba 2024-2025 no participa en esa decisión. Los predictores incluyen rezagos propios y de alcaldías vecinas. Se calculan MAE, devianza Poisson, captura top 5%/10% y diferencias contra baseline mediante bootstrap por bloques de alcaldía.

## Límites sustantivos

ATUS excluye carreteras federales; la ausencia de registros no prueba ausencia de siniestros. La población censal 2020 es fija. El mapa puntual es histórico y la predicción conserva resolución de alcaldía. La incertidumbre aumenta con cada mes recursivo entre el último dato observado y el horizonte objetivo. Las distancias a hospitales son euclidianas y no equivalen a tiempo de ambulancia. El flujo DGIS describe atenciones registradas y no contiene el origen geográfico preciso del paciente. La ocupación calculada es aproximada y no sustituye censos horarios ni tiempos de espera. La red analizada cubre SEDESA/IMSS-Bienestar y no incluye todo IMSS, ISSSTE, sector privado, Cruz Roja ni bases de ambulancias.

La definición completa de entrenamiento, pandemia, presión y ubicación está en [`docs/metodologia_actualizada.md`](docs/metodologia_actualizada.md).
