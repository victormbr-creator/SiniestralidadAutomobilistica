# ¿Qué siniestros matan?

Repositorio académico para estudiar qué características observables de un siniestro vial en México predicen víctimas (personas heridas o fallecidas) frente a daños materiales únicamente. El objetivo de política pública es comparar la priorización territorial por frecuencia de siniestros con la priorización por riesgo de severidad.

**Estado:** estructura y pipeline inicial. No se han descargado datos ni entrenado modelos. Los nombres de columnas, URLs y supuestos de cobertura deben verificarse con documentación oficial antes de ejecutar el análisis.

## Fuentes previstas

| Fuente | Uso previsto | Estado |
| --- | --- | --- |
| ATUS, INEGI | Siniestros individuales y targets | Pendiente de verificar archivos y esquema por año |
| VMRC, INEGI | Parque vehicular municipal | Pendiente de verificar claves y cobertura |
| CONAPO | Población municipal | Pendiente de verificar serie y concordancia territorial |
| Defunciones, INEGI/SINAIS | Contraste externo de mortalidad | Pendiente de definir correspondencia |
| SSC/C5, Ciudad de México | Validación espacial adicional | Opcional; pendiente de factibilidad |

## Estructura

`config/` contiene mapeos y fuentes verificadas; `src/siniestros_viales/` el código del pipeline; `scripts/` las CLI; `data/` los datos según etapa; `outputs/` los resultados generados; `notebooks/` la exploración; `dashboard/` la app; `tests/` las pruebas; `docs/` los protocolos; `paper/` los insumos de publicación.

## Instalación desde cero

Requiere **Python 3.11+**. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make test
```

`make install` instala el paquete en modo editable y las dependencias de desarrollo. En otros sistemas, active el entorno virtual con el comando apropiado. Las librerías geoespaciales pueden necesitar dependencias binarias de su plataforma. En macOS, XGBoost requiere OpenMP (`brew install libomp`) antes de entrenar.

## Datos y pipeline

1. Verificar fuentes oficiales y completar [`config/sources.json`](config/sources.json). Cada entrada necesita `url` HTTPS y `filename` relativo a `data/raw/`. `make download` omite archivos existentes; `python scripts/run_download.py --force` los reemplaza.
2. Inspeccionar los archivos ATUS y completar [`config/atus_columns.json`](config/atus_columns.json) con nombres **confirmados**. Se requieren `deaths` e `injuries`; el resto es opcional. Ejemplo de formato: `"deaths": "NOMBRE_CONFIRMADO_EN_ARCHIVO"`.
3. Ejecutar `make clean-data FILES='2018=data/raw/archivo_2018.csv 2019=data/raw/archivo_2019.csv'` con las rutas reales. También puede usar `python scripts/run_cleaning.py YEAR=PATH ...`. Se admiten CSV UTF-8 y Parquet; otros formatos o codificaciones requieren adaptación documentada.
4. Ejecutar `make features` cuando existan los años 2018–2024 limpios. Para un subconjunto exploratorio: `python scripts/run_features.py --years 2018 2019`.
5. Ejecutar `make train`, `make evaluate` y `python scripts/run_interpretation.py` cuando el dataset y los predictores estén validados.
6. Ejecutar `make dashboard` para abrir la app. `make test` corre las pruebas sintéticas.

`data/raw/` conserva archivos originales sin modificar; `data/interim/` contiene un Parquet limpio por año; `data/processed/` contiene el conjunto concatenado y el dataset con variables derivadas. Los datos brutos grandes, los intermedios y los modelos no se versionan en Git. También se excluyen archivos geoespaciales y otros artefactos pesados. Las rutas locales se resuelven desde la raíz del proyecto.

## Metodología prevista

La unidad es el siniestro. `victimas = 1` si el total de personas heridas o fallecidas supera cero; `fatal = 1` si hay al menos una persona fallecida. El entrenamiento usa 2018–2022 y la prueba temporal 2023–2024. Los modelos previstos son regresión logística, Random Forest y XGBoost. El preprocesamiento se ajusta únicamente sobre el entrenamiento; se excluyen identificadores y conteos que definen el target. Se reportarán ROC-AUC, PR-AUC (average precision), precisión, recall, F1, matriz de confusión y calibración cuando haya ambas clases. La interpretación incluirá coeficientes, importancia de variables y SHAP. El análisis municipal comparará frecuencia, proporción de siniestros con víctimas y tasas con denominadores poblacionales/vehiculares verificados. Gi* requiere geometrías y pesos espaciales definidos por el analista. Más detalles y decisiones pendientes en [`docs/methodology.md`](docs/methodology.md).

## Reproducibilidad

La semilla global es 42 y el corte temporal está centralizado en `config.py`. El manifiesto de entrenamiento guarda target, predictores, años y semilla. Los archivos de configuración deben registrar únicamente fuentes y nombres confirmados. Documente cambios por año en [`docs/variable_changes_template.md`](docs/variable_changes_template.md), problemas de calidad en [`docs/data_quality.md`](docs/data_quality.md) y decisiones en [`docs/decisions.md`](docs/decisions.md). Para reproducibilidad estricta futura, fijar versiones tras validar un entorno completo y registrar hashes/licencias de las descargas.

## Próximos pasos

- [ ] Confirmar fuentes oficiales, licencias y formatos de ATUS para 2018–2024.
- [ ] Completar el mapeo de columnas por año y documentar cambios de esquema.
- [ ] Auditar faltantes, duplicados, cobertura y posibles subregistros.
- [ ] Revisar predictores temporalmente disponibles y riesgos de fuga antes del entrenamiento.
- [ ] Verificar llaves y denominadores municipales de VMRC/CONAPO y obtener geometrías oficiales.
