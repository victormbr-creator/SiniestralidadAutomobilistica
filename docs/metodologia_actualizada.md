# Metodología actualizada: pronóstico, presión hospitalaria y ubicación

## Pregunta de política pública

El proyecto estima en qué alcaldías de Ciudad de México habrá más siniestros viales con personas heridas o fallecidas y dónde conviene evaluar hospitales de urgencias adicionales. La predicción tiene resolución alcaldía-mes; las coordenadas propuestas son centros de celdas analíticas de 1 km y requieren validación de predio, uso de suelo, red vial y operación.

## Pronóstico y tratamiento de la pandemia

La serie mensual abarca 2018-2025. Se conserva 2020 porque eliminarlo ocultaría un cambio real de movilidad y registro. No se usa una variable `pandemia` para proyectar, pues sería una etiqueta conocida después del hecho y no una condición futura recurrente.

La selección de modelo usa validación temporal progresiva:

1. Entrenamiento 2018-2020 y validación 2021.
2. Entrenamiento 2018-2021 y validación 2022.
3. Entrenamiento 2018-2022 y validación 2023.
4. Prueba final congelada 2024-2025, que no participa en la selección.

Los modelos candidatos son binomial negativa, Random Forest Poisson y XGBoost Poisson. La regla predefinida selecciona el menor MAE medio en 2021-2023. Después se reajusta el ganador con 2018-2025 y se pronostica de forma recursiva. Los meses entre diciembre de 2025 y el inicio del horizonte solicitado se generan como puente y aumentan la incertidumbre.

Los hiperparámetros evaluados son:

- Random Forest: 300 árboles, criterio Poisson y `min_samples_leaf` en 2, 5 o 10.
- XGBoost: objetivo `count:poisson`, 300 árboles, profundidad máxima en 2, 3 o 4, tasa de aprendizaje 0.04, submuestreo 0.85 y submuestreo de columnas 0.85.
- Binomial negativa: enlace logarítmico, dispersión `alpha=1` y rezagos transformados con `log1p`.

En la ejecución documentada, XGBoost con profundidad 2 obtuvo el menor MAE medio de validación (3.642), seguido de Random Forest (3.654). El manifiesto guarda el modelo y los parámetros exactos para evitar elegirlos con la prueba final.

Los predictores son rezago de un mes, promedio de doce meses, rezago de alcaldías contiguas, estacionalidad mensual, tendencia e identidad de alcaldía. Se excluyen variables sin cobertura mensual homogénea, variables conocidas después del siniestro y datos de otros estados.

## Capacidad de urgencias

La cartera SEDESA 2025 identifica 34 hospitales con al menos un servicio de urgencias. Todos se vincularon uno a uno por CLUES con SINERHIAS. Para IMSS-Bienestar, los recursos tienen corte al 15 de mayo de 2026. Se conservan por separado:

- camas en área de urgencias;
- camas censables y no censables;
- médicos urgenciólogos y médicos totales;
- enfermeras en contacto con pacientes;
- ambulancias de traslado y cuidados intensivos;
- quirófanos, tomógrafos y camas de cuidados intensivos.

No se suman recursos heterogéneos en un índice compuesto. La presión vial proyectada se presenta como tres razones paralelas:

\[
P^{cama}_a = \frac{D_a}{C^{urg}_a},\qquad
P^{med}_a = \frac{D_a}{M^{urg}_a},\qquad
P^{enf}_a = \frac{D_a}{E_a}
\]

donde \(D_a\) es el número predicho de siniestros con víctimas en la alcaldía, \(C^{urg}_a\) sus camas de urgencias, \(M^{urg}_a\) sus médicos urgenciólogos y \(E_a\) su enfermería en contacto. Un denominador cero se reporta como dato no calculable y no se imputa.

El cubo SINBA 2025 confirma que existen atenciones por unidad y CLUES, pero su publicación depende de Microsoft Office Web Components y una conexión nativa de Analysis Services. No hay una descarga CSV ni un endpoint HTTP reproducible. Por ello no se afirma ocupación, saturación ni utilización observada. `data/raw/sinba/cube_metadata.json` registra la información verificable y la limitación.

## Demanda puntual y accesibilidad

El pronóstico de cada alcaldía se distribuye entre sus siniestros georreferenciados de 2019-2024, proporcionalmente al número observado de personas heridas y fallecidas, con peso mínimo uno. Esto conserva exactamente el total predicho por alcaldía y usa los puntos solo para ubicar la demanda dentro del territorio.

Cada punto se asigna al hospital actual más cercano mediante distancia euclidiana en EPSG:32614. Esta aproximación sirve para comparar alternativas, pero no representa tiempo de ambulancia ni congestión vial.

## Regla para hospitales adicionales

Los candidatos son centros de una cuadrícula de 1 km dentro de CDMX y al menos a 2 km de un hospital existente. La recomendación principal sigue una regla de equidad lexicográfica:

1. minimizar el peor percentil 90 de distancia entre las 16 alcaldías;
2. desempatar con la menor distancia media ponderada por demanda;
3. añadir la ubicación seleccionada y recalcular antes de elegir la siguiente.

También se reporta una alternativa de eficiencia que minimiza la distancia media ponderada de toda la ciudad. Para dimensionar cada unidad nueva se usa un escenario explícito igual a la mediana de los 34 hospitales actuales: 10 camas de urgencias, 2 médicos urgenciólogos y 160 enfermeras en contacto. Es un escenario comparativo, no un programa médico-arquitectónico.

## Criterio de decisión

La secuencia recomendada se acepta para estudio de factibilidad si reduce el peor P90 territorial y la proporción de demanda a más de 5 km. La decisión final requiere añadir tiempos por red vial y hora del día, disponibilidad de predios, población flotante, unidades IMSS/ISSSTE/privadas, bases de ambulancias, ocupación observada y costos.

## Productos reproducibles

- `outputs/tables/cdmx_emergency_hospital_capacity_2026.csv`
- `outputs/tables/cdmx_borough_hospital_pressure_future.csv`
- `outputs/tables/cdmx_hospital_pressure_future.csv`
- `outputs/tables/cdmx_new_hospital_candidates.csv`
- `outputs/figures/cdmx_hospital_pressure.png`
- `outputs/figures/cdmx_new_hospital_locations.png`
