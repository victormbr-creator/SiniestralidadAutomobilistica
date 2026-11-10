# Metodología actualizada: pronóstico, presión hospitalaria y ubicación

## Pregunta de política pública

El proyecto estima en qué alcaldías de Ciudad de México habrá más siniestros viales con personas heridas o fallecidas y dónde conviene evaluar clínicas de urgencias adicionales. El modelo temporal tiene resolución alcaldía-mes. Para decidir ubicaciones, el pronóstico se desagrega en una cuadrícula de 1 km usando los puntos históricos; las coordenadas propuestas son centros de celdas analíticas y requieren validación de predio, uso de suelo, red vial y operación.

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

## Capacidad y flujo de urgencias

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

DGIS publica los registros nacionales anuales de urgencias en archivos ZIP. Se usan exclusivamente las tablas `URGENCIAS` de 2018-2025; se excluyen los módulos de afecciones, medicamentos, procedimientos, consulta externa y egresos. Los archivos se leen secuencialmente sin extraerlos. El vínculo por CLUES conserva la clave actual y la clave anterior del catálogo oficial para seguir los 34 hospitales a través del cambio institucional a IMSS-Bienestar. El repositorio conserva agregados mensuales por hospital, no los microdatos nacionales.

La cobertura por año se publica de forma explícita: 31 hospitales registraron atenciones en 2018-2020, 33 en 2021 y los 34 en 2022-2025. Una ausencia histórica no se convierte en atención cero.

Para 2025 se calculan, por hospital y alcaldía:

- atenciones de urgencias totales, calificadas y no calificadas;
- motivo de atención por accidentes, envenenamientos o violencia, sin interpretarlo como siniestro vial;
- uso registrado de cama de observación o choque;
- ingresos hospitalarios, traslados y defunciones en urgencias;
- horas registradas de estancia y horas en cama para los registros con duración válida.

El flujo observado por cama es

\[
F^{cama}_h = \frac{U_h}{C^{urg}_h},
\]

donde \(U_h\) son las atenciones de urgencias de 2025. Como indicador complementario se calcula

\[
O^{proxy}_h = \frac{H^{cama}_h}{C^{urg}_h\,24\,365},
\]

donde \(H^{cama}_h\) suma la estancia registrada de atenciones con cama de observación o choque. Es una ocupación aproximada: no observa censos horarios, camas cerradas, espera, personal por turno ni simultaneidad. Los valores faltantes y códigos `999:99` no se imputan.

La presión vial proyectada y el flujo asistencial observado se mantienen separados. La primera aproxima demanda futura atribuible al riesgo vial; el segundo describe toda la actividad registrada en urgencias, cualquiera que sea su causa.

Para priorizar ampliaciones en hospitales existentes, cada razón por cama se transforma en percentil dentro de la red y se promedian ambos percentiles. Este `dual_pressure_score` identifica unidades con flujo total alto y exposición vial futura alta sin sumar pacientes y siniestros en una unidad artificial. Se publica como tamizaje; la inversión exige comprobar turnos, ocupación censada y capacidad efectiva.

## Demanda puntual y cuadrícula de 1 km

El pronóstico de cada alcaldía se distribuye entre sus siniestros georreferenciados de 2019-2024, proporcionalmente al número observado de personas heridas y fallecidas, con peso mínimo uno. Los puntos ponderados se agregan a celdas de 1 km en EPSG:32614. Esta operación conserva exactamente el total predicho de cada alcaldía y de toda la ciudad.

La cuadrícula aprovecha la localización granular para describir dónde se ha concentrado históricamente el riesgo dentro de cada alcaldía. No convierte cada punto en un evento futuro ni constituye un modelo temporal entrenado por celda. La distancia es euclidiana y sirve para comparar alternativas de forma reproducible; no representa tiempo de ambulancia ni congestión vial.

## Regla para hospitales adicionales

Los candidatos son centros de celdas de 1 km dentro de CDMX y al menos a 2 km de un hospital existente. La oferta física se distribuye espacialmente antes de comparar demanda y capacidad. Para cada hospital \(j\) y celda \(g\), el peso de acceso es

\[
w_{gj} =
\begin{cases}
\exp(-d_{gj}/3000), & d_{gj} \leq 10000\\
0, & d_{gj} > 10000,
\end{cases}
\]

donde \(d_{gj}\) es la distancia en metros. Las camas accesibles equivalentes en cada celda son

\[
C_g = \sum_j C_j\frac{w_{gj}}{\sum_q w_{qj}}.
\]

La normalización distribuye cada hospital entre sus celdas alcanzables y garantiza que \(\sum_g C_g = \sum_j C_j\). La presión local usada en el Opportunity Atlas es

\[
P_g = \frac{D_g}{C_g},
\]

donde \(D_g\) es la demanda vial predicha de la celda. Esta presión bruta se mide en siniestros predichos por cama accesible. El mapa divide cada valor entre la mediana ponderada de presión y presenta un índice relativo sin unidades: 1 representa la mediana, verde indica menor presión y rojo mayor presión. Para conservar contraste, el color rojo se satura por encima del percentil 95. Por eso la escala del mapa no coincide numéricamente con el P90 bruto de la tabla de ubicaciones.

Las 236 celdas grises no tienen camas de urgencias a 10 km. En la ejecución actual concentran cero demanda predicha porque no contienen puntos con víctimas de 2019-2024 a los cuales desagregar el pronóstico. Se localizan principalmente en Tlalpan y Milpa Alta. No reciben peso en el objetivo ponderado por demanda; el resultado no significa que estén cubiertas ni que carezcan de población. Esta limitación debe revisarse con población, red vial y tiempos de ambulancia antes de tomar una decisión.

Cada clínica nueva se evalúa con un escenario igual a la mediana de los 34 hospitales actuales: 10 camas de urgencias, 2 médicos urgenciólogos y 160 enfermeras en contacto. Sus 10 camas se incorporan a la superficie con la misma función de accesibilidad. Como la red existente suma 364 camas de urgencias, es esperable que una sola unidad produzca una reducción moderada. La selección es lexicográfica:

1. minimizar el percentil 90 de presión de la ciudad, ponderado por demanda predicha;
2. desempatar con la menor presión media ponderada;
3. desempatar con la menor distancia media ponderada a una unidad;
4. añadir la clínica seleccionada, recalcular oferta y presión, y repetir para la segunda.

La distancia participa en la función de accesibilidad y solo aparece como tercer desempate. La demanda futura y la oferta de camas determinan el criterio principal.

El flujo observado se usa para identificar hospitales actuales que requieren ampliación o revisión operativa. No se distribuye en la cuadrícula porque los archivos abiertos no incluyen el origen geográfico preciso del paciente. Asignar todas las atenciones al hospital receptor atribuiría demanda a la zona equivocada.

## Criterio de decisión

La secuencia recomendada se acepta para estudio de factibilidad si reduce el P90 y la media ponderada de presión. La decisión final requiere añadir tiempos por red vial y hora del día, disponibilidad de predios, población flotante, unidades IMSS/ISSSTE/privadas, bases de ambulancias, ocupación observada y costos.

## Productos reproducibles

- `outputs/tables/cdmx_emergency_hospital_capacity_2026.csv`
- `outputs/tables/cdmx_emergency_flow_monthly_2018_2025.csv`
- `outputs/tables/cdmx_emergency_flow_coverage.csv`
- `outputs/tables/cdmx_hospital_observed_emergency_pressure_2025.csv`
- `outputs/tables/cdmx_borough_observed_emergency_flow_2025.csv`
- `outputs/tables/cdmx_borough_hospital_pressure_future.csv`
- `outputs/tables/cdmx_hospital_pressure_future.csv`
- `outputs/tables/cdmx_opportunity_grid_future.csv`
- `outputs/tables/cdmx_new_hospital_candidates.csv`
- `outputs/figures/cdmx_hospital_pressure.png`
- `outputs/figures/cdmx_observed_emergency_flow_pressure.png`
- `outputs/figures/cdmx_new_hospital_locations.png`
- `outputs/figures/cdmx_opportunity_atlas_pressure.png`
