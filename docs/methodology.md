# Metodología prevista

## Preguntas y unidad

Unidad: siniestro vial. Estimar la probabilidad de que un siniestro produzca personas heridas o fallecidas y evaluar si la priorización municipal cambia al considerar severidad junto con frecuencia.

## Resultados y validación

`victimas` y `fatal` se derivan de conteos no negativos confirmados. El target principal es `victimas`; `fatal` es análisis alternativo sujeto a suficiente tamaño de clase. Entrenamiento: 2018–2022. Prueba temporal: 2023–2024. La selección de umbral deberá fijarse en entrenamiento/validación, antes de examinar el test. TODO: definir validación temporal interna, hiperparámetros, intervalo de incertidumbre y tratamiento de cambios de registro.

## Variables y fuga de información

Excluir identificadores, año como predictor directo, conteos de personas heridas/fallecidas y cualquier variable generada después del siniestro que no estaría disponible al momento de intervención. Municipio y entidad quedan como predictores candidatos si sus claves y el objetivo de predicción lo justifican. El listado definitivo depende del diccionario ATUS verificado. El pipeline imputa y codifica con ajuste exclusivo en entrenamiento.

## Evaluación e interpretación

Comparar logit, Random Forest y XGBoost con ROC-AUC, average precision, precisión, recall, F1, matriz de confusión y curvas de calibración. Reportar prevalencia del target y umbral. Interpretar asociaciones con coeficientes, importancia y SHAP, sin afirmaciones causales.

## Territorio

Agregar por claves municipales y año, comprobar concordancia de códigos geográficos y calcular tasas por población y parque vehicular solo cuando existan denominadores compatibles. Documentar qué mide cada tasa: el numerador inicial es **siniestros con víctimas**, no número de personas fallecidas. La prueba Gi* exige una decisión explícita sobre geometrías, matriz de vecindad, tratamiento de islas y comparaciones múltiples.
