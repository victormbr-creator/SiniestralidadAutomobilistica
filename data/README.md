# Datos y procedencia

Este repositorio no contiene datos originales. `data/raw/` recibe copias sin modificar; `data/processed/` contiene derivados reproducibles. Nunca sustituya un archivo raw: añada una nueva versión y registre su hash.

## Registro obligatorio de fuentes

| Fuente | Edición/año | Archivo local | URL oficial | Fecha de descarga | Licencia/términos | SHA-256 | Transformación | Exclusiones |
|---|---:|---|---|---|---|---|---|---|
| ATUS anual | 2018-2025 | `conjunto_de_datos_atus_anual_csv/` | https://www.inegi.org.mx/programas/accidentes/ | Archivos con marca 2026-08-27; incorporados 2026-10-22 | https://www.inegi.org.mx/inegi/terminos.html | Véase abajo | Limpieza, suma de víctimas, panel | 2025 contiene los 12 meses y cifras definitivas |
| ATUS georreferenciado | 2019-2024 | `data/raw/atus_georreferenciado/` | https://www.inegi.org.mx/programas/accidentes/ | 2026-10-22 | https://www.inegi.org.mx/inegi/terminos.html | Véase abajo | Filtro CDMX, validación espacial y puntos con víctimas | No existe publicación georreferenciada 2025 al corte |
| Censo de Población | 2020 | TODO | TODO | TODO | TODO | TODO | Unión municipal; población fija | TODO |
| Red Nacional de Caminos | TODO | TODO | TODO | TODO | TODO | TODO | Longitud total/primaria | TODO |
| CLUES | diciembre 2024 y agosto 2026 | `data/raw/clues/ESTABLECIMIENTO_SALUD_*.xlsx` | https://gobi.salud.gob.mx/Bases_Clues.html | 2026-10-22 | Catálogo público de Secretaría de Salud | Véase abajo | Filtro CDMX, operación vigente y coordenadas válidas | Agosto es el último corte 2026 descargable al 2026-10-22 |
| Cartera hospitalaria SEDESA | 2025 | `data/raw/clues/cartera_servicios_hospitales_segundo_nivel_sedesa_2025.xlsx` | https://www.salud.cdmx.gob.mx/ver-mas/unidades/hospitales | 2026-10-22 | Portal público SEDESA | Véase abajo | Verificación de urgencias y vínculo con CLUES | Red SEDESA/IMSS-Bienestar; no incluye todas las instituciones |
| Marco geoestadístico | TODO | TODO | TODO | TODO | TODO | TODO | Geometrías municipales | TODO |
| Catálogo municipal y población | consulta 2026-10-22 | `data/external/inegi_municipal_catalog.csv` | https://gaia.inegi.org.mx/wscatgeo/v2/mgem/ | 2026-10-22 | Términos INEGI | TODO | Nombres y población contextual | Población fija; validar evento censal |
| Polígonos de alcaldías CDMX | consulta 2026-10-22 | `data/external/cdmx_alcaldias_inegi.geojson` | https://gaia.inegi.org.mx/wscatgeo/v2/geo/mgem/09 | 2026-10-22 | Términos INEGI | TODO | Mapa del horizonte futuro | Solo CDMX |
| Infraestructura de salud por alcaldía | 2020 | `data/raw/infraestructura_salud_alcaldias_cdmx_2020.csv` | URL SEDECO/CDMX consignada dentro del CSV | Incorporado 2026-10-22 | TODO: verificar términos del anexo | `b4661f2d8eed3006b6a04757b6069107b8583f26b8fdefebeff9f3393ea9126f` | Brecha relativa riesgo-capacidad | Red SEDESA; no inventario completo de urgencias ni de IMSS/ISSSTE |
| SINERHIAS, recursos por establecimiento | corte IMSS-Bienestar 2026-05-15 | `data/raw/sinerhias/unidades_cdmx_2026.json` | https://radarcisalud.salud.gob.mx/sinerhias/map_geo/sinerhias_rpe.php | 2026-10-27 | Portal público DGIS | `442ec51b94ad5925fefa2036232d581b189380f3dc242b917c60748db1b6feb6` | Filtro oficial CDMX; unión exacta por CLUES | Recursos SSA/IMB al 2026-05-15; IMSS/ISSSTE/PEMEX al 2025-12-31 |
| DGIS/SINBA, datos abiertos de urgencias | 2018-2025 | `data/raw/sinba/urgencias/Urgencias_*.zip` | http://www.dgis.salud.gob.mx/contenidos/basesdedatos/da_urgencias_gobmx.html | 2026-10-27 | Portal público DGIS | Véase manifiesto | Lectura secuencial; vínculo de CLUES actual/anterior; agregación hospital-mes | Solo tabla URGENCIAS y 34 hospitales; microdatos no versionados |

## Reglas

- ATUS cubre siniestros terrestres en zonas urbanas y suburbanas y excluye carreteras federales; verificar alcance exacto en cada edición.
- Un municipio-año ausente no se convierte en cero.
- Se distinguen `observed_zero`, `certified_zero` y `missing_report`.
- Claves geográficas se almacenan como texto y conservan ceros a la izquierda.
- No se inventan ni imputan coordenadas sin un indicador explícito.
- La condición de urgencias se toma de las columnas `URGENCIAS GENERALES`, `URGENCIAS PEDIATRÍA` y `URGENCIAS GINECO-OBSTETRICIA` de la cartera SEDESA. CLUES aporta identidad, dirección y coordenadas; no se infiere urgencias a partir del nombre.

## ATUS georreferenciado y red de urgencias

Los seis productos georreferenciados contienen coordenadas, calle, fecha, hora, clase y víctimas por evento. El subconjunto de CDMX tiene 38,185 registros; 8,820 de 8,831 siniestros con víctimas tienen coordenadas válidas dentro de los polígonos de CDMX (99.88%). Los derivados se generan con `scripts/prepare_cdmx_geospatial.py`.

La cartera SEDESA contiene 34 hospitales con al menos un tipo de urgencias marcado. Se vincularon uno a uno con CLUES agosto de 2026. La cobertura calculada es distancia euclidiana desde siniestros observados y no equivale a tiempo de viaje o respuesta.

Los mismos 34 hospitales coincidieron exactamente por CLUES con SINERHIAS: se incorporaron camas de urgencias, camas censables y no censables, médicos, enfermería, ambulancias, quirófanos y tomógrafos. Los indicadores conservan denominadores separados; no suman recursos heterogéneos. La ubicación de unidades nuevas usa puntos de demanda histórica ponderados por el pronóstico futuro y se documenta en `docs/metodologia_actualizada.md`.

Los archivos abiertos DGIS/SINBA 2018-2025 se procesan directamente desde sus ZIP. Solo se lee la tabla principal de urgencias y se filtran las CLUES actuales o anteriores de los 34 hospitales. El producto versionado es un agregado mensual sin registros individuales. La duración inválida o no especificada (`999:99`) no se imputa. El flujo incluye todas las causas atendidas en urgencias; el motivo combinado de accidentes, envenenamientos y violencia no se interpreta como accidente vial.

### SHA-256 de las nuevas fuentes

| Archivo | SHA-256 |
|---|---|
| `atus_2019_shp.zip` | `759acc8b03620b2caccd485aa8b80ceda8e9ede91b9500d123468809b71e480d` |
| `atus_2020_shp.zip` | `aede5e72cba1ba10a19f061a24a2d3d32ce1954a70a38c049bea16267581d64f` |
| `atus_2021_shp.zip` | `42e5303fb57600ec0e6be767ad10f785019e1fd8f5d91fe52e8c12cab6399b34` |
| `atus_2022_shp.zip` | `d6a87afd8dcd27b75fad4bc63d6abc87b95667d91df276fbab85a41ca069b1da` |
| `atus_2023_shp.zip` | `2c83a398389763f14135f9ef4c4e4ede97c1de5cf63995f05e0a36f1284ad4d1` |
| `atus_2024_shp.zip` | `830bc6570b18197ba0016e76fd85ee545f7c8a3eddedc02049b995d00f3bdb32` |
| `ESTABLECIMIENTO_SALUD_202412.xlsx` | `9982c09508cf559335742660dd67792e8715945f2415c438a3f15df31c3e6064` |
| `ESTABLECIMIENTO_SALUD_202608.xlsx` | `5a1cfbe7062b41d9e51249aa4b5fc8a1e623e935882e97fb5f0aac797165c95b` |
| `cartera_servicios_hospitales_segundo_nivel_sedesa_2025.xlsx` | `727a86c7761a9310f7b96722525d670aa24df521a06d6a6c4a78fd4f46502e35` |

## Edición ATUS anual utilizada

El metadato local identifica la publicación como `MEX-INEGI.40.302.03-ATUS-1997-2025`, modificada el 28 de agosto de 2026. Indica que desde julio de 2026 se eliminaron los registros con certificados cero para 2018-2024. Por ello esta edición no permite distinguir certificados cero en el panel reconstruido. También informa una corrección de víctimas 2024 y una recodificación municipal 2023. `CVEGEO`, `CVE_ENT` y `CVE_MUN` se conservan como texto.

La edición anual no contiene un identificador único de siniestro. El pipeline asigna un identificador técnico `año-fila` sin usarlo para deduplicar. Filas indistinguibles en todas las variables conservadas se reportan como control de calidad, pero no se eliminan porque podrían representar siniestros distintos con las mismas características.

Los totales de víctimas se construyen con campos documentados:

- Fallecidas: `CONDMUERTO + PASAMUERTO + PEATMUERTO + CICLMUERTO + OTROMUERTO + NEMUERTO`.
- Heridas: `CONDHERIDO + PASAHERIDO + PEATHERIDO + CICLHERIDO + OTROHERIDO + NEHERIDO`.

### SHA-256 de archivos analizados

| Año | SHA-256 |
|---:|---|
| 2018 | `071a1b28a7c831b29420a20b37da1406c72ad06ab04c3daf3b41bec4729a6ac8` |
| 2019 | `514d4c7f30958718c4ef5b79036d98ee39b89159d52364f47040fb3611be88cb` |
| 2020 | `4cd4565831108003591ca591e71d2c460603c64de368e25285f27436dda0f225` |
| 2021 | `fbc197c16b975f058156122f2a1b5ce57685a655dc47171953b2ade6801a0712` |
| 2022 | `bffa7d5e817e97d3cd0150a806463f7ae3647fdb4327f76436361f7b16e0d7ba` |
| 2023 | `121d7743238ce725eb6898fe548618d4d60e347dc1438b941060b4201b33cc22` |
| 2024 | `00bd5292cff42b13edd1e9c828e898890073e8d729eeee405de418d70263d29c` |
| 2025 | `6bc21394ea023cb75752d3741f8c828ec99cbaacde410d5b9cc0638140524fed` |

## Archivos esperados

Consulte `config/project.yml` para el mapeo canónico. Los scripts fallan con instrucciones concretas mientras los nombres de columnas no estén confirmados. Registre cambios entre ediciones en `data/schema_changes.csv`.
