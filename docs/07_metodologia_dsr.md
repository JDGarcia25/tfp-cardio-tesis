# 07 - Metodologia: Design Science Research (DSR)

## Por que DSR y no un enfoque experimental clasico

El documento original del proyecto declaraba paradigma "positivo cuantitativo" y tipo "experimental, descriptivo, explicativo". Esto tiene problemas fundamentales:

| Problema | Explicacion |
|----------|-------------|
| **No es experimental** | No hay manipulacion de sujetos, no hay grupo control, no hay aleatorizacion. Es un estudio computacional con datos secundarios. |
| **Fuerza terminologia social** | "Poblacion" y "muestra" intentan adaptar muestreo estadistico a una base de datos predefinida. |
| **No refleja lo que se hace** | El proyecto **disena un artefacto** (pipeline de deteccion), lo **construye** y lo **evalua**. Eso es Design Science. |

## Que es Design Science Research

DSR es una metodologia que se centra en la **creacion y evaluacion de artefactos** para resolver problemas identificados. Produce conocimiento a traves de **construir y evaluar**, estableciendo un puente entre relevancia practica y rigor cientifico.

**Referencias clave:**
- Peffers, Tuunanen, Rothenberger y Chatterjee (2007) — DSRM: 6 fases
- Hevner, March, Park y Ram (2004) — 7 directrices de calidad
- March y Smith (1995) — Taxonomia de artefactos

### El artefacto de este proyecto

El artefacto es el **pipeline de deteccion de anomalias ECG**: un sistema de software funcional que:
1. Procesa senales ECG
2. Implementa 4 algoritmos de deteccion
3. Permite evaluacion comparativa cuantitativa

## Las 6 Fases del DSRM aplicadas

### Fase 1: Identificacion del Problema

**Que se hace:** Definir el problema y justificar por que vale la pena resolverlo.

**En este proyecto:**
- Las ECV son la primera causa de muerte global
- Modelos supervisados requieren datos etiquetados costosos
- No existe comparacion sistematica de clustering clasico vs autoencoder para ECG bajo mismas condiciones

**Entregable:** Documento de planteamiento (Capitulo I del trabajo de grado)

**Codigo relacionado:** No aplica directamente, pero la motivacion esta reflejada en toda la documentacion.

### Fase 2: Objetivos de la Solucion

**Que se hace:** Definir que debe lograr el artefacto.

**En este proyecto:**
- Procesar senales ECG de MIT-BIH con preprocesamiento estandar
- Implementar 4 algoritmos de deteccion no supervisada
- Permitir evaluacion comparativa con metricas intrinsecas, extrinsecas y de eficiencia
- Determinar viabilidad tecnica (F1 > 0.75, Sensitivity > 0.80)

**Entregable:** Especificacion de requisitos y criterios de exito

**Codigo relacionado:** `config/default.yaml` (define modelos, metricas, parametros)

### Fase 3: Diseno y Desarrollo

**Que se hace:** Construir el artefacto.

**En este proyecto, 5 sub-etapas:**

| Sub-etapa | Descripcion | Codigo |
|-----------|-------------|--------|
| 3.1 Adquisicion de datos | Carga MIT-BIH, agrupacion AAMI | `src/ecg_anomaly/data/` |
| 3.2 Preprocesamiento | Filtrado, segmentacion, normalizacion | `src/ecg_anomaly/preprocessing/` |
| 3.3 Extraccion de features | PCA (Path A) y features manuales (Path B) | `src/ecg_anomaly/features/` |
| 3.4 Implementacion de algoritmos | KMeans, DBSCAN, HDBSCAN, Autoencoder | `src/ecg_anomaly/models/` |
| 3.5 Asignacion de anomalias | Regla por modelo (mayoria, ruido, umbral) | Dentro de cada modelo |

**Entregable:** Pipeline funcional en Python

### Fase 4: Demostracion

**Que se hace:** Ejecutar el artefacto y mostrar que funciona.

**En este proyecto:**
- Correr el pipeline sobre MIT-BIH (44 registros, ~100,000 latidos)
- Generar clusters y asignar anomalias
- Visualizar con PCA scatter, confusion matrices, graficos comparativos

**Entregable:** Resultados experimentales con visualizaciones

**Codigo relacionado:** `notebooks/04_clustering.ipynb`, `src/ecg_anomaly/visualization/`

### Fase 5: Evaluacion

**Que se hace:** Medir que tan bien el artefacto resuelve el problema.

**En este proyecto, 3 niveles:**

| Nivel | Metricas | Proposito |
|-------|----------|-----------|
| Intrinsecas | Silhouette, DB, CH | ¿Los clusters son coherentes? |
| Extrinsecas | F1, Sensitivity, Specificity, AUC | ¿Coinciden con las anotaciones clinicas? |
| Eficiencia | Tiempo, memoria | ¿Es practico? |

**Entregable:** Tablas comparativas, analisis, determinacion de viabilidad

**Codigo relacionado:** `src/ecg_anomaly/evaluation/`, `notebooks/05_evaluation.ipynb`

**Criterio de viabilidad:** Si al menos uno de los 4 metodos alcanza F1 > 0.75 y Sensitivity > 0.80, se considera tecnicamente viable.

### Fase 6: Comunicacion

**Que se hace:** Comunicar resultados, limitaciones y contribuciones.

**En este proyecto:**
- Documento final del trabajo de grado
- Explicitar: viabilidad tecnica SI, implementacion clinica NO
- Lineas futuras: datos locales, implementacion en dispositivos, validacion clinica

**Entregable:** Trabajo de grado, potencial articulo

## Directrices de Hevner

Hevner et al. (2004) proponen 7 directrices para asegurar calidad en DSR:

| # | Directriz | Requisito | Cumplimiento |
|---|-----------|-----------|-------------|
| 1 | **Artefacto** | Producir artefacto viable | Pipeline funcional de deteccion de anomalias |
| 2 | **Relevancia** | Resolver problema importante | ECV primera causa de muerte; limitaciones de modelos supervisados |
| 3 | **Evaluacion** | Demostrar calidad rigurosa | 3 niveles de metricas usando MIT-BIH |
| 4 | **Contribucion** | Aportar al conocimiento | Comparacion sistematica de 4 metodos para ECG |
| 5 | **Rigor** | Usar metodos rigurosos | Algoritmos validados (sklearn, TF), datos estandar (MIT-BIH) |
| 6 | **Busqueda** | Explorar espacio de soluciones | 4 modelos x 2 representaciones = 8 configuraciones |
| 7 | **Comunicacion** | Audiencia tecnica y de dominio | Documento de ingenieria con proyeccion a salud |

## Implicaciones de DSR en el documento de grado

| Seccion del documento | Cambio al adoptar DSR |
|----------------------|----------------------|
| Planteamiento | Enfatizar que es un **problema de diseno**: no existe artefacto comparativo de estos metodos |
| Objetivos | Verbos DSR: **disenar, desarrollar, evaluar, comparar**. No "gestionar" |
| Justificacion | El aporte es un **artefacto funcional evaluado**, no solo teoria |
| Marco teorico | Agregar seccion DSR citando Peffers (2007), Hevner (2004) |
| Hipotesis | DSR usa "proposiciones de diseno". Si la universidad exige hipotesis, contextualizarlas como proposiciones evaluables |
| Cronograma | Organizar actividades segun las **6 fases DSRM** |

## Mapeo: Fases DSR ↔ Codigo

```
Fase 1 (Problema)     →  docs/00_vision_general.md, docs/01_datos_mitbih.md
Fase 2 (Objetivos)    →  config/default.yaml, README.md
Fase 3 (Desarrollo)   →  src/ecg_anomaly/ (todo el codigo)
Fase 4 (Demostracion) →  notebooks/01-04, src/ecg_anomaly/visualization/
Fase 5 (Evaluacion)   →  src/ecg_anomaly/evaluation/, notebooks/05
Fase 6 (Comunicacion) →  docs/, README.md, trabajo de grado
```

---

**Anterior:** [06 - Pipeline y Ejecucion](06_pipeline_y_ejecucion.md) | **Siguiente:** [08 - Arquitectura del Codigo](08_arquitectura.md)
