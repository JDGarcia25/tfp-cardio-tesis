# Análisis de Métricas y Plan de Mejora — Tesis Arritmia Cardíaca

> **Actualizado 2026-07-11:** los 5 cambios del plan original ya están implementados y el
> notebook 05 fue re-ejecutado con los resultados nuevos. La tabla "Resultados Actuales"
> de más abajo corresponde al estado ANTES de los cambios (línea base histórica, 2026-05-20);
> los resultados DESPUÉS están en la sección "Resultados Tras el Plan de Mejora".

## Contexto del Proyecto

- **Proyecto:** Evaluación comparativa de métodos de clustering no supervisado y autoencoder para detección de anomalías en señales electrocardiográficas
- **Dataset:** 100,705 latidos, 44 registros MIT-BIH, PCA 200→12 componentes (95.3% varianza)
- **Distribución real:** 89.5% normal, 10.5% anomalía (10,606 anómalos)
- **Hipótesis:** Las técnicas de procesamiento de señales ECG combinadas con algoritmos de clustering no supervisados permiten detectar y clasificar de manera efectiva patrones anómalos que determinan las arritmias cardíacas.

---

## Tabla de Resultados Actuales — línea base (2026-05-20, antes del plan)

| Modelo | TP | FP | TN | FN | F1 | Sensitivity | Specificity | Precisión | Tiempo (s) | Memoria (MB) |
|--------|----|----|----|----|-----|-------------|-------------|-----------|------------|--------------|
| kmeans | 5,296 | 43,973 | 46,126 | 5,310 | **0.1769** | 0.4993 | 0.5119 | 0.1075 | 2.39 | 18.51 |
| dbscan | 2,117 | 5,050 | 85,049 | 8,489 | **0.2382** | 0.1996 | 0.9440 | 0.2954 | 64.58 | 38.35 |
| hdbscan | 3,758 | 17,627 | 72,472 | 6,848 | **0.2349** | 0.3543 | 0.8044 | 0.1757 | 150.41 | 32.39 |
| autoencoder | 914 | 4,122 | 85,977 | 9,692 | **0.1169** | 0.0862 | 0.9543 | 0.1815 | 418.51 | 393.49 |

**Umbral de viabilidad del proyecto:** F1 > 0.75 y Sensitivity > 0.80. Ninguno lo alcanzaba en esta línea base.

---

## Diagnóstico por Modelo (línea base, ya resuelto — ver sección siguiente)

### K-Means (F1=0.1769)
- **Problema raíz:** Regla "cluster mayoritario = normal" con k=2 fuerza una partición ~50/50. La tasa real es 90/10. Por eso genera 43,973 falsos positivos.
- **AUC-ROC = 0.5056** → es aleatorio puro. No discrimina en absoluto.

### DBSCAN (F1=0.2382) — Mejor F1 actual
- **Problema raíz:** Solo detecta 7,167 anomalías (7.1%) cuando la real es 10.5%. `eps=auto` (percentil 90) es demasiado conservador.
- **Fortaleza:** Precisión más alta (29.5%). Cuando dice "anomalía", acierta 1 de cada 3 veces. Especificidad de 94.4%.

### HDBSCAN (F1=0.2349)
- **Problema raíz:** Predice 21,385 anomalías (21.2%), el doble de la real. 17,627 falsos positivos.
- **Fortaleza:** Mejor estructura intrínseca (Silhouette = 0.4869). Los clusters tienen sentido morfológico pero no clínico.

### Autoencoder (F1=0.1169) — El peor
- **Problema raíz crítico:** Umbral fijo en percentil 95 fuerza exactamente 5% de anomalías. La tasa real es 10.5%. Sensitivity de solo 8.6%.
- **Agravante:** 418 segundos y 393 MB para ser peor que K-Means.

---

## Resultados Tras el Plan de Mejora (notebook 05, re-ejecutado 2026-07-09)

| Modelo | Accuracy | Sensitivity | Specificity | F1 | AUC-ROC | Silhouette | Tiempo (s) | Memoria (MB) | Anomalías detectadas |
|--------|----------|-------------|-------------|-----|---------|------------|------------|---------------|----------------------|
| kmeans | 0.8727 | 0.3943 | 0.9291 | **0.3949** | 0.6617 | 0.1549 | 4.91 | 33.87 | 10,574 |
| dbscan | 0.8712 | 0.1559 | 0.9554 | **0.2032** | 0.5557 | -0.0501 | 28.05 | 54.73 | 5,667 |
| hdbscan | 0.5872 | 0.6959 | 0.5744 | **0.2620** | 0.6351 | 0.2809 | 287.50 | 33.02 | 45,730 |
| autoencoder | 0.8916 | **0.8576** | 0.8956 | **0.6251** | 0.8766 | n/a | 113.66 | 282.84 | 18,498 |

**Umbral de viabilidad:** F1 > 0.75 y Sensitivity > 0.80. El autoencoder **supera el umbral de Sensitivity** (0.858 > 0.80) y queda cerca en F1 (0.625 vs 0.75), una mejora de +434% en F1 y +895% en Sensitivity respecto a su línea base.

**Ranking multi-criterio (pesos por defecto, `get_multi_criteria_ranking`):**

| Rank | Modelo | Composite |
|------|--------|-----------|
| 1 | autoencoder | 0.9001 |
| 2 | kmeans | 0.5864 |
| 3 | hdbscan | 0.3636 |
| 4 | dbscan | 0.3349 |

**Ranking por escenario clínico:** el autoencoder gana en los tres escenarios (Screening 0.9386, Diagnóstico 0.9372, Equilibrado 0.9001), desplazando a K-Means como recomendación global — un cambio respecto a la línea base, donde el ranking dependía mucho del escenario.

### Qué cambió en el diagnóstico por modelo

- **K-Means (F1 0.18→0.39, +122%):** el distance-scoring con k=10 reemplazó la regla binaria "cluster mayoritario=normal"; ya discrimina mejor que el azar (AUC-ROC 0.50→0.66).
- **DBSCAN (F1 0.24→0.20, ligera baja):** con auto-optimización de eps por percentil (p90) sigue siendo conservador (solo 5,667 anomalías detectadas vs 10,606 reales); es el único modelo que no mejoró — candidato para ajuste futuro de `eps_percentile` en `config/default.yaml`.
- **HDBSCAN (F1 0.23→0.26):** sigue sobre-prediciendo (45,730 anomalías, 4.3x la tasa real), pero ahora corre en un espacio de features más informativo (features manuales + RR en vez de PCA de señal).
- **Autoencoder (F1 0.12→0.63, el que más mejoró):** el umbral dinámico por `anomaly_rate=0.105` (en vez del percentil 95 fijo) más el fit solo-normal (sin data leakage) lo convirtieron en el mejor modelo por un margen amplio.

---

## Causas Raíz Transversales

1. **Reglas de anomalía arbitrarias:** Los 4 modelos usan reglas fijas que no se alinean con la distribución real de clases (10.5% anomalía).
2. **PCA pierde información discriminante:** 12 componentes PCA retienen varianza global pero no separabilidad clínica.
3. **Sin features de intervalo RR:** El RR es el marcador más potente para arritmias y no se usa.
4. **Modelo global (44 registros juntos):** Las arritmias son paciente-específicas. Un modelo global promedia patrones distintos.
5. **Etiquetas AAMI son semánticas, no morfológicas:** N y L son morfológicamente distintos pero ambos "normales" para AAMI. Ningún clustering basado solo en forma de onda va a alinearse perfectamente.

---

## Plan de Mejora (5 Cambios) — ✅ Todos implementados

### Cambio 1: K-Means — Distance-scoring + k=10 — ✅ Implementado (`kmeans.py`)

**Problema:** Con k=2, la regla "cluster entero = anomalía" es incorrecta.
**Solución aplicada:** `n_clusters=10`, `distance_percentile=89.5` en `config/default.yaml`; `KMeansDetector.score_anomalies()` calcula la distancia euclidiana de cada punto a su centroide más cercano y `fit()` marca como anomalía todo lo que supera el percentil configurado.

**Resultado real:** F1 0.1769 → **0.3949** (estimado ~0.38-0.42, en rango).

---

### Cambio 2: DBSCAN — Epsilon más agresivo — ✅ Implementado, con enfoque distinto (`dbscan.py`)

**Problema:** `eps: auto` daba 3.9, muy grande. Detecta pocas anomalías.
**Solución aplicada:** en vez de un `eps=2.5` fijo, se implementó auto-optimización vía gráfico k-distancias con `eps_percentile` configurable (actualmente 90, ver `config/default.yaml: dbscan_params`). Más robusto que un valor fijo porque se adapta a la escala de cada representación de features.

**Resultado real:** F1 0.2382 → **0.2032** (bajó ligeramente; ver nota abajo — es el único de los 4 modelos que no mejoró con el plan).

---

### Cambio 3: Autoencoder — Umbral dinámico — ✅ Implementado (`autoencoder.py`)

**Problema:** Percentil 95 fijo fuerza 5% de anomalías.
**Solución aplicada:** `fit()` calcula `threshold_percentile = (1 - anomaly_rate) * 100` con `anomaly_rate=0.105` (10.5%, la tasa real de anomalías), y marca como anómalo todo latido con error de reconstrucción por encima de ese percentil.

**Resultado real:** F1 0.1169 → **0.6251** (muy por encima del estimado ~0.22-0.25 — el cambio con mayor impacto real, potenciado también por el fix de data leakage del notebook 04/05).

---

### Cambio 4: Features RR-interval — ✅ Implementado (`features/manual.py`)

**Problema:** Las features no incluían intervalo RR.
**Solución aplicada:** se añadieron `rr_pre`, `rr_post`, `rr_ratio_pre_post`, `rr_dev` a `FEATURE_NAMES` (además de 6 features de ventana temporal de una fase posterior: `rr_mean_5`, `rr_std_5`, `rr_mean_10`, `rr_std_10`, `rmssd_5`, `pnn_5`). Total: 22 features manuales (`N_MANUAL_FEATURES_TOTAL`). `config/default.yaml` usa `representation: manual_features`.

**Resultado real:** confirmado en el log del pipeline — `Features manuales: 100705 latidos x 22 features`.

---

### Cambio 5: Evaluación por registro — ✅ Implementado (`evaluation/comparator.py`)

**Problema:** Modelo global promedia 44 pacientes distintos.
**Solución aplicada:** `ModelComparator.run_all_per_record()` itera por registro, entrena/evalúa cada modelo por paciente y agrega una fila `<modelo>_macro_avg` con la media y `<modelo>_macro_avg` con desviación estándar (`f1_std`, etc.) y `f1_above_05` (fracción de registros con F1>0.5). Se invoca automáticamente desde `pipeline.py` (`ecg-run`) tras guardar los modelos.

**Nota:** esta evaluación no corre dentro del notebook 05 (vive en el pipeline CLI); ver `models/best_model.json` y el log de `ecg-run` para los resultados por registro más recientes.

---

## Proyección Final vs. Resultado Real

| Escenario | F1 proyectado | F1 real | Sensitivity proyectada | Sensitivity real | ¿Hipótesis cumplida? |
|-----------|---------|---------|-------------|---------|---------------------|
| Línea base | 0.238 | 0.238 | 0.199 | 0.199 | ❌ No |
| + Todo combinado (autoencoder, mejor modelo) | ~0.65 | **0.625** | ~0.72 | **0.858** | ⚠️ Moderado — Sensitivity superó lo proyectado y el umbral clínico (0.80); F1 quedó cerca de lo proyectado pero bajo el umbral (0.75) |

**Conclusión:** la proyección de F1~0.65 se cumplió casi exactamente (0.625 real). La Sensitivity real (0.858) superó tanto la proyección (0.72) como el umbral de viabilidad clínica (0.80) — el hallazgo más fuerte para la defensa: el autoencoder con umbral calibrado a la tasa real de anomalías y fit solo-normal (sin data leakage) es un detector de screening viable (prioriza no perder anomalías), aunque su F1 todavía no alcanza el umbral de 0.75 por el volumen de falsos positivos. Esto permite defender la hipótesis con matices: **"tiene utilidad como herramienta de apoyo al screening, no de diagnóstico definitivo"**.

---

## Archivos Modificados (todos aplicados)

| Archivo | Cambio | Estado |
|---------|--------|--------|
| `config/default.yaml` | kmeans.n_clusters=10, dbscan.eps_percentile=90, representation=manual_features, random_seed=42 | ✅ |
| `src/ecg_anomaly/models/kmeans.py` | Distance-scoring + score_anomalies() | ✅ |
| `src/ecg_anomaly/models/base.py` | Método score_anomalies() abstracto | ✅ |
| `src/ecg_anomaly/models/dbscan.py` | score_anomalies() + auto-eps por percentil | ✅ |
| `src/ecg_anomaly/models/hdbscan_model.py` | score_anomalies() | ✅ |
| `src/ecg_anomaly/models/autoencoder.py` | Umbral dinámico + score_anomalies() | ✅ |
| `src/ecg_anomaly/features/manual.py` | 4 features RR-interval + 6 de ventana temporal | ✅ |
| `src/ecg_anomaly/evaluation/comparator.py` | run_all_per_record() | ✅ |
| `src/ecg_anomaly/data/splitting.py` | make_normal_fit_split() (fix de leakage, guía de mejoras #1) | ✅ |
| `src/ecg_anomaly/seeding.py` | set_global_seed() (guía de mejoras #9) | ✅ |
| `src/ecg_anomaly/cache.py` | get_or_build_preprocessed() (guía de mejoras #2) | ✅ |

---

## Evaluación por Registro (Cambio 5, resultado de `ecg-run` 2026-07-11)

`ModelComparator.run_all_per_record()` entrena y evalúa cada modelo dentro de cada uno de los 44 registros por separado (el autoencoder se salta por ser muy lento a este nivel de granularidad — ver log de `ecg-run`). Promedio macro (media simple entre los 44 registros):

| Modelo | F1 macro-avg | Sensitivity | Specificity | Precisión | F1 std | % registros con F1>0.5 |
|--------|---------------|-------------|-------------|-----------|--------|--------------------------|
| kmeans | 0.2272 | 0.3131 | 0.9058 | 0.2681 | 0.2163 | 15.9% |
| dbscan | 0.1667 | 0.2788 | 0.9425 | 0.2539 | 0.1656 | 4.5% |
| hdbscan | 0.1862 | 0.3966 | 0.9563 | 0.2570 | 0.1606 | 6.8% |

**Hallazgo clave — confirma la causa raíz #4:** el F1 global de K-Means (0.395) es casi el doble del F1 promedio por-paciente (0.227). El modelo global se beneficia de patrones que se promedian entre 44 pacientes distintos; a nivel individual, **solo 15.9% de los pacientes alcanzan F1>0.5 con K-Means**, y ese porcentaje cae a 4.5%–6.8% con DBSCAN/HDBSCAN. La desviación estándar alta (F1 std ≈ 0.16–0.22) confirma que el rendimiento es muy heterogéneo entre pacientes — consistente con la hipótesis de que las arritmias son paciente-específicas y un modelo global no captura esa variabilidad. Esto es un matiz importante para la tesis: el rendimiento "global" reportado en las tablas anteriores es optimista respecto al caso de uso real (evaluar un paciente nuevo).

---

## Pendientes Reales (2026-07-11)

1. ~~Implementar Cambios 1-5~~ — hecho.
2. ~~Re-ejecutar el pipeline completo~~ — hecho vía `ecg-run` (ver `models/best_model.json`, regenerado con la config actual: mejor modelo = autoencoder, representation=manual_features).
3. ~~Comparar antes/después en tabla~~ — hecho en este documento.
4. ~~Revisar evaluación por registro~~ — hecho, ver sección anterior. Hallazgo: rendimiento por-paciente muy inferior y heterogéneo respecto al global.
5. **Ajustar hipótesis/conclusiones de la tesis** con dos hallazgos: (a) Sensitivity 0.858 del autoencoder supera el umbral clínico; (b) el rendimiento por-paciente es sustancialmente menor que el global — pendiente de redactar en el documento final de tesis.
6. **Generar nuevos gráficos** con los resultados post-mejora para el documento final (los del notebook 05 ya reflejan estos números, falta exportarlos al informe).
7. **DBSCAN no mejoró** (F1 0.2382→0.2032): considerar bajar `eps_percentile` (más agresivo) o probar `min_samples` distinto para el espacio de 22 features manuales.
8. Considerar correr `run_all_per_record` también para el autoencoder (actualmente se salta por lentitud) si el tiempo de tesis lo permite, ya que es el modelo recomendado y sería valioso saber si también degrada por-paciente.

---

*Documento generado el 2026-05-20; actualizado 2026-07-11 tras implementar y validar los 5 cambios del plan, ejecutar el pipeline completo (`ecg-run`) y revisar la evaluación por registro.*
