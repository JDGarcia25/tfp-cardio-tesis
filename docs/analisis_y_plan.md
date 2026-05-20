# Análisis de Métricas y Plan de Mejora — Tesis Arritmia Cardíaca

## Contexto del Proyecto

- **Proyecto:** Evaluación comparativa de métodos de clustering no supervisado y autoencoder para detección de anomalías en señales electrocardiográficas
- **Dataset:** 100,705 latidos, 44 registros MIT-BIH, PCA 200→12 componentes (95.3% varianza)
- **Distribución real:** 89.5% normal, 10.5% anomalía (10,606 anómalos)
- **Hipótesis:** Las técnicas de procesamiento de señales ECG combinadas con algoritmos de clustering no supervisados permiten detectar y clasificar de manera efectiva patrones anómalos que determinan las arritmias cardíacas.

---

## Tabla de Resultados Actuales (del notebook 05_evaluation.ipynb)

| Modelo | TP | FP | TN | FN | F1 | Sensitivity | Specificity | Precisión | Tiempo (s) | Memoria (MB) |
|--------|----|----|----|----|-----|-------------|-------------|-----------|------------|--------------|
| kmeans | 5,296 | 43,973 | 46,126 | 5,310 | **0.1769** | 0.4993 | 0.5119 | 0.1075 | 2.39 | 18.51 |
| dbscan | 2,117 | 5,050 | 85,049 | 8,489 | **0.2382** | 0.1996 | 0.9440 | 0.2954 | 64.58 | 38.35 |
| hdbscan | 3,758 | 17,627 | 72,472 | 6,848 | **0.2349** | 0.3543 | 0.8044 | 0.1757 | 150.41 | 32.39 |
| autoencoder | 914 | 4,122 | 85,977 | 9,692 | **0.1169** | 0.0862 | 0.9543 | 0.1815 | 418.51 | 393.49 |

**Umbral de viabilidad del proyecto:** F1 > 0.75 y Sensitivity > 0.80. Ninguno lo alcanza.

---

## Diagnóstico por Modelo

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

## Causas Raíz Transversales

1. **Reglas de anomalía arbitrarias:** Los 4 modelos usan reglas fijas que no se alinean con la distribución real de clases (10.5% anomalía).
2. **PCA pierde información discriminante:** 12 componentes PCA retienen varianza global pero no separabilidad clínica.
3. **Sin features de intervalo RR:** El RR es el marcador más potente para arritmias y no se usa.
4. **Modelo global (44 registros juntos):** Las arritmias son paciente-específicas. Un modelo global promedia patrones distintos.
5. **Etiquetas AAMI son semánticas, no morfológicas:** N y L son morfológicamente distintos pero ambos "normales" para AAMI. Ningún clustering basado solo en forma de onda va a alinearse perfectamente.

---

## Plan de Mejora (5 Cambios)

### Cambio 1: K-Means — Distance-scoring + k=10 (código en `kmeans.py`)

**Problema:** Con k=2, la regla "cluster entero = anomalía" es incorrecta.
**Solución:** Usar `n_clusters=10`, calcular distancia de cada punto a su centroide, marcar el ~10.5% más lejano como anomalía.

**Cambios en config/default.yaml:**
```yaml
kmeans_params:
  n_clusters: 10
  random_state: 42
  n_init: 10
```

**Cambios en `src/ecg_anomaly/models/kmeans.py`:**
- Reemplazar `_assign_anomalies()`: ya no usar `_majority_cluster`, sino calcular distancias a centroides y usar percentil.
- Implementar método `score_anomalies(X)` que retorne distancia al centroide más cercano.

**Impacto estimado:** F1 0.18 → ~0.38-0.42

---

### Cambio 2: DBSCAN — Epsilon fijo más agresivo (config)

**Problema:** `eps: auto` da 3.9, muy grande. Detecta pocas anomalías.
**Solución:** Usar `eps: 2.5` y `min_samples: 5`.

**Cambios en config/default.yaml:**
```yaml
dbscan_params:
  eps: 2.5
  min_samples: 5
```

**Impacto estimado:** F1 0.24 → ~0.30-0.33

---

### Cambio 3: Autoencoder — Umbral dinámico (código en `autoencoder.py`)

**Problema:** Percentil 95 fijo fuerza 5% de anomalías.
**Solución:** En lugar de percentil fijo, calcular el percentil necesario para marcar ~10.5% como anomalía, o exponer el score de reconstrucción para calibración externa.

**Cambios en `src/ecg_anomaly/models/autoencoder.py`:**
- En `fit()`, calcular umbral como `np.percentile(reconstruction_errors_, 100 - anomaly_rate * 100)` donde `anomaly_rate = 0.105`.
- O permitir `anomaly_percentile: auto` en params.

**Impacto estimado:** F1 0.12 → ~0.22-0.25

---

### Cambio 4: Features RR-interval (código en `manual.py`)

**Problema:** Las features actuales no incluyen intervalo RR.
**Solución:** Añadir 4 features: RR previo, RR posterior, ratio RR, desviación del RR promedio del registro.

**Cambios en `src/ecg_anomaly/features/manual.py`:**
```python
# Features a añadir
rr_pre = current_r_peak - previous_r_peak
rr_post = next_r_peak - current_r_peak
rr_ratio = rr_pre / rr_post
rr_dev = abs(rr_pre - mean_rr) / mean_rr
```

**Cambios en config/default.yaml:**
```yaml
representation: manual_features
```

**Impacto estimado:** F1 +0.08-0.12 sobre cualquier modelo

---

### Cambio 5: Evaluación por registro (código en `comparator.py`)

**Problema:** Modelo global promedia 44 pacientes distintos.
**Solución:** Entrenar y evaluar un modelo por cada registro, promediar métricas al final.

**Cambios en `src/ecg_anomaly/evaluation/comparator.py`:**
- Nuevo método `run_all_per_record()` que itera por registro.
- Cada registro: fit, predict, evaluate, guardar métricas.
- Al final: promedio de F1, Sensitivity, etc. entre todos los registros.
- Reportar también desviación estándar.

**Impacto estimado:** F1 +0.05-0.10

---

## Proyección Final

| Escenario | Mejor F1 | Sensitivity | ¿Hipótesis cumplida? |
|-----------|---------|-------------|---------------------|
| Actual | 0.238 | 0.199 | ❌ No |
| + Distance-scoring K-Means (k=10) | ~0.42 | ~0.55 | ❌ Parcial |
| + RR-interval features | ~0.52 | ~0.65 | ❌ Parcial |
| + Per-record + manual features + DBSCAN tuning | ~0.60 | ~0.70 | ⚠️ Aceptable con limitaciones |
| + Todo combinado | ~0.65 | ~0.72 | ⚠️ Moderado |

**Conclusión:** Incluso con todos los cambios, el F1 máximo realista es ~0.65. No se alcanza el umbral de viabilidad clínica (F1 > 0.75), pero se pasa de "no funciona" (F1=0.24) a **"tiene utilidad parcial como herramienta de apoyo"** (F1=0.65). Esto permite defender la hipótesis con matices.

---

## Archivos a Modificar

| Archivo | Cambio | Líneas |
|---------|--------|--------|
| `config/default.yaml` | kmeans.n_clusters=10, dbscan.eps=2.5, representation=manual_features | ~5 |
| `src/ecg_anomaly/models/kmeans.py` | Distance-scoring + score_anomalies() | ~25 |
| `src/ecg_anomaly/models/base.py` | Añadir método score_anomalies() abstracto | ~5 |
| `src/ecg_anomaly/models/dbscan.py` | Implementar score_anomalies() | ~15 |
| `src/ecg_anomaly/models/hdbscan_model.py` | Implementar score_anomalies() | ~15 |
| `src/ecg_anomaly/models/autoencoder.py` | Umbral dinámico + score_anomalies() | ~10 |
| `src/ecg_anomaly/features/manual.py` | Añadir 4 features RR-interval | ~30 |
| `src/ecg_anomaly/evaluation/comparator.py` | Evaluación por registro | ~50 |
| `src/ecg_anomaly/config.py` | Soporte para nuevos parámetros si es necesario | ~5 |

---

## Próximos Pasos (mañana)

1. Implementar Changes 1-5 en orden de impacto
2. Re-ejecutar el pipeline completo
3. Comparar antes/después en tabla
4. Ajustar hipótesis/conclusiones de la tesis
5. Generar nuevos gráficos para el documento

---

*Documento generado el 2026-05-20 para continuar al día siguiente.*
