# 05 - Evaluacion: Tres Niveles de Metricas

## Por que tres niveles

La evaluacion tiene tres niveles porque cada uno responde una pregunta diferente:

| Nivel | Pregunta | Usa etiquetas | Metricas |
|-------|----------|---------------|----------|
| **Intrinsecas** | ¿Los clusters son coherentes internamente? | No | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| **Extrinsecas** | ¿Las anomalias detectadas coinciden con las reales? | Si (ground truth AAMI) | Accuracy, Sensitivity, Specificity, F1, AUC-ROC |
| **Eficiencia** | ¿Cual es mas practico? | No | Tiempo, memoria |

Las **intrinsecas van primero** porque son las propias del enfoque no supervisado (no necesitan etiquetas). Las extrinsecas validan contra la realidad clinica. La eficiencia completa el analisis costo-beneficio.

---

## Nivel 1: Metricas Intrinsecas (sin etiquetas)

Estas metricas miden la **calidad de los clusters** sin saber si los grupos corresponden a normal/anomalo.

### Silhouette Score (-1 a 1)

**Que mide:** Para cada punto, compara que tan cerca esta de su propio cluster vs que tan lejos esta del cluster mas cercano.

```
silhouette(punto) = (b - a) / max(a, b)

donde:
  a = distancia promedio a los otros puntos de SU cluster
  b = distancia promedio a los puntos del cluster MAS CERCANO
```

**Interpretacion:**
- **+1.0:** Clusters perfectamente separados
- **0.0:** Puntos en el borde entre clusters
- **-1.0:** Puntos asignados al cluster equivocado

**Bueno si:** > 0.5. Aceptable: > 0.25.

### Davies-Bouldin Index (0 a infinito, MENOR es mejor)

**Que mide:** El promedio de "similitud" entre cada cluster y su cluster mas parecido. Clusters compactos y bien separados dan valores bajos.

```
DB = (1/K) * Σ max_j≠i [ (s_i + s_j) / d_ij ]

donde:
  s_i = dispersion promedio del cluster i
  d_ij = distancia entre centros de cluster i y j
```

**Interpretacion:**
- **0:** Clusters perfectamente separados y compactos
- **> 1:** Los clusters se solapan significativamente

### Calinski-Harabasz Index (0 a infinito, MAYOR es mejor)

**Que mide:** El ratio entre la dispersion **entre clusters** y la dispersion **dentro de clusters**. Alto = clusters densos y bien separados.

```
CH = [B / (K-1)] / [W / (N-K)]

donde:
  B = varianza entre clusters (que tan lejos estan los centros entre si)
  W = varianza dentro de clusters (que tan dispersos son internamente)
```

**Interpretacion:** No tiene escala absoluta. Se usa para **comparar entre modelos**: el que tenga mayor CH tiene mejores clusters.

### Implementacion

```python
# src/ecg_anomaly/evaluation/intrinsic.py
def evaluate_intrinsic(X, labels):
    # Filtrar puntos de ruido (label=-1) que no pertenecen a ningun cluster
    mask = labels >= 0
    if len(set(labels[mask])) < 2:
        return {"silhouette": -1, "davies_bouldin": inf, "calinski_harabasz": 0}

    return {
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
    }
```

**Nota importante:** El autoencoder NO genera clusters, por lo que las metricas intrinsecas **no aplican** para el. Solo se calculan para KMeans, DBSCAN y HDBSCAN.

---

## Nivel 2: Metricas Extrinsecas (con ground truth AAMI)

Aqui comparamos las **anomalias detectadas** por cada modelo contra las **etiquetas reales** de los cardiologos (AAMI). Usamos la terminologia de deteccion binaria:

### Matriz de Confusion

```
                    Predicho Normal    Predicho Anomalia
                  ┌─────────────────┬─────────────────────┐
Real Normal       │  TN (Verdadero  │  FP (Falso          │
                  │     Negativo)   │     Positivo)        │
                  ├─────────────────┼─────────────────────┤
Real Anomalia     │  FN (Falso      │  TP (Verdadero      │
                  │     Negativo)   │     Positivo)        │
                  └─────────────────┴─────────────────────┘
```

- **TP (True Positive):** Anomalia real detectada correctamente
- **TN (True Negative):** Normal real clasificado correctamente
- **FP (False Positive):** Normal real clasificado como anomalia (falsa alarma)
- **FN (False Negative):** Anomalia real NO detectada (lo mas peligroso en clinica)

### Accuracy (Exactitud)

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Proporcion total de aciertos. **Cuidado:** Con datos desbalanceados (75% normal), un modelo que diga "todo es normal" tiene accuracy del 75% sin detectar ninguna anomalia. Por eso NO es suficiente sola.

### Sensitivity (Sensibilidad / Recall para anomalias)

```
Sensitivity = TP / (TP + FN)
```

**¿De todas las anomalias reales, cuantas detectamos?** Esta es la metrica mas critica en contexto medico: un FN (arritmia no detectada) puede ser mortal.

**Umbral del proyecto:** Sensitivity > 0.80 para considerar viabilidad tecnica.

### Specificity (Especificidad)

```
Specificity = TN / (TN + FP)
```

**¿De todos los normales reales, cuantos clasificamos bien?** Alta especificidad = pocas falsas alarmas. Importante para no sobrecargar al medico con alertas falsas.

### Precision

```
Precision = TP / (TP + FP)
```

**¿De todas las alertas que generamos, cuantas son anomalias reales?** Baja precision = muchas falsas alarmas.

### F1-Score

```
F1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
```

**Media armonica de precision y sensibilidad.** Es la metrica mas equilibrada para clases desbalanceadas. F1 > 0.75 es el umbral de viabilidad tecnica del proyecto.

### AUC-ROC (Area Under ROC Curve)

```
AUC = Area bajo la curva que grafica Sensitivity vs (1 - Specificity)
```

**Capacidad general de discriminacion.** 0.5 = aleatorio, 1.0 = perfecto.

### Implementacion

```python
# src/ecg_anomaly/evaluation/extrinsic.py
def evaluate_extrinsic(true_labels, pred_labels):
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    return {
        "accuracy": (tp + tn) / (tn + fp + fn + tp),
        "sensitivity": tp / (tp + fn),          # Recall anomalia
        "specificity": tn / (tn + fp),           # Recall normal
        "precision": tp / (tp + fp),
        "f1": 2 * prec * sens / (prec + sens),
        "auc_roc": roc_auc_score(true_labels, pred_labels),
    }
```

---

## Nivel 3: Metricas de Eficiencia

### Tiempo de ejecucion

Mide cuantos **segundos** tarda el modelo en entrenar sobre los datos. Critico para contextos de recursos limitados donde el procesamiento debe ser rapido.

### Memoria pico

Mide cuantos **megabytes** de RAM consume el modelo durante el entrenamiento. Importante si se quiere ejecutar en hardware limitado.

### Implementacion

```python
# src/ecg_anomaly/evaluation/efficiency.py
class EfficiencyTracker:
    def __enter__(self):
        tracemalloc.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_seconds = time.perf_counter() - self._start
        _, peak = tracemalloc.get_traced_memory()
        self.peak_memory_mb = peak / (1024 * 1024)
        tracemalloc.stop()
```

Uso:
```python
with EfficiencyTracker() as tracker:
    detector.fit(X)
print(f"Tiempo: {tracker.elapsed_seconds:.2f}s")
print(f"Memoria: {tracker.peak_memory_mb:.1f} MB")
```

---

## Criterio de viabilidad tecnica

Basado en la literatura, el proyecto establece:

> Si al menos uno de los cuatro metodos alcanza **F1-score > 0.75** y **Sensitivity > 0.80**, se considera **tecnicamente viable** para deteccion de anomalias en ECG.

Esto NO significa que sea clinicamente apto. Significa que el enfoque no supervisado tiene potencial suficiente para justificar investigacion futura.

## Entregable final

Una tabla comparativa con:

```
┌────────────┬───────────┬──────────────┬──────────────┬──────────┬─────────┬──────────┬──────────┐
│ Modelo     │ Silhouette│ Davies-Bouldin│ Calinski-Har.│ F1-Score │ Sensit. │ Tiempo(s)│Memoria(MB)│
├────────────┼───────────┼──────────────┼──────────────┼──────────┼─────────┼──────────┼──────────┤
│ K-Means    │  0.xx     │  x.xx        │  xxx.x       │  0.xx    │  0.xx   │  x.xx    │  xx.x    │
│ DBSCAN     │  0.xx     │  x.xx        │  xxx.x       │  0.xx    │  0.xx   │  x.xx    │  xx.x    │
│ HDBSCAN    │  0.xx     │  x.xx        │  xxx.x       │  0.xx    │  0.xx   │  x.xx    │  xx.x    │
│ Autoencoder│  N/A      │  N/A         │  N/A         │  0.xx    │  0.xx   │  x.xx    │  xx.x    │
└────────────┴───────────┴──────────────┴──────────────┴──────────┴─────────┴──────────┴──────────┘
```

## Archivos del codigo relevantes

| Archivo | Que contiene |
|---------|-------------|
| `src/ecg_anomaly/evaluation/intrinsic.py` | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| `src/ecg_anomaly/evaluation/extrinsic.py` | Accuracy, Sensitivity, Specificity, F1, AUC-ROC |
| `src/ecg_anomaly/evaluation/efficiency.py` | EfficiencyTracker (tiempo + memoria) |
| `src/ecg_anomaly/evaluation/comparator.py` | ModelComparator: orquesta los 3 niveles |

---

**Anterior:** [04 - Modelos](04_modelos.md) | **Siguiente:** [06 - Pipeline y Ejecucion](06_pipeline_y_ejecucion.md)
