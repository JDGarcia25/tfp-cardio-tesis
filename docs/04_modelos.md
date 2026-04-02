# 04 - Modelos de Deteccion de Anomalias

## Logica general

Todos los modelos hacen lo mismo: reciben datos de latidos y clasifican cada uno como **normal** o **anomalia**. La diferencia esta en **como** llegan a esa decision.

Ninguno de los 4 modelos ve las etiquetas AAMI durante el entrenamiento. Son **no supervisados**. Cada uno tiene su propia regla para decidir que es anomalo.

## Nivel 1: K-Means (Baseline)

### Que es
K-Means divide los datos en K grupos (clusters) minimizando la distancia de cada punto al centro de su grupo. Es el algoritmo de clustering mas simple y conocido (MacQueen, 1967).

### Como funciona

```
1. Elegir K centros aleatorios
2. Asignar cada latido al centro mas cercano
3. Recalcular los centros como el promedio de cada grupo
4. Repetir 2-3 hasta que no cambie
```

### Regla de anomalia

> **El cluster con mas latidos = normal. Los demas = anomalias.**

La logica: si la mayoria de los latidos son normales (que es cierto en ECG), el cluster mas grande captura los latidos normales. Los clusters mas pequenos contienen latidos con morfologias diferentes (posibles arritmias).

### Parametros

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `n_clusters` | 2 | Binario: normal vs anomalo |
| `n_init` | 10 | Ejecutar 10 veces con diferentes inicializaciones y quedarse con el mejor |
| `random_state` | 42 | Reproducibilidad |

### Limitaciones
- **Asume clusters esfericos:** Todos los clusters tienen la misma forma circular. Las arritmias reales no siempre forman clusters esfericos.
- **Requiere fijar K:** Hay que decidir cuantos clusters antes de ejecutar.
- **No detecta ruido:** Cada punto debe pertenecer a un cluster; no hay concepto de "outlier".

### Implementacion

```python
# src/ecg_anomaly/models/kmeans.py
class KMeansDetector(BaseAnomalyDetector):
    def fit(self, X):
        self.model = KMeans(**self.params)
        self.labels_ = self.model.fit_predict(X)
        # Cluster mayoritario = normal
        unique, counts = np.unique(self.labels_, return_counts=True)
        majority = unique[np.argmax(counts)]
        self.anomaly_labels_ = np.where(self.labels_ == majority, 0, 1)
```

## Nivel 2: DBSCAN (Densidad)

### Que es
DBSCAN (Density-Based Spatial Clustering of Applications with Noise, Ester et al., 1996) agrupa puntos que estan **densamente conectados** y marca los puntos aislados como **ruido**.

### Como funciona

```
Para cada punto:
  1. Contar cuantos vecinos tiene dentro de radio 'epsilon'
  2. Si tiene >= min_samples vecinos → es un "core point" (punto central)
  3. Los core points conectados forman un cluster
  4. Los puntos que no son core ni estan cerca de uno → RUIDO (-1)
```

### Regla de anomalia

> **Puntos etiquetados como ruido (label = -1) son anomalias.**

La logica: los latidos normales forman regiones densas (muchos latidos parecidos juntos). Los latidos anomalos estan aislados, lejos de las regiones densas, por eso DBSCAN los marca como ruido.

### Parametros

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `eps` | auto | Se calcula automaticamente usando percentil 90 de k-distancias |
| `min_samples` | 10 | Minimo de vecinos para ser core point |

### Auto-optimizacion de epsilon

```python
# El radio epsilon se calcula automaticamente:
# 1. Para cada punto, calcular distancia al vecino min_samples-esimo
# 2. Ordenar esas distancias
# 3. Tomar el percentil 90 como epsilon
neigh = NearestNeighbors(n_neighbors=min_samples)
distances = neigh.fit(X).kneighbors(X)[0]
eps = np.percentile(np.sort(distances[:, -1]), 90)
```

### Ventajas sobre K-Means
- **No asume forma** de los clusters (pueden ser alargados, irregulares)
- **Detecta ruido** nativamente
- **No requiere fijar K:** Encuentra el numero de clusters automaticamente

### Limitaciones
- **Sensible a epsilon:** Un epsilon mal elegido produce resultados pobres
- **Problemas con densidades variables:** Si los clusters tienen densidades muy diferentes, un solo epsilon no funciona bien

## Nivel 3: HDBSCAN (Densidad Jerarquica)

### Que es
HDBSCAN (Hierarchical DBSCAN, Campello et al., 2013) es la **evolucion** de DBSCAN. Resuelve el problema de elegir epsilon construyendo una jerarquia de clusters a multiples escalas de densidad.

### Por que HDBSCAN en vez de OPTICS

El proyecto originalmente usaba OPTICS, pero se reemplazo por HDBSCAN porque:

| Aspecto | OPTICS | HDBSCAN |
|---------|--------|---------|
| Genera clusters directamente | No (genera diagrama de alcanzabilidad que hay que interpretar) | Si |
| Seleccion de parametros | Manual (requiere xi o epsilon) | Automatica |
| Datos alta dimensionalidad | Menos robusto | Mas robusto |
| Implementacion Python | sklearn.cluster.OPTICS | sklearn.cluster.HDBSCAN (nativo desde v1.3) |

### Regla de anomalia

> **Igual que DBSCAN: puntos de ruido (label = -1) son anomalias.**

### Parametros

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `min_cluster_size` | 15 | Tamano minimo para considerar un grupo como cluster |
| `min_samples` | 10 | Conservatividad del algoritmo |

### Ventajas sobre DBSCAN
- **No requiere epsilon:** Solo necesita `min_cluster_size`
- **Maneja densidades variables:** Encuentra clusters de diferentes densidades
- **Mas robusto:** Menos sensible a la eleccion de parametros

### Implementacion

```python
# src/ecg_anomaly/models/hdbscan_model.py
class HDBSCANDetector(BaseAnomalyDetector):
    def fit(self, X):
        from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
        self.model = HDBSCAN(**self.params)
        self.labels_ = self.model.fit_predict(X)
        self.anomaly_labels_ = np.where(self.labels_ == -1, 1, 0)
```

## Nivel 4: Autoencoder (Deep Learning)

### Que es
Un autoencoder es una red neuronal que aprende a **reconstruir su propia entrada**. La red tiene forma de reloj de arena: comprime los datos a una representacion pequena (encoding) y luego los reconstruye.

### Intuicion clave

> Si el autoencoder se entrena con latidos (mayoria normales), **aprende a reconstruir bien los latidos normales**. Cuando le llega un latido anomalo (forma diferente), **la reconstruccion es mala** y el error es alto.

```
Latido normal → Encoder → [32 dim] → Decoder → Reconstruccion buena (error bajo)
Latido anomalo → Encoder → [32 dim] → Decoder → Reconstruccion mala (error ALTO)
```

### Arquitectura

```
Input (200) → Dense(128) → BN → Dropout → Dense(64) → BN → Dropout
           → Dense(32) [encoding]
           → Dense(64) → BN → Dense(128) → BN → Dense(200) [output]
```

| Capa | Neuronas | Funcion |
|------|----------|---------|
| Input | 200 | Latido escalado (sin PCA) |
| Encoder capa 1 | 128 | Compresion inicial + BatchNorm + Dropout(0.2) |
| Encoder capa 2 | 64 | Compresion intermedia + BatchNorm + Dropout(0.2) |
| **Encoding** | **32** | **Representacion comprimida (cuello de botella)** |
| Decoder capa 1 | 64 | Expansion intermedia + BatchNorm |
| Decoder capa 2 | 128 | Expansion final + BatchNorm |
| Output | 200 | Reconstruccion del latido |

### Regla de anomalia

> **Latidos con error de reconstruccion > percentil 95 son anomalias.**

```python
error = MSE(latido_original, latido_reconstruido)
umbral = percentil_95(todos_los_errores)
if error > umbral:
    anomalia = 1
else:
    anomalia = 0
```

### Por que percentil 95
Si el dataset tiene ~25% de anomalias pero no todas son "extremas", el percentil 95 captura los latidos con la reconstruccion mas deficiente. Este umbral es configurable.

### Parametros

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `encoding_dim` | 32 | Dimension del cuello de botella |
| `hidden_layers` | [128, 64] | Capas del encoder (decoder es espejo) |
| `epochs` | 50 | Maximo de iteraciones de entrenamiento |
| `batch_size` | 256 | Tamano del lote para gradient descent |
| `anomaly_percentile` | 95 | Umbral de deteccion |
| `learning_rate` | 0.001 | Tasa de aprendizaje (Adam optimizer) |
| EarlyStopping | patience=5 | Detener si val_loss no mejora en 5 epochs |

### Implementacion (simplificada)

```python
# src/ecg_anomaly/models/autoencoder.py
class AutoencoderDetector(BaseAnomalyDetector):
    def fit(self, X):
        model = build_autoencoder(X.shape[1])  # 200 -> 128 -> 64 -> 32 -> 64 -> 128 -> 200
        model.fit(X, X, epochs=50, callbacks=[EarlyStopping(patience=5)])

        reconstructed = model.predict(X)
        errors = np.mean((X - reconstructed) ** 2, axis=1)  # MSE por latido
        self.threshold_ = np.percentile(errors, 95)
        self.anomaly_labels_ = np.where(errors > self.threshold_, 1, 0)
```

## Tabla resumen comparativa

| Aspecto | K-Means | DBSCAN | HDBSCAN | Autoencoder |
|---------|---------|--------|---------|-------------|
| **Tipo** | Particional | Densidad | Densidad jerarquica | Deep Learning |
| **Ano** | 1957 | 1996 | 2013 | ~2010s |
| **Regla anomalia** | Cluster minoritario | Ruido (-1) | Ruido (-1) | Error > umbral |
| **Asume forma** | Esferica | Cualquiera | Cualquiera | Aprende |
| **Detecta ruido** | No | Si | Si | Si (indirectamente) |
| **Parametros clave** | K | eps, min_samples | min_cluster_size | arquitectura, umbral |
| **Ventaja** | Rapido, simple | No asume forma | Auto-configura | Aprende representacion |
| **Libreria** | sklearn | sklearn | sklearn/hdbscan | TensorFlow/Keras |

## Archivos del codigo relevantes

| Archivo | Que contiene |
|---------|-------------|
| `src/ecg_anomaly/models/base.py` | Clase abstracta `BaseAnomalyDetector` |
| `src/ecg_anomaly/models/kmeans.py` | K-Means con regla de cluster mayoritario |
| `src/ecg_anomaly/models/dbscan.py` | DBSCAN con auto-eps |
| `src/ecg_anomaly/models/hdbscan_model.py` | HDBSCAN con fallback a sklearn/hdbscan |
| `src/ecg_anomaly/models/autoencoder.py` | Autoencoder TF/Keras con umbral adaptativo |
| `src/ecg_anomaly/models/factory.py` | Fabrica para crear detectores por nombre |

---

**Anterior:** [03 - Extraccion de Features](03_extraccion_features.md) | **Siguiente:** [05 - Evaluacion](05_evaluacion.md)
