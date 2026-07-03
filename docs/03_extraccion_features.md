# 03 - Extraccion de Caracteristicas (Features)

## Por que extraer features

Despues del preprocesamiento tenemos latidos de **200 muestras** cada uno. Hay dos filosofias para alimentar los algoritmos:

1. **Darles la senal directa** (200 numeros por latido) y dejar que el algoritmo encuentre patrones
2. **Calcular mediciones especificas** (12 numeros por latido) que resumen las propiedades mas relevantes

Ambos caminos tienen ventajas. El proyecto evalua **ambos** para responder: *¿La ingenieria de caracteristicas manual sigue siendo relevante o es mejor la senal cruda?*

## Path A: Senal directa + PCA

### Que es PCA (Analisis de Componentes Principales)

PCA es una tecnica de **reduccion de dimensionalidad**. Transforma datos de muchas dimensiones a menos dimensiones, reteniendo la mayor cantidad posible de informacion.

**Analogia:** Si tienes 200 fotos de un auto desde diferentes angulos, PCA encuentra que con ~8 "fotos resumen" puedes reconstruir casi perfectamente las 200 originales. Esas 8 fotos capturan el 95% de la informacion.

### El proceso

```
Latidos [N, 200]
    │
    ▼ StandardScaler (media=0, var=1 por columna)
Datos escalados [N, 200]
    │
    ▼ PCA (retener 95% de varianza)
Datos reducidos [N, ~8]     ← Para clustering (KMeans, DBSCAN, HDBSCAN)
    │
    └── Datos escalados [N, 200]  ← Para autoencoder (SIN PCA)
```

### Por que PCA para clustering pero NO para autoencoder

- **Clustering** (KMeans, DBSCAN, HDBSCAN): Estos algoritmos sufren con datos de alta dimensionalidad (200 dimensiones). PCA reduce a ~8 dimensiones donde las distancias entre puntos son mas significativas.
- **Autoencoder**: Su proposito ES aprender una representacion comprimida. Si le damos datos ya comprimidos con PCA, le estamos quitando trabajo. El autoencoder recibe las 200 dimensiones escaladas y aprende su propia compresion interna.

### Parametros

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| Escalado | StandardScaler | Media=0, varianza=1 por dimension |
| Varianza retenida | 95% | Estandar en la literatura, buen balance |
| Componentes resultantes | ~8 | Determinado automaticamente por el 95% |

### Implementacion

```python
# src/ecg_anomaly/features/signal_pca.py
class SignalPCAExtractor:
    def __init__(self, variance_threshold=0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_threshold)

    def fit_transform(self, segments):       # Para clustering
        scaled = self.scaler.fit_transform(segments)
        return self.pca.fit_transform(scaled)  # [N, ~8]

    def get_raw_for_autoencoder(self, segments):  # Para autoencoder
        return self.scaler.transform(segments)     # [N, 200]
```

## Path B: Features manuales (~12 caracteristicas)

### Que son

En vez de dar la senal cruda, calculamos **mediciones especificas** que un cardiologo miraria: la altura del pico R, la duracion del complejo QRS, los intervalos entre latidos, etc.

### Las 12 features

#### Morfologicas (4 features) - "¿Como se ve el latido?"

| # | Feature | Que mide | Por que importa |
|---|---------|----------|-----------------|
| 1 | `r_amplitude` | Altura del pico R | La onda R es la mas prominente del ECG. Su amplitud varia en arritmias ventriculares. |
| 2 | `s_amplitude` | Profundidad de la onda S | Una S profunda o ausente indica bloqueo de rama o hipertrofia. |
| 3 | `qrs_duration` | Duracion del complejo QRS (ms) | QRS ancho (>120ms) indica conduccion ventricular anormal. Valores tipicos: 80-120ms. |
| 4 | `amplitude_range` | Diferencia max-min | Mide la "energia" total del latido. Latidos ectopicos suelen tener morfologia diferente. |

#### Intervalos RR (3 features) - "¿Que tan regular es el ritmo?"

| # | Feature | Que mide | Por que importa |
|---|---------|----------|-----------------|
| 5 | `rr_current` | Intervalo RR actual (ms) | Tiempo entre este latido y el anterior. Un corazon normal late cada 600-1000ms. |
| 6 | `rr_ratio` | Ratio RR actual / RR medio | Valores lejos de 1.0 indican latidos prematuros (< 1.0) o pausas (> 1.0). |
| 7 | `rr_diff` | Diferencia RR actual - RR previo | Cambios bruscos sugieren arritmias. Un latido prematuro tiene RR_diff muy negativo. |

#### Estadisticas (3 features) - "¿Cual es la distribucion de valores?"

| # | Feature | Que mide | Por que importa |
|---|---------|----------|-----------------|
| 8 | `mean` | Media del segmento | Tras normalizacion deberia ser ~0, pero variaciones indican asimetria morfologica. |
| 9 | `std` | Desviacion estandar | Mide la variabilidad. Latidos con morfologia "plana" tienen std bajo. |
| 10 | `kurtosis` | Curtosis (concentracion en pico) | Alta curtosis = pico muy pronunciado. Baja = forma mas distribuida. |

#### Frecuencia (2 features) - "¿Que frecuencias tiene la forma de onda?"

| # | Feature | Que mide | Por que importa |
|---|---------|----------|-----------------|
| 11 | `dominant_freq` | Frecuencia dominante (FFT) | Los complejos QRS normales tienen componentes de frecuencia caracteristicos. |
| 12 | `spectral_energy` | Energia espectral total | Latidos anomalos con formas distorsionadas tienen diferente distribucion energetica. |

### Implementacion

```python
# src/ecg_anomaly/features/manual.py
class ManualFeatureExtractor:
    def extract(self, segments, r_peak_positions, fs=360):
        features = np.zeros((len(segments), 12))
        rr_intervals = np.diff(r_peak_positions) / fs * 1000  # ms

        for i, seg in enumerate(segments):
            r_idx = 90  # Pico R esta en posicion 90
            features[i, 0] = seg[r_idx]                       # r_amplitude
            features[i, 1] = np.min(seg[r_idx:r_idx+30])      # s_amplitude
            features[i, 2] = estimate_qrs_duration(seg, r_idx) # qrs_duration
            features[i, 3] = np.max(seg) - np.min(seg)        # amplitude_range
            features[i, 4] = rr_intervals[i]                  # rr_current
            # ... (12 features total)

        return self.scaler.fit_transform(features)  # Escalar
```

## Comparacion de ambos caminos

| Aspecto | Path A (Senal + PCA) | Path B (Features manuales) |
|---------|---------------------|---------------------------|
| **Input** | 200 muestras de senal | 12 numeros calculados |
| **Dimensiones para clustering** | ~8 (tras PCA) | 12 |
| **Dimensiones para autoencoder** | 200 (sin PCA) | 200 (senal escalada) |
| **Informacion preservada** | 95% de la varianza total | Solo lo que el humano eligio medir |
| **Ventaja** | No pierde informacion morfologica sutil | Vector compacto, mas interpretable |
| **Desventaja** | Menos interpretable | Puede perder patrones que no se midieron |

### Pregunta que esto responde

> Si Path B (12 features humanas) funciona igual de bien que Path A (200 dim + PCA), entonces la ingenieria de features manual sigue siendo valiosa y es mas eficiente. Si Path A es claramente superior, indica que hay informacion en la senal que las 12 features no capturan.

## Archivos del codigo relevantes

| Archivo | Que contiene |
|---------|-------------|
| `src/ecg_anomaly/features/signal_pca.py` | Path A: StandardScaler + PCA + datos raw para autoencoder |
| `src/ecg_anomaly/features/manual.py` | Path B: 12 features + escalado |

---

**Anterior:** [02 - Preprocesamiento](02_preprocesamiento.md) | **Siguiente:** [04 - Modelos](04_modelos.md)
