# 06 - Pipeline y Ejecucion

## Que es el pipeline

El pipeline es el **flujo completo** que conecta todos los pasos documentados anteriormente en una sola ejecucion automatizada. Toma datos crudos de MIT-BIH y produce la tabla comparativa final.

```
ecg-run --config config/default.yaml
    │
    ├── [1/5] Carga datos MIT-BIH (44 registros, agrupacion AAMI)
    ├── [2/5] Preprocesamiento (filtrado → segmentacion → normalizacion)
    ├── [3/5] Extraccion de features (Path A o Path B)
    ├── [4/5] Ejecucion de 4 modelos + evaluacion
    └── [5/5] Generacion de reporte (tabla + graficos)
```

## Configuracion

Todo se controla desde un archivo YAML:

```yaml
# config/default.yaml
dataset_name: mitbih
dataset_path: ./data/mitbih

models:                    # Los 4 modelos a comparar
  - kmeans
  - dbscan
  - hdbscan
  - autoencoder

representation: signal_pca  # "signal_pca" o "manual_features"

# Parametros de senal
sampling_rate: 360
before_r_samples: 90       # 90 muestras antes del pico R
after_r_samples: 110       # 110 muestras despues
filter_lowcut: 0.5         # Hz
filter_highcut: 40.0       # Hz

# Hiperparametros de cada modelo
kmeans_params:
  n_clusters: 2
  random_state: 42

dbscan_params:
  eps: auto                # Se calcula automaticamente
  min_samples: 10

hdbscan_params:
  min_cluster_size: 15
  min_samples: 10

autoencoder_params:
  encoding_dim: 32
  hidden_layers: [128, 64]
  epochs: 50
  anomaly_percentile: 95
```

### Cambiar representacion

Para comparar Path A vs Path B, solo cambia una linea:

```yaml
representation: manual_features  # Cambia de signal_pca a manual_features
```

## Formas de ejecutar

### 1. Linea de comandos (CLI)

```bash
# Con configuracion por defecto
poetry run ecg-run

# Con representacion manual
poetry run ecg-run --representation manual_features

# Con configuracion personalizada
poetry run ecg-run --config config/mi_config.yaml
```

### 2. Desde Python

```python
from ecg_anomaly.config import SystemConfig
from ecg_anomaly.pipeline import ECGAnomalyPipeline

config = SystemConfig.from_yaml("config/default.yaml")
pipeline = ECGAnomalyPipeline(config)
results = pipeline.run()

# results es un DataFrame con la tabla comparativa
print(results)
```

### 3. Paso a paso (para explorar)

```python
from ecg_anomaly.config import SystemConfig
from ecg_anomaly.data.loader import MITBIHLoader
from ecg_anomaly.preprocessing.pipeline import PreprocessingPipeline
from ecg_anomaly.features.signal_pca import SignalPCAExtractor
from ecg_anomaly.models.factory import DetectorFactory
from ecg_anomaly.evaluation.extrinsic import evaluate_extrinsic

config = SystemConfig.from_yaml("config/default.yaml")

# Paso 1: Cargar
loader = MITBIHLoader(config)
dataset = loader.load(config.dataset_path)

# Paso 2: Preprocesar
preprocessor = PreprocessingPipeline(config)
data = preprocessor.run(dataset)

# Paso 3: Features
extractor = SignalPCAExtractor(0.95)
X = extractor.fit_transform(data.segments)

# Paso 4: Un modelo
detector = DetectorFactory.create("hdbscan", {"min_cluster_size": 15})
detector.fit(X)

# Paso 5: Evaluar
metrics = evaluate_extrinsic(data.labels, detector.anomaly_labels_)
print(f"F1: {metrics['f1']:.3f}")
```

## Salida del pipeline

### En consola
```
08:30:01 [ecg_anomaly.pipeline] INFO: ============================================================
08:30:01 [ecg_anomaly.pipeline] INFO: SISTEMA DE DETECCION DE ANOMALIAS ECG
08:30:01 [ecg_anomaly.pipeline] INFO: ============================================================
08:30:01 [ecg_anomaly.pipeline] INFO: [1/5] Cargando datos MIT-BIH...
08:30:15 [ecg_anomaly.data.loader] INFO: Cargados 44 registros: 98750 latidos (74120 normal, 24630 anomalo)
08:30:15 [ecg_anomaly.pipeline] INFO: [2/5] Preprocesando senales...
08:30:22 [ecg_anomaly.preprocessing.pipeline] INFO: Preprocesamiento completo: 97500 latidos
08:30:22 [ecg_anomaly.pipeline] INFO: [3/5] Extrayendo features (representacion: signal_pca)...
08:30:23 [ecg_anomaly.features.signal_pca] INFO: PCA: 200 -> 8 componentes (95.2% varianza)
08:30:23 [ecg_anomaly.pipeline] INFO: [4/5] Ejecutando y evaluando modelos...
08:30:24 [ecg_anomaly.evaluation.comparator] INFO: kmeans completado: F1=0.xxx, Silhouette=0.xxx, Tiempo=0.xxs
08:30:25 [ecg_anomaly.evaluation.comparator] INFO: dbscan completado: F1=0.xxx, Silhouette=0.xxx, Tiempo=0.xxs
...
08:31:00 [ecg_anomaly.pipeline] INFO: MEJOR MODELO (F1): hdbscan

 Modelo  Silhouette  ...  F1     Sensitivity  Tiempo(s)
 kmeans  0.xxx       ...  0.xxx  0.xxx        0.xx
 dbscan  0.xxx       ...  0.xxx  0.xxx        0.xx
 hdbscan 0.xxx       ...  0.xxx  0.xxx        0.xx
 autoenc None        ...  0.xxx  0.xxx        x.xx
```

### Archivos generados en `results/`
```
results/
└── report_20260401_083100/
    ├── comparison_table.csv       # Tabla en formato CSV
    ├── metrics_comparison.png     # Grafico de barras
    └── confusion_matrices.png     # Matrices de confusion
```

## Patron Facade

El pipeline usa el **patron Facade** (Fachada): oculta toda la complejidad interna detras de una interfaz simple.

```
                    Usuario
                       │
                       ▼
            ┌─────────────────────┐
            │  ECGAnomalyPipeline │  ← Facade (interfaz simple)
            │     .run()          │
            └────────┬────────────┘
                     │
    ┌────────────────┼──────────────────────┐
    ▼                ▼                      ▼
MITBIHLoader  PreprocessingPipeline  ModelComparator
                                           │
                                    ┌──────┼──────┐──────┐
                                    ▼      ▼      ▼      ▼
                                 KMeans DBSCAN HDBSCAN AutoEnc
```

El usuario solo necesita: `pipeline.run()`. Internamente se coordinan 7+ clases.

## Archivos del codigo relevantes

| Archivo | Que contiene |
|---------|-------------|
| `src/ecg_anomaly/pipeline.py` | `ECGAnomalyPipeline` (Facade) + `main()` (CLI) |
| `config/default.yaml` | Configuracion por defecto |
| `pyproject.toml` | Script `ecg-run` apuntando a `pipeline:main` |

---

**Anterior:** [05 - Evaluacion](05_evaluacion.md) | **Siguiente:** [07 - Metodologia DSR](07_metodologia_dsr.md)
