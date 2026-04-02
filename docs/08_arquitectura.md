# 08 - Arquitectura del Codigo

## Principios de diseno

El codigo sigue principios de ingenieria de software profesional:

1. **Modularidad:** Cada modulo tiene una responsabilidad unica
2. **Extensibilidad:** Agregar un nuevo modelo o metrica requiere minimo cambio
3. **Configurabilidad:** Todo se controla desde YAML, no hardcodeado
4. **Testabilidad:** Cada componente es testeable independientemente
5. **Reproducibilidad:** Seeds fijos, configuracion versionada, logging completo

## Patrones de diseno utilizados

### 1. Factory (Fabrica)

**Problema:** Crear diferentes tipos de objetos (modelos, cargadores) sin que el codigo que los usa sepa los detalles de cada tipo.

**Solucion:** Una clase fabrica que recibe un nombre y retorna la instancia correcta.

```python
# Sin Factory (malo - hay que modificar cada vez que se agrega un modelo):
if model_name == "kmeans":
    detector = KMeansDetector(params)
elif model_name == "dbscan":
    detector = DBSCANDetector(params)
elif model_name == "hdbscan":
    ...

# Con Factory (bueno - agregar modelo = una linea):
detector = DetectorFactory.create("kmeans", params)
```

**Donde se usa:**
- `src/ecg_anomaly/models/factory.py` → Crea detectores por nombre

**Para agregar un nuevo modelo:**
```python
# 1. Crear la clase (ej: src/ecg_anomaly/models/isolation_forest.py)
class IsolationForestDetector(BaseAnomalyDetector):
    def fit(self, X): ...
    def predict_anomalies(self, X): ...

# 2. Registrar en la fabrica (una linea)
DetectorFactory.register("isolation_forest", IsolationForestDetector)

# 3. Agregar al config YAML
models: [kmeans, dbscan, hdbscan, autoencoder, isolation_forest]
```

### 2. Strategy (Estrategia)

**Problema:** Los 4 modelos hacen lo mismo (detectar anomalias) pero de formas diferentes.

**Solucion:** Una interfaz comun (`BaseAnomalyDetector`) que cada modelo implementa a su manera.

```python
# Interfaz comun (src/ecg_anomaly/models/base.py)
class BaseAnomalyDetector(ABC):
    @abstractmethod
    def fit(self, X): ...           # Entrenar

    @abstractmethod
    def predict_anomalies(self, X): ...  # Predecir anomalias

    @abstractmethod
    def get_params(self): ...       # Obtener parametros
```

**Beneficio:** El comparador no sabe ni le importa que modelo esta evaluando. Todos se tratan igual:

```python
# src/ecg_anomaly/evaluation/comparator.py
for model_name in config.models:
    detector = DetectorFactory.create(model_name, params)
    detector.fit(X)  # Funciona igual para KMeans, DBSCAN, HDBSCAN o Autoencoder
    metrics = evaluate_extrinsic(true_labels, detector.anomaly_labels_)
```

### 3. Facade (Fachada)

**Problema:** El pipeline tiene muchos pasos y componentes internos. El usuario no deberia coordinarlos manualmente.

**Solucion:** Una clase que oculta la complejidad y expone una interfaz simple.

```python
# El usuario solo necesita esto:
pipeline = ECGAnomalyPipeline(config)
results = pipeline.run()  # Una linea ejecuta TODO

# Internamente, run() coordina:
# loader.load() → preprocessor.run() → extractor.fit_transform()
# → comparator.run_all() → reporter.save_full_report()
```

**Donde se usa:** `src/ecg_anomaly/pipeline.py` → `ECGAnomalyPipeline`

### 4. Dataclass (Datos estructurados)

**Problema:** Pasar multiples datos entre funciones usando tuplas o diccionarios es fragil y dificil de documentar.

**Solucion:** Clases de datos con campos tipados.

```python
@dataclass
class ECGRecord:
    record_id: str
    signal: np.ndarray           # Senal continua
    r_peak_positions: np.ndarray # Posiciones R
    symbols: np.ndarray          # Simbolos AAMI
    binary_labels: np.ndarray    # 0/1
    sampling_rate: int = 360

@dataclass
class PreprocessedData:
    segments: np.ndarray    # [N, 200]
    labels: np.ndarray      # [N]
    r_peaks_flat: np.ndarray
```

**Beneficio:** Autocompletado en IDE, documentacion implicita, imposible confundir campos.

## Mapa de dependencias entre modulos

```
config.py (sin dependencias)
    │
    ├──→ data/registry.py (sin dependencias externas)
    ├──→ data/loader.py (depende de: config, registry)
    │
    ├──→ preprocessing/filters.py (sin dependencias del proyecto)
    ├──→ preprocessing/qrs_detection.py (sin dependencias del proyecto)
    ├──→ preprocessing/segmentation.py (sin dependencias del proyecto)
    ├──→ preprocessing/pipeline.py (depende de: config, loader, filters, segmentation)
    │
    ├──→ features/signal_pca.py (sin dependencias del proyecto)
    ├──→ features/manual.py (sin dependencias del proyecto)
    │
    ├──→ models/base.py (sin dependencias del proyecto)
    ├──→ models/kmeans.py (depende de: base)
    ├──→ models/dbscan.py (depende de: base)
    ├──→ models/hdbscan_model.py (depende de: base)
    ├──→ models/autoencoder.py (depende de: base)
    ├──→ models/factory.py (depende de: todos los modelos)
    │
    ├──→ evaluation/intrinsic.py (sin dependencias del proyecto)
    ├──→ evaluation/extrinsic.py (sin dependencias del proyecto)
    ├──→ evaluation/efficiency.py (sin dependencias del proyecto)
    ├──→ evaluation/comparator.py (depende de: config, modelos, evaluation/*)
    │
    ├──→ visualization/* (sin dependencias del proyecto)
    │
    └──→ pipeline.py (depende de: TODO lo anterior - es el Facade)
```

**Nota:** Los modulos de hojas (sin dependencias del proyecto) son los mas faciles de testear y reutilizar.

## Estructura de archivos completa

```
src/ecg_anomaly/
├── __init__.py              # Paquete raiz
├── config.py                # SystemConfig dataclass + carga YAML/JSON
│
├── data/
│   ├── __init__.py
│   ├── registry.py          # Catalogo MIT-BIH, constantes AAMI, clasificacion
│   └── loader.py            # MITBIHLoader → ECGDataset con registros
│
├── preprocessing/
│   ├── __init__.py
│   ├── filters.py           # butterworth_bandpass(), notch_filter()
│   ├── qrs_detection.py     # pan_tompkins(), xqrs_detect()
│   ├── segmentation.py      # segment_beats(), normalize_beats()
│   └── pipeline.py          # PreprocessingPipeline → PreprocessedData
│
├── features/
│   ├── __init__.py
│   ├── signal_pca.py        # SignalPCAExtractor (Path A)
│   └── manual.py            # ManualFeatureExtractor (Path B, 12 features)
│
├── models/
│   ├── __init__.py
│   ├── base.py              # BaseAnomalyDetector (ABC)
│   ├── kmeans.py            # KMeansDetector (anomalia = cluster minoritario)
│   ├── dbscan.py            # DBSCANDetector (anomalia = ruido, auto-eps)
│   ├── hdbscan_model.py     # HDBSCANDetector (anomalia = ruido, auto-config)
│   ├── autoencoder.py       # AutoencoderDetector (anomalia = error > umbral)
│   └── factory.py           # DetectorFactory (crea por nombre)
│
├── evaluation/
│   ├── __init__.py
│   ├── intrinsic.py         # evaluate_intrinsic() → Silhouette, DB, CH
│   ├── extrinsic.py         # evaluate_extrinsic() → F1, Sensitivity, AUC
│   ├── efficiency.py        # EfficiencyTracker (context manager)
│   └── comparator.py        # ModelComparator (orquesta 3 niveles)
│
├── visualization/
│   ├── __init__.py
│   ├── signals.py           # Graficos de senales ECG
│   ├── clusters.py          # Scatter PCA, distribucion anomalias
│   └── reports.py           # Tablas comparativas, confusion matrices
│
└── pipeline.py              # ECGAnomalyPipeline (Facade) + main() (CLI)
```

## Gestion de dependencias con Poetry

Poetry maneja las dependencias del proyecto de forma reproducible:

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
numpy = "^1.24"           # Calculo numerico
scipy = "^1.10"           # Filtros digitales
pandas = "^2.0"           # Tablas de resultados
scikit-learn = "^1.3"     # KMeans, DBSCAN, HDBSCAN, PCA, metricas
wfdb = "^4.1"             # Lectura MIT-BIH
hdbscan = "^0.8.33"       # Fallback si sklearn < 1.3
tensorflow = "^2.14"      # Autoencoder
matplotlib = "^3.7"       # Graficos
seaborn = "^0.13"         # Graficos estadisticos
plotly = "^5.18"          # Graficos interactivos
pyyaml = "^6.0"           # Configuracion YAML
psutil = "^5.9"           # Medicion de memoria
```

**Comandos clave:**
```bash
poetry install              # Instalar todo
poetry add nueva-libreria   # Agregar dependencia
poetry run pytest           # Ejecutar tests
poetry run ecg-run          # Ejecutar pipeline
```

## Tests

Los tests verifican que cada componente funciona correctamente de forma aislada:

```
tests/
├── test_preprocessing.py   # Filtros, segmentacion, normalizacion
├── test_features.py        # PCA, features manuales
└── test_models.py          # Detectores, factory, metricas
```

```bash
poetry run pytest -v        # Ejecutar con detalle
poetry run pytest --cov     # Con cobertura de codigo
```

---

**Anterior:** [07 - Metodologia DSR](07_metodologia_dsr.md) | **Indice:** [Documentacion](README.md)
