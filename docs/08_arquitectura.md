# 08 - Arquitectura del Codigo

## Principios de diseno

El codigo sigue principios de ingenieria de software profesional:

1. **Modularidad:** Cada modulo tiene una responsabilidad unica
2. **Extensibilidad:** Agregar un nuevo modelo o metrica requiere minimo cambio
3. **Configurabilidad:** Todo se controla desde YAML, no hardcodeado
4. **Testabilidad:** Cada componente es testeable independientemente
5. **Reproducibilidad:** Seeds fijos, configuracion versionada, logging completo

## Estructura de carpetas del proyecto

```
tfp-cardio/
├── config/                  # Configuracion centralizada
│   └── default.yaml
├── data/                    # Datos MIT-BIH descargados (no versionados)
├── docs/                    # Documentacion del TFP
├── notebooks/               # Jupyter notebooks de exploracion
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_evaluation.ipynb
├── results/                 # Salidas del pipeline (graficos, reportes, modelos)
├── src/ecg_anomaly/         # Paquete principal (codigo de produccion)
│   ├── config.py
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── visualization/
│   └── pipeline.py
├── tests/                   # Tests unitarios
├── pyproject.toml           # Configuracion del proyecto y dependencias (Poetry)
└── requirements.txt         # Dependencias alternativas (pip)
```

### Justificacion de cada carpeta

| Carpeta | Proposito | Por que esta separada |
|---|---|---|
| `config/` | Archivos YAML con hiperparametros, rutas y opciones del pipeline | Permite modificar el comportamiento del sistema **sin tocar codigo**. Facilita la experimentacion y la reproducibilidad al versionar la configuracion junto al proyecto. |
| `data/` | Almacena los registros MIT-BIH en formato WFDB (.dat, .hea, .atr) | Se excluye del control de versiones (`.gitignore`) porque los datos son pesados y publicos. Separar datos de codigo evita que el repositorio crezca innecesariamente. |
| `docs/` | Documentacion tecnica del trabajo final de posgrado | Mantener la documentacion junto al codigo garantiza que se actualice en paralelo y sirve como referencia inmediata para cualquier colaborador. |
| `notebooks/` | Jupyter notebooks para exploracion, prototipado y visualizacion interactiva | Los notebooks son herramientas de investigacion exploratoria; separarlos del paquete `src/` deja claro que no son parte del sistema de produccion sino del proceso de analisis previo. |
| `results/` | Graficos, tablas comparativas, modelos serializados y reportes generados por el pipeline | Se excluye del control de versiones porque son **artefactos reproducibles**: ejecutar el pipeline los regenera. Separarlos evita mezclar salidas con codigo fuente. |
| `src/ecg_anomaly/` | Paquete Python instalable con todo el codigo de produccion | Usar la convencion `src/` layout aisla el paquete del directorio raiz, evitando importaciones accidentales de archivos locales y asegurando que los tests siempre importan la version instalada del paquete. |
| `tests/` | Tests unitarios con pytest | Separar tests del codigo fuente es la convencion estandar de Python. Permite ejecutar `pytest` desde la raiz sin contaminar el paquete distribuible. |

### Por que esta organizacion

La estructura sigue tres criterios:

1. **Separacion de concerns:** Cada carpeta tiene un rol unico — configuracion, datos, codigo, tests, resultados y documentacion no se mezclan.
2. **Reproducibilidad:** Al separar lo que se versiona (codigo, config, docs) de lo que se genera (data, results), cualquier persona puede clonar el repositorio, descargar los datos y reproducir los resultados desde cero.
3. **Escalabilidad:** Agregar un nuevo modelo, una nueva metrica o un nuevo notebook no requiere reorganizar carpetas existentes; cada tipo de artefacto tiene su lugar predefinido.

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

## Detalle del paquete `src/ecg_anomaly/`

El paquete `src/ecg_anomaly/` contiene todo el codigo de produccion. Cada subpaquete corresponde a una etapa del pipeline de deteccion de anomalias en senales ECG y se puede entender, testear y reutilizar de forma independiente.

```
src/ecg_anomaly/
├── config.py                # Configuracion centralizada
├── data/                    # Carga de datos MIT-BIH
├── preprocessing/           # Filtrado, segmentacion y normalizacion
├── features/                # Extraccion de representaciones
├── models/                  # Algoritmos de deteccion
├── evaluation/              # Metricas y comparacion
├── visualization/           # Graficos y reportes visuales
└── pipeline.py              # Orquestador principal (Facade)
```

### `config.py` — Configuracion centralizada

Define la dataclass `SystemConfig` que contiene **todos** los parametros del sistema: rutas de datos, hiperparametros de cada modelo, especificaciones de filtros (Butterworth 0.5–40 Hz, orden 4), ventana de segmentacion (90 muestras antes y 110 despues del pico R = 200 muestras por latido), y umbral de varianza para PCA (95%).

**Por que existe:** Centralizar la configuracion en un unico punto permite cambiar el comportamiento del sistema editando un archivo YAML, sin modificar codigo. Tambien facilita la reproducibilidad: cada experimento queda definido por su archivo de configuracion.

**Que expone:**
- `SystemConfig.from_yaml()` / `from_json()` — Carga desde archivo
- `SystemConfig.save_yaml()` — Persiste la configuracion usada
- `SystemConfig.setup_logging()` — Configura el logging del sistema

### `data/` — Carga y catalogo de datos

| Archivo | Que hace |
| --- | --- |
| `registry.py` | Define las constantes del estandar AAMI: simbolos normales (N, L, R, e, j), anomalos (A, a, J, S, V, E, F, /, f, Q) y no-latido. Cataloga los 48 registros MIT-BIH en 5 categorias clinicas y excluye los 4 registros con marcapasos (102, 104, 107, 217). |
| `loader.py` | `MITBIHLoader` carga los registros usando la libreria `wfdb`, aplica la clasificacion binaria del registry (0=normal, 1=anomalo) y retorna un `ECGDataset` con objetos `ECGRecord` (senal, posiciones R, etiquetas). |

**Por que es un subpaquete separado:** Aislar la logica de acceso a datos permite cambiar la fuente (archivos locales vs PhysioNet) sin afectar al resto del sistema. El registry actua como fuente unica de verdad para la clasificacion de latidos.

### `preprocessing/` — Procesamiento de senales

| Archivo | Que hace |
| --- | --- |
| `filters.py` | Implementa `butterworth_bandpass()` (elimina deriva de linea base <0.5 Hz y ruido >40 Hz con filtrado de fase cero) y `notch_filter()` (elimina interferencia de red electrica 50/60 Hz). |
| `qrs_detection.py` | Detectores de picos R: `pan_tompkins()` (algoritmo clasico con bandpass, derivada, cuadrado y media movil) y `xqrs_detect()` (detector robusto de WFDB). Opcionales, ya que MIT-BIH provee anotaciones. |
| `segmentation.py` | `segment_beats()` extrae ventanas de longitud fija alrededor de cada pico R. `normalize_beats()` aplica normalizacion Z-score por latido. |
| `pipeline.py` | `PreprocessingPipeline` orquesta las tres etapas (filtrar → segmentar → normalizar) y retorna un `PreprocessedData` con los segmentos listos para extraer features. |

**Por que es un subpaquete separado:** El preprocesamiento de senales biomedicas tiene complejidad propia (filtros digitales, deteccion QRS, normalizacion). Separarlo permite testear cada transformacion individualmente y reutilizar los filtros en otros contextos.

### `features/` — Extraccion de representaciones

Ofrece **dos caminos** para representar los latidos como vectores numericos que los modelos de clustering pueden procesar:

| Archivo | Camino | Que hace |
| --- | --- | --- |
| `signal_pca.py` | **Path A: Senal + PCA** | `SignalPCAExtractor` aplica StandardScaler + PCA, reduciendo los 200 puntos del latido a las componentes que retienen el 95% de la varianza. Tambien expone `get_raw_for_autoencoder()` que retorna los datos escalados sin reducir (el autoencoder maneja su propia reduccion via la capa de encoding). |
| `manual.py` | **Path B: Features manuales** | `ManualFeatureExtractor` calcula 12 features morfologicas y estadisticas por latido: amplitud R, amplitud S, duracion QRS, rango, intervalo RR actual, ratio RR, diferencia RR, media, desviacion estandar, curtosis, frecuencia dominante (FFT) y energia espectral. |

**Por que es un subpaquete separado:** La representacion de datos es una decision de diseno clave que afecta directamente el rendimiento de los modelos. Tener dos caminos independientes permite comparar empiricamente cual representacion produce mejores resultados, que es uno de los objetivos del trabajo.

### `models/` — Algoritmos de deteccion de anomalias

Implementa 4 detectores con complejidad creciente, todos bajo la misma interfaz abstracta:

| Archivo | Nivel | Estrategia de deteccion |
| --- | --- | --- |
| `base.py` | — | Define `BaseAnomalyDetector` (ABC) con los metodos `fit()`, `predict_anomalies()` y `get_params()`. Todos los detectores heredan de esta clase. |
| `kmeans.py` | 1 — Baseline | `KMeansDetector`: el cluster mayoritario es normal, los minoritarios son anomalias. Parametros: k=2, seed=42. |
| `dbscan.py` | 2 — Densidad | `DBSCANDetector`: los puntos marcados como ruido (label=-1) son anomalias. Incluye `_optimize_eps()` que calcula epsilon automaticamente con el metodo de la distancia k al percentil 90. |
| `hdbscan_model.py` | 3 — Densidad jerarquica | `HDBSCANDetector`: misma logica que DBSCAN pero sin necesidad de epsilon; selecciona la densidad optima automaticamente. |
| `autoencoder.py` | 4 — Deep learning | `AutoencoderDetector`: entrena una red encoder-decoder simetrica (input→128→64→32→64→128→input). Los latidos con error de reconstruccion > percentil 95 se clasifican como anomalias. |
| `factory.py` | — | `DetectorFactory`: crea detectores por nombre. Agregar un modelo nuevo requiere una sola linea de registro. |

**Por que es un subpaquete separado:** La comparacion de multiples algoritmos es el nucleo del trabajo. El patron Strategy permite que el comparador trate todos los modelos de forma identica, y el Factory desacopla la creacion de la ejecucion.

### `evaluation/` — Metricas y comparacion

Implementa tres niveles de evaluacion alineados con la metodologia del trabajo:

| Archivo | Nivel | Que mide |
| --- | --- | --- |
| `intrinsic.py` | Intrinseco | `evaluate_intrinsic()` calcula Silhouette (-1 a 1), Davies-Bouldin (menor=mejor) y Calinski-Harabasz — miden la calidad de los clusters sin usar etiquetas reales. |
| `extrinsic.py` | Extrinseco | `evaluate_extrinsic()` compara predicciones contra etiquetas AAMI reales: Accuracy, Sensitivity, Specificity, Precision, F1, AUC-ROC y la matriz de confusion (TP, FP, TN, FN). |
| `efficiency.py` | Eficiencia | `EfficiencyTracker` es un context manager que mide tiempo de ejecucion y pico de memoria (via `tracemalloc`) de cada modelo. |
| `comparator.py` | Orquestador | `ModelComparator` ejecuta los 4 modelos, recolecta las metricas de los 3 niveles y genera una tabla comparativa como DataFrame. |

**Por que es un subpaquete separado:** Separar las metricas del entrenamiento permite reutilizar las funciones de evaluacion con cualquier modelo (incluso externo) y agregar nuevas metricas sin modificar los detectores.

### `visualization/` — Graficos y reportes visuales

| Archivo | Que genera |
| --- | --- |
| `signals.py` | Graficos de senales ECG: senal cruda vs filtrada, senal con picos R marcados, superposicion de latidos segmentados. |
| `clusters.py` | Scatter 2D de las dos primeras componentes PCA coloreado por cluster o por etiqueta binaria; graficos de barras comparando distribucion real vs predicha de anomalias. |
| `reports.py` | Paneles comparativos de metricas (intrinsecas, extrinsecas, eficiencia) por modelo, heatmaps de matrices de confusion, y `save_full_report()` que genera un directorio completo con CSV + PNG. |

**Por que es un subpaquete separado:** La visualizacion no tiene logica de negocio — solo consume resultados y produce graficos. Separarla permite que los modulos de computo (models, evaluation) sean independientes de la capa de presentacion.

### `pipeline.py` — Orquestador principal (Facade)

`ECGAnomalyPipeline` es el punto de entrada que coordina todas las etapas en una sola llamada:

```python
pipeline = ECGAnomalyPipeline(config)
results = pipeline.run()  # Ejecuta TODO el flujo
```

Internamente ejecuta: carga de datos → preprocesamiento → extraccion de features (Path A o B segun configuracion) → evaluacion de los 4 modelos → generacion de reportes. Tambien expone `main()` como CLI con argumentos `--config` y `--representation`.

**Por que existe como modulo raiz:** Es el unico archivo que depende de todos los demas. Ubicarlo en la raiz del paquete deja claro que es la fachada del sistema y no pertenece a ningun subpaquete especifico.

### Flujo de datos entre subpaquetes

```
config.py ──→ Parametros a todos los modulos
                │
         data/ (MIT-BIH)
                │ ECGDataset
                ▼
      preprocessing/ (Filtrar → Segmentar → Normalizar)
                │ PreprocessedData [N, 200]
                ▼
        features/ (Path A: PCA  |  Path B: 12 features manuales)
                │ X_clustering [N, d]  +  X_autoencoder [N, 200]
                ▼
         models/ (KMeans → DBSCAN → HDBSCAN → Autoencoder)
                │ anomaly_labels_ [N]
                ▼
      evaluation/ (Intrinseco + Extrinseco + Eficiencia)
                │ DataFrame comparativo
                ▼
     visualization/ (Graficos + Reportes)
                │
                ▼
           results/ (CSV + PNG)
```
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
