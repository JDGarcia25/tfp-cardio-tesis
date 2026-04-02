# 09 - Manual de Ejecucion

## Requisitos del sistema

| Requisito | Minimo | Recomendado |
|-----------|--------|-------------|
| **Python** | 3.10 | 3.12 |
| **Poetry** | 1.7+ | 2.0+ |
| **RAM** | 4 GB | 8 GB |
| **Disco** | 500 MB (sin datos) | 2 GB (con MIT-BIH + venv) |
| **OS** | macOS, Linux, Windows | macOS (Apple Silicon), Linux |

## Paso 1: Clonar el repositorio

```bash
git clone <URL-del-repositorio>
cd tfp-cardio
```

## Paso 2: Instalar Poetry

Poetry es el gestor de dependencias del proyecto. Si no lo tienes instalado:

```bash
# macOS / Linux
curl -sSL https://install.python-poetry.org | python3 -

# Verificar instalacion
poetry --version
```

> **¿Por que Poetry y no pip?** Poetry resuelve dependencias de forma determinista
> (genera `poetry.lock`), maneja el entorno virtual automaticamente, y permite
> declarar todas las dependencias en un solo archivo (`pyproject.toml`). Esto
> garantiza que cualquier persona que instale el proyecto obtendra exactamente
> las mismas versiones de librerias.

## Paso 3: Instalar dependencias con Poetry

```bash
# Configurar Poetry para crear el venv dentro del proyecto
poetry env use 3.12
poetry config virtualenvs.in-project true

# Instalar todas las dependencias (produccion + desarrollo)
poetry install
```

Esto automaticamente:
1. Crea un entorno virtual en `.venv/`
2. Instala todas las dependencias de `pyproject.toml` (numpy, scipy, sklearn, tensorflow, etc.)
3. Instala el paquete `ecg_anomaly` en modo editable

### Verificar instalacion

```bash
poetry run python -c "
import numpy, scipy, pandas, sklearn, wfdb, pywt, hdbscan
import tensorflow, matplotlib, seaborn, plotly, yaml, psutil
print('Todas las dependencias instaladas correctamente')
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
import sklearn
print(f'scikit-learn: {sklearn.__version__}')
"
```

## Paso 4: Verificar datos MIT-BIH

Los archivos de MIT-BIH deben estar accesibles en `data/mitbih/`. El proyecto incluye un symlink:

```bash
# Verificar que el symlink funciona
ls data/mitbih/100.dat

# Si no funciona, crear el symlink manualmente:
ln -sf ../BD2-20260218T192722Z-1-001/BD2 data/mitbih

# O cambiar la ruta en config/default.yaml:
# dataset_path: ./BD2-20260218T192722Z-1-001/BD2
```

### Verificar que los datos se cargan

```bash
poetry run python -c "
from ecg_anomaly.config import SystemConfig
from ecg_anomaly.data.loader import MITBIHLoader
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

config = SystemConfig.from_yaml('config/default.yaml')
loader = MITBIHLoader(config)
dataset = loader.load(config.dataset_path)
print(f'OK: {dataset.total_beats} latidos de {len(dataset.records)} registros')
"
```

**Salida esperada:**
```
Cargados 44 registros: 100733 latidos (90125 normal, 10608 anomalo)
OK: 100733 latidos de 44 registros
```

## Paso 5: Ejecutar los tests

```bash
poetry run pytest tests/ -v
```

**Salida esperada:**
```
tests/test_features.py::TestSignalPCAExtractor::test_reduces_dimensions PASSED
tests/test_features.py::TestSignalPCAExtractor::test_preserves_variance PASSED
...
tests/test_models.py::TestDetectorFactory::test_list_detectors PASSED
tests/test_preprocessing.py::TestButterworthBandpass::test_removes_dc_offset PASSED
...
============================== 31 passed in 1.24s ==============================
```

## Paso 6: Ejecutar el pipeline completo

### Opcion A: Usando el script registrado en Poetry

```bash
poetry run ecg-run --config config/default.yaml
```

### Opcion B: Ejecutar el modulo directamente

```bash
poetry run python -m ecg_anomaly.pipeline --config config/default.yaml
```

### Opcion C: Desde un script Python

```python
from ecg_anomaly.config import SystemConfig
from ecg_anomaly.pipeline import ECGAnomalyPipeline

config = SystemConfig.from_yaml("config/default.yaml")
pipeline = ECGAnomalyPipeline(config)
results = pipeline.run()
```

### Opcion D: Solo clustering (sin autoencoder, mas rapido)

```bash
poetry run python -c "
import matplotlib; matplotlib.use('Agg')
from ecg_anomaly.config import SystemConfig
from ecg_anomaly.pipeline import ECGAnomalyPipeline

config = SystemConfig.from_yaml('config/default.yaml')
config.models = ['kmeans', 'dbscan', 'hdbscan']  # Sin autoencoder
pipeline = ECGAnomalyPipeline(config)
results = pipeline.run()
"
```

### Opcion E: Cambiar representacion a features manuales

```bash
poetry run ecg-run --config config/default.yaml --representation manual_features
```

## Salida esperada del pipeline completo

```
17:31:43 [ecg_anomaly.data.loader] INFO: Cargados 44 registros: 100733 latidos (90125 normal, 10608 anomalo)
17:31:43 [ecg_anomaly.preprocessing.pipeline] INFO: Preprocesamiento completo: 100705 latidos
17:31:43 [ecg_anomaly.features.signal_pca] INFO: PCA: 200 -> 12 componentes (95.3% varianza)
17:31:43 [ecg_anomaly.evaluation.comparator] INFO: Evaluando kmeans...
17:32:35 [ecg_anomaly.evaluation.comparator] INFO: kmeans completado: F1=0.177, Tiempo=0.24s
17:32:35 [ecg_anomaly.evaluation.comparator] INFO: Evaluando dbscan...
17:33:32 [ecg_anomaly.evaluation.comparator] INFO: dbscan completado: F1=0.238, Tiempo=11.26s
17:33:32 [ecg_anomaly.evaluation.comparator] INFO: Evaluando hdbscan...
17:34:26 [ecg_anomaly.evaluation.comparator] INFO: hdbscan completado: F1=0.235, Tiempo=21.70s
17:34:26 [ecg_anomaly.evaluation.comparator] INFO: Evaluando autoencoder...
17:35:17 [ecg_anomaly.evaluation.comparator] INFO: autoencoder completado: F1=0.141, Tiempo=50.62s

     Modelo  Silhouette  Davies-Bouldin  Calinski-Harabasz  Accuracy  Sensitivity  Specificity    F1  AUC-ROC  Tiempo (s)  Memoria (MB)  Anomalias
     kmeans       0.256           1.473          38848.107     0.511        0.499        0.512 0.177    0.506        0.24         18.51      49269
     dbscan       0.008           0.861           1391.186     0.866        0.200        0.944 0.238    0.572       11.26         38.35       7167
    hdbscan       0.487           0.756          13092.009     0.757        0.355        0.804 0.235    0.580       21.70         32.47      21387
autoencoder         NaN             NaN                NaN     0.867        0.104        0.956 0.141    0.530       50.62        432.93       5036

Best model by F1-score: dbscan
```

## Tiempos aproximados de ejecucion

| Etapa | Tiempo |
|-------|--------|
| Carga de datos (44 registros) | ~1 segundo |
| Preprocesamiento (filtrado + segmentacion) | ~1 segundo |
| Feature extraction (PCA) | < 1 segundo |
| KMeans | ~0.2 segundos |
| DBSCAN | ~11 segundos |
| HDBSCAN | ~22 segundos |
| Autoencoder | ~50 segundos |
| **Total** | **~1.5 minutos** |

## Paso 7: Usar los notebooks

```bash
# Agregar Jupyter al entorno
poetry add --group dev jupyter

# Lanzar Jupyter
poetry run jupyter notebook notebooks/
```

Los notebooks importan desde `src/ecg_anomaly` usando `sys.path.insert(0, "../src")` en la primera celda. Ejecutar en orden:

1. `01_data_exploration.ipynb` - Explorar datos y distribucion
2. `02_preprocessing.ipynb` - Visualizar filtrado y segmentacion
3. `03_feature_extraction.ipynb` - Comparar PCA vs features manuales
4. `04_clustering.ipynb` - Ejecutar modelos y visualizar clusters
5. `05_evaluation.ipynb` - Evaluacion comparativa completa

## Comandos Poetry mas usados

```bash
poetry install              # Instalar todo (primera vez)
poetry run pytest           # Ejecutar tests
poetry run ecg-run          # Ejecutar pipeline
poetry add <paquete>        # Agregar nueva dependencia
poetry shell                # Activar el venv en la terminal actual
poetry show                 # Listar dependencias instaladas
poetry update               # Actualizar dependencias
```

## Solucion de problemas comunes

### "ModuleNotFoundError: No module named 'ecg_anomaly'"

Poetry no instalo el paquete. Ejecutar:
```bash
poetry install
```

### "No such file or directory: 'data/mitbih/100.dat'"

El symlink a los datos no funciona. Opciones:
```bash
# Opcion 1: Recrear symlink
ln -sf ../BD2-20260218T192722Z-1-001/BD2 data/mitbih

# Opcion 2: Editar config/default.yaml
# dataset_path: ./BD2-20260218T192722Z-1-001/BD2
```

### "ModuleNotFoundError: No module named 'tensorflow'"

TensorFlow no se instalo correctamente. Reinstalar:
```bash
poetry add tensorflow
```

Para ejecutar sin autoencoder mientras tanto:
```python
config.models = ['kmeans', 'dbscan', 'hdbscan']
```

### Poetry no encuentra Python 3.10+

Indicar la version de Python explicitamente:
```bash
poetry env use python3.12
poetry install
```

### "FutureWarning: The default value of `copy` will change..."

Warning cosmetico de sklearn para HDBSCAN. No afecta resultados. Se resolvera con sklearn 1.10+.

### Error de memoria con HDBSCAN o DBSCAN

Con datasets muy grandes (>200k latidos), DBSCAN y HDBSCAN pueden consumir mucha RAM. Soluciones:
```python
# Usar menos registros
dataset = loader.load(config.dataset_path, records=['100', '101', '105', '200', '208'])

# O aumentar min_cluster_size para HDBSCAN
config.hdbscan_params = {"min_cluster_size": 50, "min_samples": 20}
```

## Estructura de resultados generados

```
results/
└── report_YYYYMMDD_HHMMSS/
    ├── comparison_table.csv       # Tabla exportable a Excel
    ├── metrics_comparison.png     # Barras comparativas por metrica
    └── confusion_matrices.png     # Una por modelo
```

---

**Anterior:** [08 - Arquitectura](08_arquitectura.md) | **Indice:** [Documentacion](README.md)
