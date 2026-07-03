# ECG Anomaly Detection System

**Evaluacion comparativa de metodos de clustering no supervisado y autoencoder para la deteccion de anomalias en senales electrocardiograficas**

Universidad CESMAG - Ingenieria de Sistemas, 2026

Autores: Garcia Alvarez, Elian - Garcia Zambrano, Juan David  
Asesor: Mg. Mora Paz, Hector Andres

---

## Descripcion

Sistema de deteccion de anomalias en senales ECG que compara cuatro metodos de complejidad creciente bajo las mismas condiciones de preprocesamiento, usando la base de datos estandar MIT-BIH Arrhythmia Database.

**Pregunta de investigacion:** ¿Cual es el rendimiento comparativo de los metodos de clustering no supervisado (K-Means, DBSCAN, HDBSCAN) frente a un autoencoder para la deteccion de anomalias en senales ECG de MIT-BIH, y cual ofrece la mejor relacion rendimiento-costo computacional?

### Niveles de Complejidad

| Nivel | Metodo | Tipo | Rol |
|-------|--------|------|-----|
| 1 | K-Means | Clustering particional | Baseline - establece piso de rendimiento |
| 2 | DBSCAN | Clustering por densidad | Identifica ruido como anomalia |
| 3 | HDBSCAN | Densidad jerarquica | Seleccion automatica de parametros |
| 4 | Autoencoder | Deep Learning no supervisado | Error de reconstruccion como anomalia |

### Representaciones de Datos

- **Path A (signal_pca):** Senal directa (200 muestras) + PCA (95% varianza)
- **Path B (manual_features):** 22 features manuales (morfologicas, RR, estadisticas, frecuencia, ventanas temporales de 5/10 latidos)

### Evaluacion en Tres Niveles

| Tipo | Metricas | Requiere etiquetas |
|------|----------|--------------------|
| Intrinsecas | Silhouette, Davies-Bouldin, Calinski-Harabasz | No |
| Extrinsecas | Accuracy, Sensitivity, Specificity, F1, AUC-ROC | Si (ground truth AAMI) |
| Eficiencia | Tiempo de ejecucion, memoria pico | No |

---

## Estructura del Proyecto

```
tfp-cardio/
├── config/default.yaml          # Configuracion del pipeline
├── src/ecg_anomaly/
│   ├── config.py                # Configuracion centralizada
│   ├── data/                    # Carga de datos y registro MIT-BIH
│   ├── preprocessing/           # Filtrado, QRS, segmentacion
│   ├── features/                # Extraccion de features (2 paths)
│   ├── models/                  # 4 detectores de anomalias
│   ├── evaluation/              # Metricas intrinsecas/extrinsecas/eficiencia
│   ├── visualization/           # Graficos y reportes
│   └── pipeline.py              # Orquestador principal (Facade)
├── notebooks/                   # 5 notebooks exploratorios
├── tests/                       # Tests unitarios
└── results/                     # Resultados generados
```

---

## Instalacion

### Requisitos Previos

- Python >= 3.10
- [Poetry](https://python-poetry.org/docs/#installation)

### Pasos

```bash
# Clonar repositorio
git clone <repo-url>
cd tfp-cardio

# Instalar dependencias
poetry install

# Verificar instalacion
poetry run pytest
```

### Datos MIT-BIH

Los archivos de MIT-BIH (.dat, .hea, .atr) deben estar en `BD2-20260218T192722Z-1-001/BD2/` (ruta configurada en `dataset_path` de `config/default.yaml`). Descargalos desde [PhysioNet](https://physionet.org/content/mitdb/) — no estan versionados en git (ver `.gitignore`), asi que hay que colocarlos manualmente tras clonar el repo.

---

## Uso

### Pipeline Completo (CLI)

```bash
# Con configuracion por defecto (4 modelos, signal+PCA)
poetry run ecg-run

# Con representacion manual de features
poetry run ecg-run --representation manual_features

# Con configuracion personalizada
poetry run ecg-run --config config/custom.yaml
```

### Uso Programatico

```python
from ecg_anomaly.config import SystemConfig
from ecg_anomaly.pipeline import ECGAnomalyPipeline

config = SystemConfig.from_yaml("config/default.yaml")
pipeline = ECGAnomalyPipeline(config)
results = pipeline.run()

print(results.to_string())
```

### Notebooks

Los notebooks en `notebooks/` documentan cada fase del pipeline DSR:

1. `01_data_exploration.ipynb` - Exploracion de datos y distribucion AAMI
2. `02_preprocessing.ipynb` - Filtrado, segmentacion, normalizacion
3. `03_feature_extraction.ipynb` - PCA y features manuales
4. `04_clustering.ipynb` - Ejecucion de los 4 modelos
5. `05_evaluation.ipynb` - Evaluacion comparativa final

---

## Pipeline Tecnico

```
MIT-BIH ECG (.dat/.hea/.atr)
    |
    v
Carga (wfdb) + Agrupacion AAMI (Normal/Anomalo)
    |
    v
Filtrado Butterworth pasa-banda (0.5-40 Hz, orden 4)
    |
    v
Segmentacion (90 muestras antes R + 110 despues = 200 muestras/latido)
    |
    v
Normalizacion Z-score por latido
    |
    +---> Path A: StandardScaler + PCA (95% varianza)
    |         |
    |         +---> K-Means / DBSCAN / HDBSCAN (datos PCA)
    |         +---> Autoencoder (datos escalados, sin PCA)
    |
    +---> Path B: 12 features manuales
              |
              +---> K-Means / DBSCAN / HDBSCAN / Autoencoder
    |
    v
Evaluacion: Intrinsecas + Extrinsecas (ground truth) + Eficiencia
    |
    v
Tabla Comparativa: 4 modelos x metricas
```

---

## Metodologia

El proyecto sigue **Design Science Research Methodology (DSRM)** (Peffers et al., 2007):

1. **Identificacion del problema:** Dificultad para detectar arritmias sin datos etiquetados
2. **Objetivos de la solucion:** Pipeline comparativo de 4 metodos no supervisados
3. **Diseno y desarrollo:** Implementacion del pipeline en Python
4. **Demostracion:** Ejecucion sobre MIT-BIH (44 registros)
5. **Evaluacion:** Metricas intrinsecas, extrinsecas y de eficiencia
6. **Comunicacion:** Documento de trabajo de grado

---

## Interpretacion Clinica de Resultados

### Niveles de Alerta Clinica

El frontend incluye un sistema de **semáforo clínico** que clasifica cada latido en cuatro niveles de alerta:

| Nivel | Color | Criterio | Accion Recomendada |
|-------|-------|----------|--------------------|
| **Normal** | 🟢 Verde | Sin anomalia detectada | Continuar monitoreo |
| **Alerta Leve** | 🟡 Amarillo | Error de reconstruccion > umbral (ratio < 2×) | Revisar en siguiente ciclo |
| **Alerta Moderada** | 🟠 Naranja | Error de reconstruccion 2-3× el umbral | Evaluacion prioritaria |
| **Critico** | 🔴 Rojo | Error de reconstruccion > 3× el umbral | Intervencion inmediata |

Los umbrales de alerta se basan en el **ratio error/umbral** del autoencoder. En modo CSV, el nivel de alerta global depende del porcentaje de anomalias detectadas en el registro completo.

### Metricas Clave para Diagnostico

- **Sensibilidad (Recall):** La metrica mas importante en contexto clinico. Un falso negativo (anomalia no detectada) puede tener consecuencias graves. El modelo con mayor sensibilidad debe preferirse para screening.
- **Especificidad:** Relevante para evitar falsas alarmas que saturan al personal medico.
- **F1 Score:** Balance entre precision y sensibilidad. Util como metrica unificada de comparacion.
- **Multi-Criteria Ranking:** Combina todas las metricas en un puntaje compuesto para seleccion objetiva del mejor modelo.

### Limitaciones

- Los 4 metodos son **no supervisados**: no requieren etiquetas para entrenar, pero la asignacion de clases normal/anomalo es **heuristica**.
- La validacion clinica requiere un estudio con especialistas y datos prospectivos.
- El dataset MIT-BIH esta desbalanceado (~10% anomalias), lo que favorece clasificadores que predicen "normal" siempre.

---

## Tests

```bash
poetry run pytest                    # Todos los tests
poetry run pytest -v                 # Verbose
poetry run pytest tests/test_models.py  # Solo modelos
poetry run pytest --cov=ecg_anomaly  # Con cobertura
```

---

## Tecnologias

- **Python 3.10+** con Poetry para gestion de dependencias
- **NumPy, SciPy, Pandas** - Procesamiento numerico
- **scikit-learn** - KMeans, DBSCAN, HDBSCAN, PCA, metricas
- **TensorFlow/Keras** - Autoencoder
- **wfdb** - Lectura de datos MIT-BIH (PhysioNet)
- **PyWavelets** - Transformada wavelet (features opcionales)
- **Matplotlib, Seaborn, Plotly** - Visualizacion

---

## Referencias Clave

- Peffers, K. et al. (2007). *A Design Science Research Methodology.* JMIS.
- de Chazal, P. et al. (2004). *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE Trans. Biomed. Eng.
- Moody, G.B. & Mark, R.G. (2001). *The impact of the MIT-BIH Arrhythmia Database.* IEEE EMB Magazine.
