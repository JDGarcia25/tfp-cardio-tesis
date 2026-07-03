# Documentacion del Proyecto

## Deteccion No Supervisada de Anomalias en Senales ECG

Esta documentacion explica **paso a paso** el por que y el como de cada parte del sistema. Esta pensada para que cualquier persona (jurado, companero, o tu yo del futuro) pueda entender las decisiones tomadas.

---

## Indice

### Contexto y Motivacion
- **[00 - Vision General](00_vision_general.md)** — Que es el proyecto, por que importa, que NO es, y como se estructura el enfoque de comparacion progresiva de complejidad.

### El Pipeline Paso a Paso

- **[01 - Datos MIT-BIH y Agrupacion AAMI](01_datos_mitbih.md)** — La base de datos, los 44 registros validos, por que se excluyen 4, como se agrupan los simbolos de anotacion en Normal/Anomalo, y por que usar etiquetas para evaluar no contradice el enfoque no supervisado.

- **[02 - Preprocesamiento](02_preprocesamiento.md)** — Filtrado Butterworth (0.5-40 Hz), deteccion de picos R, segmentacion en ventanas de 200 muestras (90+110), y normalizacion Z-score. Con parametros concretos y justificacion de cada uno.

- **[03 - Extraccion de Features](03_extraccion_features.md)** — Los dos caminos: Path A (senal directa + PCA al 95% de varianza) y Path B (22 features manuales: morfologicas, intervalos RR, estadisticas, frecuencia, ventanas temporales). Por que probar ambos y que pregunta responde.

- **[04 - Modelos de Deteccion](04_modelos.md)** — Los 4 niveles de complejidad: K-Means (baseline), DBSCAN (densidad), HDBSCAN (densidad jerarquica, reemplaza OPTICS), Autoencoder (deep learning). Como cada uno decide que es anomalia, sus parametros y limitaciones.

- **[05 - Evaluacion](05_evaluacion.md)** — Los 3 niveles de metricas: intrinsecas (Silhouette, DB, CH), extrinsecas (Accuracy, Sensitivity, Specificity, F1, AUC-ROC), y eficiencia (tiempo, memoria). Con la matriz de confusion explicada y el criterio de viabilidad tecnica.

### Ejecucion y Metodologia

- **[06 - Pipeline y Ejecucion](06_pipeline_y_ejecucion.md)** — Como ejecutar el sistema (CLI, Python, notebooks), la configuracion YAML, que produce como salida, y como cambiar entre representaciones.

- **[07 - Metodologia DSR](07_metodologia_dsr.md)** — Por que Design Science Research, las 6 fases del DSRM aplicadas al proyecto, las directrices de Hevner, y como DSR impacta cada seccion del documento de grado.

### Codigo y Ejecucion

- **[08 - Arquitectura del Codigo](08_arquitectura.md)** — Patrones de diseno (Factory, Strategy, Facade), mapa de dependencias entre modulos, estructura de archivos completa, y como extender el sistema.

- **[09 - Manual de Ejecucion](09_manual_ejecucion.md)** — Guia paso a paso para instalar, configurar, ejecutar el pipeline y los tests. Incluye tiempos esperados, salida real verificada, y solucion de problemas comunes.

### Referencia

- **[10 - Fundamentos de Python](10_python_fundamentos.md)** — Mini curso de Python usando ejemplos reales del proyecto: variables, imports (`from`), funciones, clases, herencia, clases abstractas, decoradores (`@dataclass`, `@property`, `@classmethod`), type hints, context managers, list comprehensions y patrones de diseno.

---

## Orden de lectura recomendado

Si es tu primera vez:

```
10 (Python) → 00 (Vision) → 01 (Datos) → 02 (Preprocesamiento) → 03 (Features)
→ 04 (Modelos) → 05 (Evaluacion) → 06 (Pipeline) → 07 (DSR) → 08 (Arquitectura)
```

Si solo quieres ejecutar el proyecto:

```
09 (Manual) → 06 (Pipeline)
```

Si eres jurado o asesor:

```
00 (Vision) → 07 (DSR) → 05 (Evaluacion) → 04 (Modelos)
```
