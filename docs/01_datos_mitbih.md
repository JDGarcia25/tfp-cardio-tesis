# 01 - Datos: MIT-BIH Arrhythmia Database y Agrupacion AAMI

## Por que MIT-BIH

La MIT-BIH Arrhythmia Database (Moody & Mark, 1992) es el **benchmark internacional mas usado** en investigacion de ECG. Cualquier resultado es directamente comparable con la literatura cientifica.

- **48 registros** de ~30 minutos cada uno
- **Frecuencia de muestreo:** 360 Hz
- **Dos canales:** MLII (derivacion principal) y V1/V5
- **Anotaciones manuales** por dos cardiologos independientes
- **~110,000 latidos** etiquetados con >15 tipos de arritmia

MIT-BIH es una **fortaleza**, no una debilidad. Lo que hay que evitar es pretender que usar datos de Boston equivale a estudiar la poblacion de Narino. La referencia regional se mantiene como motivacion, no como alcance.

## Estructura de los archivos

Cada registro tiene tres archivos en formato WFDB (WaveForm DataBase):

```
100.dat  →  Datos binarios de la senal ECG (amplitudes)
100.hea  →  Header: metadatos (frecuencia, duracion, canales, ganancia)
100.atr  →  Anotaciones: posicion y tipo de cada latido
```

El archivo `.atr` es clave: contiene la **posicion exacta** (en numero de muestra) de cada pico R y su **simbolo de clasificacion** (N, V, A, etc.).

### Codigo que lo implementa

```python
# src/ecg_anomaly/data/loader.py
record = wfdb.rdrecord(record_path)        # Lee .dat + .hea
annotation = wfdb.rdann(record_path, "atr") # Lee .atr

signal = record.p_signal[:, 0]             # Canal MLII
positions = annotation.sample               # Posiciones de picos R
symbols = annotation.symbol                 # Tipo de cada latido
```

## Registros excluidos

Se excluyen **4 registros** que contienen ritmos de marcapasos, siguiendo la recomendacion del estandar AAMI:

| Registro | Razon de exclusion |
|----------|-------------------|
| 102 | Ritmo de marcapasos |
| 104 | Ritmo de marcapasos |
| 107 | Ritmo de marcapasos |
| 217 | Ritmo de marcapasos |

Esto deja **44 registros validos** para el analisis.

### Codigo que lo implementa

```python
# src/ecg_anomaly/data/registry.py
PACEMAKER_RECORDS = frozenset({"102", "104", "107", "217"})

class RecordRegistry:
    def get_valid_records(self):
        return [r for r in self.ALL_RECORDS if r not in PACEMAKER_RECORDS]
```

## Agrupacion AAMI: De 15+ tipos a 2 categorias

MIT-BIH tiene mas de 15 simbolos de anotacion diferentes. Para deteccion de anomalias binaria, los agrupamos siguiendo el **estandar AAMI** (referencia clave: de Chazal, O'Dwyer & Reilly, 2004):

### Normal (etiqueta = 0)

| Simbolo | Significado |
|---------|------------|
| N | Latido normal |
| L | Bloqueo de rama izquierda |
| R | Bloqueo de rama derecha |
| e | Escape auricular |
| j | Escape nodal |

Estos latidos estan dentro de la **variabilidad normal** del ritmo cardiaco.

### Anomalo (etiqueta = 1)

| Simbolo | Significado |
|---------|------------|
| A | Contraccion auricular prematura (APC) |
| a | APC aberrante |
| J | Contraccion nodal prematura |
| S | Contraccion supraventricular prematura |
| V | Contraccion ventricular prematura (PVC) |
| E | Escape ventricular |
| F | Fusion de normal y ventricular |
| / | Ritmo de marcapasos (en registros no excluidos) |
| f | Fusion de marcapasos y normal |
| Q | No clasificable |

### Simbolos no-latido (se ignoran)

Simbolos como `+` (cambio de ritmo), `~` (ruido), `!`, `[`, `]` no representan latidos individuales y se descartan del analisis.

### Codigo que lo implementa

```python
# src/ecg_anomaly/data/registry.py
AAMI_NORMAL = frozenset({"N", "L", "R", "e", "j"})
AAMI_ANOMALOUS = frozenset({"A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"})

@staticmethod
def classify_symbol(symbol: str) -> int:
    if symbol in AAMI_NORMAL:
        return 0   # Normal
    if symbol in AAMI_ANOMALOUS:
        return 1   # Anomalo
    return -1       # No-latido (ignorar)
```

## Por que el enfoque "no supervisado" usa etiquetas para evaluar

Esta es una pregunta que cualquier jurado hara. La respuesta es sencilla y debe quedar explicita:

> **El entrenamiento se realiza sin utilizar etiquetas (no supervisado). La evaluacion requiere un criterio de referencia. Se utilizan las anotaciones de MIT-BIH como ground truth para calcular metricas extrinsecas. Este enfoque es estandar en la literatura de deteccion no supervisada de anomalias.**

Es decir:
- Los **algoritmos nunca ven** las etiquetas AAMI durante el entrenamiento
- Las etiquetas se usan **solo despues** para medir que tan bien funcionaron
- Es como un examen: el estudiante resuelve sin ver las respuestas, pero el profesor si las necesita para calificar

## Categorias de registros por patologia

Los 44 registros no son homogeneos. Algunos son predominantemente normales, otros tienen alta concentracion de arritmias:

| Categoria | Registros | Descripcion |
|-----------|-----------|-------------|
| Normal | 100, 101, 103, 112, 113, 115, 116, 117, 121, 122, 123, 220, 230, 231, 232 | Ritmo sinusal normal predominante |
| PVC/Ventricular | 105, 106, 108, 109, 119, 200, 201, 203, 205, 208, 210, 213, 214, 215, 219, 221, 228, 233, 234 | Contracciones ventriculares prematuras |
| APC/Atrial | 209, 222, 223 | Contracciones auriculares prematuras |
| Bloqueos | 111, 118, 124, 207, 212 | Bloqueos de conduccion |
| Mixto/Complejo | 114, 202 | Multiples tipos de arritmia |

## Desbalance de clases

Un aspecto critico: los latidos normales son **significativamente mas frecuentes** que los anomalos (aproximadamente 75% normal vs 25% anomalo). Este desbalance es una realidad clinica (la mayoria del tiempo el corazon late normalmente) y afecta directamente como los algoritmos asignan anomalias.

Esto NO es un problema a "resolver" con tecnicas de balanceo. Es una **caracteristica del dominio** que debemos entender y reportar.

## Archivos del codigo relevantes

| Archivo | Que contiene |
|---------|-------------|
| `src/ecg_anomaly/data/registry.py` | Catalogo de registros, constantes AAMI, clasificacion de simbolos |
| `src/ecg_anomaly/data/loader.py` | Carga de datos con wfdb, filtrado de no-latidos, etiquetado binario |
| `config/default.yaml` | Lista de registros excluidos, ruta de datos |

---

**Anterior:** [00 - Vision General](00_vision_general.md) | **Siguiente:** [02 - Preprocesamiento](02_preprocesamiento.md)
