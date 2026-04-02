# 02 - Preprocesamiento de Senales ECG

## Por que preprocesar

Una senal ECG cruda contiene **ruido** que dificulta el analisis:

- **Deriva de linea base** (< 0.5 Hz): Causada por la respiracion y movimiento del paciente. La senal "sube y baja" lentamente.
- **Interferencia de linea electrica** (50/60 Hz): Ruido de la red electrica captado por los electrodos.
- **Ruido muscular** (> 40 Hz): Contracciones musculares del paciente.

El preprocesamiento elimina estas fuentes de ruido **preservando las componentes clinicamente relevantes** del ECG (complejo QRS, ondas P y T), que estan entre 0.5 y 40 Hz.

## Paso 1: Filtrado pasa-banda Butterworth

### Que hace
Aplica un filtro que **solo deja pasar** frecuencias entre 0.5 Hz y 40 Hz. Todo lo que esta fuera de ese rango se atenua.

### Por que Butterworth
- Tiene una **respuesta de frecuencia maximamente plana** en la banda de paso (no distorsiona las frecuencias que queremos conservar)
- Es el filtro mas usado en la literatura de ECG
- Orden 4: buen compromiso entre atenuacion y distorsion de fase

### Parametros concretos

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| Frecuencia de corte inferior | 0.5 Hz | Elimina deriva de linea base |
| Frecuencia de corte superior | 40 Hz | Elimina ruido muscular y de linea |
| Orden del filtro | 4 | Balance entre atenuacion y fase |
| Tipo | Butterworth | Maximamente plano en banda |

### Implementacion

```python
# src/ecg_anomaly/preprocessing/filters.py
from scipy.signal import butter, filtfilt

def butterworth_bandpass(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)  # filtfilt: zero-phase (no desfasa)
```

**Nota:** Usamos `filtfilt` (filtrado bidireccional) en vez de `lfilter` porque `filtfilt` tiene **fase cero**: no desfasa la senal en el tiempo. Esto es critico para que los picos R mantengan su posicion exacta.

### Efecto visual

```
Senal cruda:     ∼∼∼/\∼∼∼∼∼∼/\∼∼∼∼  (ruido visible, linea base inestable)
                    ↓ Butterworth 0.5-40 Hz
Senal filtrada:  ___/\________/\____  (QRS limpio, linea base estable)
```

## Paso 2: Posiciones de picos R

### Dos opciones

Para MIT-BIH, **no necesitamos detectar picos R**: las anotaciones ya incluyen la posicion exacta de cada latido (archivo `.atr`). Estas posiciones fueron marcadas manualmente por cardiologos.

Sin embargo, el proyecto incluye dos algoritmos de deteccion para uso con senales sin anotaciones:

#### Opcion A: Anotaciones MIT-BIH (la que usamos)
```python
annotation = wfdb.rdann(record_path, "atr")
r_peaks = annotation.sample  # Posiciones exactas del cardiologo
```

#### Opcion B: Pan-Tompkins (para senales sin anotaciones)
El algoritmo Pan-Tompkins (1985) es el metodo clasico de deteccion QRS:

1. **Filtro pasa-banda 5-15 Hz**: Resalta el complejo QRS
2. **Derivada**: Detecta pendientes empinadas (tipicas del QRS)
3. **Cuadrado**: Amplifica las diferencias y hace todo positivo
4. **Integracion por ventana movil**: Suaviza para obtener envolvente
5. **Deteccion de picos**: Busca maximos sobre un umbral adaptativo
6. **Refinamiento**: Ajusta al maximo real de la senal original

```python
# src/ecg_anomaly/preprocessing/qrs_detection.py
def pan_tompkins(signal, fs=360):
    filtered = bandpass_5_15hz(signal)
    diff = np.diff(filtered)
    squared = diff ** 2
    integrated = moving_average(squared, window=0.150*fs)
    peaks = find_peaks(integrated, distance=0.2*fs)
    return refine_peaks(signal, peaks)  # Buscar maximo real
```

## Paso 3: Segmentacion de latidos

### Que hace
Corta la senal continua en **ventanas individuales** centradas en cada pico R. Cada ventana es un "latido aislado" listo para analizar.

### Parametros de la ventana

| Parametro | Valor | Equivalente temporal |
|-----------|-------|---------------------|
| Muestras antes del pico R | 90 | ~250 ms |
| Muestras despues del pico R | 110 | ~305 ms |
| **Total por latido** | **200 muestras** | **~555 ms** |

### Por que 90 + 110 = 200 muestras

- **90 muestras antes:** Captura la onda P (despolarizacion auricular) y el inicio del complejo QRS
- **110 muestras despues:** Captura el segmento ST, la onda T (repolarizacion ventricular) y parte del intervalo T-P
- **200 total:** Vector de tamano fijo necesario para que los algoritmos procesen todos los latidos de manera uniforme

### Manejo de bordes
Los latidos muy cerca del inicio o final de la senal (donde la ventana no cabe completa) se **descartan**. Esto es correcto: preferimos perder unos pocos latidos a tener segmentos incompletos.

```python
# src/ecg_anomaly/preprocessing/segmentation.py
def segment_beats(signal, r_peaks, before=90, after=110):
    segments = []
    valid_indices = []
    for i, peak in enumerate(r_peaks):
        start = peak - before
        end = peak + after
        if start < 0 or end > len(signal):
            continue  # Descartamos latidos en los bordes
        segments.append(signal[start:end])
        valid_indices.append(i)
    return np.array(segments), np.array(valid_indices)
```

### Diagrama de segmentacion

```
Senal continua:
    ... ___/\________/\________/\________/\_____ ...
           R1          R2          R3          R4

Segmentacion (90 antes + 110 despues):
    [----90---|---110----]  →  Latido 1 (200 muestras)
              [----90---|---110----]  →  Latido 2
                        [----90---|---110----]  →  Latido 3
```

## Paso 4: Normalizacion Z-score

### Que hace
Cada latido se normaliza **independientemente** para tener media 0 y desviacion estandar 1.

### Por que normalizar por latido

- **Diferentes amplitudes:** La amplitud del ECG varia entre pacientes, entre registros, e incluso entre latidos del mismo registro (por respiracion, movimiento de electrodos).
- **Los algoritmos comparan forma, no amplitud:** Nos interesa la **morfologia** del latido (su forma), no su amplitud absoluta.
- **Hace comparables** latidos de diferentes registros y pacientes.

### Formula

```
latido_normalizado = (latido - media_del_latido) / desviacion_estandar_del_latido
```

### Implementacion

```python
# src/ecg_anomaly/preprocessing/segmentation.py
def normalize_beats(segments):
    means = segments.mean(axis=1, keepdims=True)
    stds = segments.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-10, 1.0, stds)  # Evitar division por cero
    return (segments - means) / stds
```

### Efecto visual

```
Antes:  Latido A tiene amplitud 2.5 mV, Latido B tiene 0.8 mV
        (parecen muy diferentes por escala)

Despues: Ambos tienen media=0, std=1
         (ahora se compara solo la FORMA)
```

## Pipeline completo integrado

El pipeline encadena los 4 pasos automaticamente:

```python
# src/ecg_anomaly/preprocessing/pipeline.py
class PreprocessingPipeline:
    def run(self, dataset):
        for record in dataset.records:
            # Paso 1: Filtrar senal continua
            filtered = butterworth_bandpass(record.signal, 0.5, 40, 360, order=4)

            # Paso 2: Usar posiciones R de las anotaciones
            r_peaks = record.r_peak_positions

            # Paso 3: Segmentar latidos (200 muestras cada uno)
            segments, valid_idx = segment_beats(filtered, r_peaks, 90, 110)

            # Paso 4: Normalizar cada latido
            segments = normalize_beats(segments)

            # Filtrar etiquetas correspondientes
            labels = record.binary_labels[valid_idx]
```

### Resultado del preprocesamiento

| Dato | Forma | Descripcion |
|------|-------|-------------|
| `segments` | [N, 200] | N latidos de 200 muestras cada uno, normalizados |
| `labels` | [N] | 0=normal, 1=anomalo (de AAMI) |
| `r_peaks_flat` | [N] | Posiciones R originales (para calcular intervalos RR) |

## Archivos del codigo relevantes

| Archivo | Que contiene |
|---------|-------------|
| `src/ecg_anomaly/preprocessing/filters.py` | Filtro Butterworth pasa-banda y notch |
| `src/ecg_anomaly/preprocessing/qrs_detection.py` | Pan-Tompkins y XQRS (wfdb) |
| `src/ecg_anomaly/preprocessing/segmentation.py` | Segmentacion y normalizacion Z-score |
| `src/ecg_anomaly/preprocessing/pipeline.py` | Pipeline integrado |

---

**Anterior:** [01 - Datos MIT-BIH](01_datos_mitbih.md) | **Siguiente:** [03 - Extraccion de Features](03_extraccion_features.md)
