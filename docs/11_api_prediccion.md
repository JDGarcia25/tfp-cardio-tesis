# 11 - API de Prediccion de Anomalias ECG

## ¿Que es y para que sirve?

El pipeline de entrenamiento (`ecg-run`) produce una **tabla comparativa** de los cuatro modelos y determina cual es el mejor por F1-score. Una vez identificado ese modelo, el sistema lo serializa en disco y lo expone a traves de una **API REST** construida con [FastAPI](https://fastapi.tiangolo.com/).

La API permite que cualquier aplicacion externa (sistema de monitoreo cardiaco, dashboard clinico, otro servicio) envie un latido ECG preprocesado y reciba en milisegundos si ese latido es **normal** o **anomalo**, sin necesidad de re-entrenar el modelo ni conocer los detalles internos del pipeline.

---

## Flujo completo: entrenar → servir → predecir

```
┌─────────────────────────────────────────────────────────────┐
│  Paso 1: Entrenamiento (ecg-run)                            │
│                                                             │
│  MIT-BIH → Pipeline → 4 modelos → evaluar F1               │
│                                ↓                            │
│                    mejor modelo guardado en ./models/       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Paso 2: Servidor (ecg-serve)                               │
│                                                             │
│  Lee ./models/ → carga modelo + scaler + PCA               │
│  Inicia FastAPI en http://0.0.0.0:8000                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Paso 3: Inferencia (POST /predict)                         │
│                                                             │
│  Cliente envia beat[200] → API escala → modelo predice      │
│  ← { prediction: 0|1, label: "normal"|"anomalia", ... }     │
└─────────────────────────────────────────────────────────────┘
```

---

## Paso 1: Entrenar y guardar el mejor modelo

El comando `ecg-run` ya existia para entrenar los cuatro modelos. Ahora, al finalizar la evaluacion, **automaticamente guarda el mejor modelo** (por F1-score) en el directorio `./models/`:

```bash
poetry run ecg-run --config config/default.yaml
```

Al terminar, el log mostrara:

```
[5/5] Guardando mejor modelo y generando reporte...
Mejor modelo 'autoencoder' (F1) guardado en: models/autoencoder
```

### Estructura del directorio `./models/`

```
models/
├── best_model.json          # Metadatos: nombre, tipo, representacion, metrica
└── autoencoder/             # (o kmeans/, dbscan/, hdbscan/)
    ├── model.h5             # Modelo Keras serializado (solo autoencoder)
    ├── detector.joblib      # Objeto detector sklearn (solo clustering)
    ├── scaler.joblib        # StandardScaler entrenado sobre el dataset
    ├── pca.joblib           # PCA entrenado (solo si usas signal_pca + clustering)
    └── config.json          # Umbral de anomalia o cluster mayoritario
```

#### Contenido de `best_model.json`

```json
{
  "model_name": "autoencoder",
  "model_type": "autoencoder",
  "representation": "signal_pca",
  "metric": "extrinsic_f1"
}
```

#### Contenido de `autoencoder/config.json`

```json
{
  "threshold": 0.02183,
  "representation": "signal_pca"
}
```

> **¿Por que guardar el scaler?**
> Durante el entrenamiento, el `StandardScaler` aprende la media y desviacion
> estandar de los 200 features de *todo el dataset* (~100 k latidos). Para que
> la inferencia sea consistente, cada nuevo latido debe escalarse con esos mismos
> parametros aprendidos, no con los suyos propios.

---

## Paso 2: Iniciar el servidor

```bash
poetry run ecg-serve
```

Con opciones personalizadas:

```bash
poetry run ecg-serve --model-dir ./models --host 0.0.0.0 --port 8000
```

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `--model-dir` | `./models` | Directorio donde se guardo el modelo |
| `--host` | `0.0.0.0` | IP en la que escucha el servidor |
| `--port` | `8000` | Puerto del servidor |

**Salida esperada al iniciar:**

```
INFO:     Cargando modelo desde './models'...
INFO:     Modelo 'autoencoder' listo para inferencia.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Paso 3: Usar los endpoints

### `GET /health` — Estado del servicio

Verifica que el servidor esta activo y el modelo esta cargado.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response `200 OK`:**
```json
{
  "status": "ok",
  "model_name": "autoencoder",
  "model_type": "autoencoder"
}
```

---

### `POST /predict` — Clasificar un latido ECG

Recibe un latido ECG y retorna si es normal o anomalo. El API puede detectar
y aplicar la normalizacion Z-score automaticamente si el beat aun no la tiene.

#### Campos del request

| Campo | Tipo | Requerido | Descripcion |
|-------|------|-----------|-------------|
| `beat` | `float[200]` | Si | 200 muestras del latido ECG segmentado |
| `preprocessed` | `bool \| null` | No (default `null`) | Estado de normalizacion del beat |

El campo `preprocessed` controla como el API maneja la normalizacion Z-score:

| Valor | Comportamiento |
|-------|---------------|
| `null` (default) | **Deteccion automatica**: si `\|media\| < 0.5` y `0.5 < std < 1.5`, se asume normalizado; de lo contrario, el API aplica Z-score |
| `true` | Beat ya normalizado; el API no modifica nada |
| `false` | Beat sin normalizar; el API aplica Z-score siempre |

**Request — beat ya normalizado (comportamiento anterior):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"beat": [0.12, -0.45, 0.87, ...], "preprocessed": true}'
```

**Request — beat crudo (el API normaliza automaticamente):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"beat": [1.23, 0.98, 1.45, ...], "preprocessed": false}'
```

**Request — deteccion automatica (default):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"beat": [0.12, -0.45, 0.87, ...]}'
```

**Response `200 OK` — latido normal (beat ya estaba normalizado):**
```json
{
  "prediction": 0,
  "label": "normal",
  "reconstruction_error": 0.00412,
  "threshold": 0.02183,
  "model_name": "autoencoder",
  "normalization_applied": false
}
```

**Response `200 OK` — latido anomalo (API aplico Z-score al beat crudo):**
```json
{
  "prediction": 1,
  "label": "anomalia",
  "reconstruction_error": 0.05871,
  "threshold": 0.02183,
  "model_name": "autoencoder",
  "normalization_applied": true
}
```

#### Campos de la respuesta

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `prediction` | `int` | `0` = normal, `1` = anomalia |
| `label` | `str` | `"normal"` o `"anomalia"` |
| `reconstruction_error` | `float \| null` | Error MSE de reconstruccion (solo autoencoder) |
| `threshold` | `float \| null` | Umbral de decision del modelo |
| `model_name` | `str` | Nombre del modelo utilizado |
| `normalization_applied` | `bool` | `true` si el API aplico Z-score al beat recibido |

> `reconstruction_error` y `threshold` son `null` cuando el mejor modelo es
> un algoritmo de clustering (kmeans, dbscan, hdbscan), ya que en ese caso la
> decision se basa en asignacion de cluster, no en error de reconstruccion.

---

## Documentacion interactiva

FastAPI genera automaticamente una interfaz web para explorar y probar la API:

| Interfaz | URL |
|----------|-----|
| Swagger UI (interactiva) | http://localhost:8000/docs |
| ReDoc (solo lectura) | http://localhost:8000/redoc |

Desde Swagger UI se puede enviar una peticion de prueba directamente en el navegador sin necesidad de `curl` ni codigo.

---

## Ejemplos con curl

Los beats tienen 200 valores float, por lo que se usa la forma `-d @archivo.json`.
El primer bloque genera los archivos de prueba; los siguientes muestran cada caso.

---

### Preparar archivos JSON de prueba

```bash
# Beat normal sintetico (Z-score aplicado)
poetry run python -c "
import numpy as np, json
t = np.arange(200)
beat = (0.25*np.exp(-((t-60)**2)/(2*8**2))
      - 0.30*np.exp(-((t-85)**2)/(2*3**2))
      + 1.50*np.exp(-((t-90)**2)/(2*4**2))
      - 0.40*np.exp(-((t-97)**2)/(2*3**2))
      + 0.40*np.exp(-((t-140)**2)/(2*15**2))
      + np.random.default_rng(0).normal(0, 0.03, 200))
beat = (beat - beat.mean()) / beat.std()
print(json.dumps({'beat': beat.tolist(), 'preprocessed': True}))
" > /tmp/beat_normal.json

# Beat anomalo sintetico — fibrilacion ventricular (Z-score aplicado)
poetry run python -c "
import numpy as np, json
t = np.arange(200)
rng = np.random.default_rng(42)
beat = (0.5*np.sin(2*np.pi*5*t/200)
      + 0.3*np.sin(2*np.pi*13*t/200)
      + 0.4*np.sin(2*np.pi*7*t/200)
      + rng.normal(0, 0.2, 200))
beat = (beat - beat.mean()) / beat.std()
print(json.dumps({'beat': beat.tolist(), 'preprocessed': True}))
" > /tmp/beat_anomalo.json

# Beat crudo — amplitud raw en uV, SIN Z-score (preprocessed=false)
poetry run python -c "
import numpy as np, json
t = np.arange(200)
beat = (0.25*np.exp(-((t-60)**2)/(2*8**2))
      - 0.30*np.exp(-((t-85)**2)/(2*3**2))
      + 1.50*np.exp(-((t-90)**2)/(2*4**2))
      - 0.40*np.exp(-((t-97)**2)/(2*3**2))
      + 0.40*np.exp(-((t-140)**2)/(2*15**2)))
beat = beat * 1000 + 0.5          # simular ADC crudo en uV
print(json.dumps({'beat': beat.tolist(), 'preprocessed': False}))
" > /tmp/beat_crudo.json

# Beat sin indicar preprocessed — deteccion automatica
poetry run python -c "
import numpy as np, json
t = np.arange(200)
beat = (0.25*np.exp(-((t-60)**2)/(2*8**2))
      + 1.50*np.exp(-((t-90)**2)/(2*4**2))
      + 0.40*np.exp(-((t-140)**2)/(2*15**2)))
beat = (beat - beat.mean()) / beat.std()
print(json.dumps({'beat': beat.tolist()}))   # sin preprocessed
" > /tmp/beat_auto.json
```

---

### 1. Verificar estado del servidor

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_name": "autoencoder",
  "model_type": "autoencoder"
}
```

---

### 2. Clasificar un beat normal (`preprocessed: true`)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_normal.json
```

```json
{
  "prediction": 0,
  "label": "normal",
  "reconstruction_error": 0.003241,
  "threshold": 0.021830,
  "model_name": "autoencoder",
  "normalization_applied": false
}
```

---

### 3. Clasificar un beat anomalo (`preprocessed: true`)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_anomalo.json
```

```json
{
  "prediction": 1,
  "label": "anomalia",
  "reconstruction_error": 0.058741,
  "threshold": 0.021830,
  "model_name": "autoencoder",
  "normalization_applied": false
}
```

---

### 4. Enviar beat crudo — el API aplica Z-score (`preprocessed: false`)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_crudo.json
```

```json
{
  "prediction": 0,
  "label": "normal",
  "reconstruction_error": 0.003241,
  "threshold": 0.021830,
  "model_name": "autoencoder",
  "normalization_applied": true
}
```

> Notese que `normalization_applied` es `true`: el API detecto que el beat
> estaba en escala de ADC y le aplico Z-score antes de pasarlo al modelo.

---

### 5. Deteccion automatica (sin campo `preprocessed`)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_auto.json
```

```json
{
  "prediction": 0,
  "label": "normal",
  "reconstruction_error": 0.004102,
  "threshold": 0.021830,
  "model_name": "autoencoder",
  "normalization_applied": false
}
```

---

### 6. Ver la respuesta formateada con `jq`

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_anomalo.json | jq '{resultado: .label, mse: .reconstruction_error, umbral: .threshold}'
```

```json
{
  "resultado": "anomalia",
  "mse": 0.058741,
  "umbral": 0.021830
}
```

---

### 7. Clasificar un beat de un registro MIT-BIH real

```bash
# Extraer el latido #5 del registro 100 y guardarlo como JSON
poetry run python -c "
import numpy as np, json, wfdb
from scipy.signal import butter, filtfilt

rec = wfdb.rdrecord('data/mitbih/100')
ann = wfdb.rdann('data/mitbih/100', 'atr')
sig = rec.p_signal[:, 0]

# Filtro Butterworth pasa-banda (mismo que el pipeline)
nyq = 0.5 * 360
b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
sig_f = filtfilt(b, a, sig)

# Segmentar latido #5 centrado en el pico R
r = ann.sample[5]
beat = sig_f[r-90 : r+110]

# Z-score por latido
beat = (beat - beat.mean()) / beat.std()
print(json.dumps({'beat': beat.tolist(), 'preprocessed': True}))
" > /tmp/beat_mitbih_100_5.json

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_mitbih_100_5.json
```

```json
{
  "prediction": 0,
  "label": "normal",
  "reconstruction_error": 0.002817,
  "threshold": 0.021830,
  "model_name": "autoencoder",
  "normalization_applied": false
}
```

---

### 8. Error de validacion — beat con longitud incorrecta

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"beat": [0.1, 0.2, 0.3]}'
```

```json
{
  "detail": [
    {
      "type": "too_short",
      "loc": ["body", "beat"],
      "msg": "List should have at least 200 items after validation, not 3",
      "input": [0.1, 0.2, 0.3]
    }
  ]
}
```

---

## Que debe cumplir el latido de entrada

El campo `beat` debe contener exactamente **200 valores `float`** que representen
un latido ECG. Los pasos 1 y 2 (filtrado y segmentacion) son responsabilidad del
cliente; el paso 3 (Z-score) puede delegarse al API usando el campo `preprocessed`.

| Paso | Responsable | Descripcion | Parametro |
|------|-------------|-------------|-----------|
| **1. Filtrado** | Cliente | Butterworth pasa-banda | 0.5–40 Hz, orden 4 |
| **2. Segmentacion** | Cliente | Ventana centrada en el pico R | 90 muestras antes + 110 despues |
| **3. Normalizacion Z-score** | Cliente o API | media=0, desv. est.=1 por latido | `preprocessed` en el request |

> **¿Por que el filtrado y la segmentacion no los hace el API?**
> Esos dos pasos requieren la **senal ECG continua completa** y la posicion
> del pico R, informacion que no esta disponible en un endpoint que recibe
> solo 200 muestras. La normalizacion Z-score si puede aplicarse sobre el
> segmento aislado, por eso es el unico paso que el API puede delegar.

### Heuristica de deteccion automatica (`preprocessed=null`)

Cuando `preprocessed` no se envia (o se envia `null`), el predictor llama a
`ModelPredictor.is_zscore_normalized()` con las siguientes condiciones:

```
|media(beat)| < 0.5   AND   0.5 < std(beat) < 1.5
         ↓ cumple               ↓ no cumple
  beat ya normalizado     API aplica Z-score
```

Esta heuristica funciona correctamente en la gran mayoria de casos, pero puede
fallar en beats anormalmente planos (std ≈ 0) o con amplitud raw dentro del
rango estadistico de un beat normalizado. Para produccion con datos heterogeneos,
se recomienda indicar `preprocessed` explicitamente.

---

## Pipeline interno de inferencia

Cuando llega una peticion a `POST /predict`, el predictor ejecuta internamente:

### Si el mejor modelo es el Autoencoder

```
beat[200]  (raw o Z-score, segun 'preprocessed')
    ↓
¿preprocessed? ──────────────────────────────────────
  null → is_zscore_normalized()  →  ya normalizado?
  true  → skip                      si: continuar
  false → aplicar Z-score           no: Z-score per beat
    ↓
beat[200]  (Z-score garantizado)
    ↓
StandardScaler.transform()     ← scaler entrenado en ~100k latidos
    ↓
scaled[200]
    ↓
Autoencoder.predict()          ← red neuronal encoder-decoder
    ↓
reconstructed[200]
    ↓
error_mse = mean((scaled - reconstructed)²)
    ↓
prediction = 1 si error_mse > threshold else 0
```

### Si el mejor modelo es un algoritmo de Clustering

```
beat[200]  (raw o Z-score, segun 'preprocessed')
    ↓
¿preprocessed? → deteccion/normalizacion Z-score (mismo flujo que arriba)
    ↓
beat[200]  (Z-score garantizado)
    ↓
StandardScaler.transform()     ← scaler entrenado en ~100k latidos
    ↓
scaled[200]
    ↓
PCA.transform()                ← PCA con 95% varianza (~12 componentes)
    ↓
features[k]
    ↓
detector.predict_anomalies()   ← KMeans / DBSCAN / HDBSCAN
    ↓
prediction = 0 (normal) o 1 (anomalia)
```

---

## Estructura del modulo `ecg_anomaly.api`

```
src/ecg_anomaly/api/
├── __init__.py          # Inicializacion del paquete
├── schemas.py           # Modelos Pydantic: BeatInput, PredictionOutput, HealthOutput
├── predictor.py         # ModelPredictor: carga modelo + ejecuta inferencia
└── app.py               # Aplicacion FastAPI + funcion serve() (CLI ecg-serve)
```

### `schemas.py` — Contrato de entrada y salida

Define con [Pydantic](https://docs.pydantic.dev/) los tipos exactos que acepta
y retorna la API. Pydantic valida automaticamente que:
- `beat` tenga exactamente 200 elementos
- Todos sean convertibles a `float`
- Los campos de respuesta tengan el tipo correcto

Si la validacion falla, la API retorna `422 Unprocessable Entity` con detalle
del error antes de llegar al modelo.

### `predictor.py` — Logica de inferencia

`ModelPredictor` encapsula toda la logica de carga y ejecucion:

```python
class ModelPredictor:
    def __init__(self, model_dir: str = "./models"):
        self._load()   # Lee best_model.json y carga artefactos

    def predict(self, beat: List[float]) -> Dict:
        # 1. Escala el beat con el scaler entrenado
        # 2. Aplica PCA si es modelo de clustering
        # 3. Ejecuta prediccion y retorna dict
```

### `app.py` — Aplicacion FastAPI

- **`lifespan`**: contexto de inicio/cierre del servidor. Carga el modelo
  una sola vez al arrancar (no en cada peticion).
- **`GET /health`**: verifica que el modelo esta listo.
- **`POST /predict`**: valida el input, llama al predictor, retorna el resultado.
- **`serve()`**: funcion invocada por el CLI `ecg-serve`.

---

## Errores comunes

### `FileNotFoundError: No se encontro el archivo de metadatos en 'models/best_model.json'`

El modelo no ha sido entrenado y guardado aun. Ejecutar primero:
```bash
poetry run ecg-run --config config/default.yaml
```

### `422 Unprocessable Entity`

El campo `beat` no tiene exactamente 200 elementos, o contiene valores no numericos. Verificar la longitud del array enviado.

### `503 Service Unavailable`

El servidor arranco pero el modelo no cargo correctamente. Revisar los logs del servidor y verificar que `./models/best_model.json` existe y es valido.

### El servidor tarda en responder la primera peticion (autoencoder)

TensorFlow realiza compilacion JIT en la primera llamada a `predict()`. Las peticiones subsiguientes son significativamente mas rapidas.

---

## Dependencias agregadas

| Libreria | Version | Rol |
|----------|---------|-----|
| `fastapi` | ^0.111 | Framework web para construir la API REST |
| `uvicorn[standard]` | ^0.29 | Servidor ASGI de alto rendimiento para FastAPI |

Estas se instalan automaticamente con:
```bash
poetry install
```

---

**Anterior:** [10 - Fundamentos de Python](10_python_fundamentos.md) | **Indice:** [Documentacion](README.md)
