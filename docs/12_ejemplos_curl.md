# 12 - Ejemplos con curl

Antes de ejecutar cualquier ejemplo, el servidor debe estar corriendo:

```bash
poetry run ecg-run --config config/default.yaml   # entrena y guarda el modelo
poetry run ecg-serve                               # inicia el servidor en :8000
```

---

## Generar los archivos JSON de prueba

Los beats tienen 200 valores float. Se generan una sola vez y se reutilizan:

```bash
# Beat NORMAL sintetico — Z-score ya aplicado
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
```

```bash
# Beat ANOMALO sintetico — fibrilacion ventricular, Z-score ya aplicado
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
```

```bash
# Beat CRUDO — amplitud raw en uV, SIN Z-score (preprocessed: false)
poetry run python -c "
import numpy as np, json
t = np.arange(200)
beat = (0.25*np.exp(-((t-60)**2)/(2*8**2))
      - 0.30*np.exp(-((t-85)**2)/(2*3**2))
      + 1.50*np.exp(-((t-90)**2)/(2*4**2))
      - 0.40*np.exp(-((t-97)**2)/(2*3**2))
      + 0.40*np.exp(-((t-140)**2)/(2*15**2)))
beat = beat * 1000 + 0.5
print(json.dumps({'beat': beat.tolist(), 'preprocessed': False}))
" > /tmp/beat_crudo.json
```

```bash
# Beat sin campo preprocessed — deteccion automatica
poetry run python -c "
import numpy as np, json
t = np.arange(200)
beat = (0.25*np.exp(-((t-60)**2)/(2*8**2))
      + 1.50*np.exp(-((t-90)**2)/(2*4**2))
      + 0.40*np.exp(-((t-140)**2)/(2*15**2)))
beat = (beat - beat.mean()) / beat.std()
print(json.dumps({'beat': beat.tolist()}))
" > /tmp/beat_auto.json
```

---

## Ejemplo 1 — Verificar estado del servidor

```bash
curl http://localhost:8000/health
```

**Respuesta:**

```json
{
  "status": "ok",
  "model_name": "autoencoder",
  "model_type": "autoencoder"
}
```

---

## Ejemplo 2 — Clasificar un beat normal

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_normal.json
```

**Respuesta:**

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

## Ejemplo 3 — Clasificar un beat anomalo

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_anomalo.json
```

**Respuesta:**

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

## Ejemplo 4 — Beat crudo: el API aplica Z-score automaticamente

El beat llega en escala de ADC (µV). Como `preprocessed` es `false`,
el API normaliza antes de pasar al modelo.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_crudo.json
```

**Respuesta:**

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

`normalization_applied: true` confirma que el API aplico Z-score.

---

## Ejemplo 5 — Deteccion automatica (sin indicar `preprocessed`)

El API inspecciona la media y la desviacion estandar del beat para decidir
si ya esta normalizado.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_auto.json
```

**Respuesta:**

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

## Ejemplo 6 — Filtrar la respuesta con `jq`

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_anomalo.json \
  | jq '{resultado: .label, mse: .reconstruction_error, umbral: .threshold}'
```

**Respuesta:**

```json
{
  "resultado": "anomalia",
  "mse": 0.058741,
  "umbral": 0.021830
}
```

---

## Ejemplo 7 — Beat real de MIT-BIH (registro 100, latido #5)

```bash
# Extraer, preprocesar y guardar el latido
poetry run python -c "
import numpy as np, json, wfdb
from scipy.signal import butter, filtfilt

rec = wfdb.rdrecord('data/mitbih/100')
ann = wfdb.rdann('data/mitbih/100', 'atr')
sig = rec.p_signal[:, 0]

nyq = 0.5 * 360
b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
sig_f = filtfilt(b, a, sig)

r = ann.sample[5]
beat = sig_f[r-90 : r+110]
beat = (beat - beat.mean()) / beat.std()
print(json.dumps({'beat': beat.tolist(), 'preprocessed': True}))
" > /tmp/beat_mitbih.json

# Clasificar
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_mitbih.json
```

**Respuesta:**

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

## Ejemplo 8 — Beat de registro anomalo MIT-BIH (registro 208)

El registro 208 contiene taquicardia ventricular.

```bash
poetry run python -c "
import numpy as np, json, wfdb
from scipy.signal import butter, filtfilt

rec = wfdb.rdrecord('data/mitbih/208')
ann = wfdb.rdann('data/mitbih/208', 'atr')
sig = rec.p_signal[:, 0]

nyq = 0.5 * 360
b, a = butter(4, [0.5/nyq, 40.0/nyq], btype='band')
sig_f = filtfilt(b, a, sig)

# Buscar el primer latido anomalo (simbolo distinto de 'N')
simbolos_anomalos = [i for i, s in enumerate(ann.symbol) if s != 'N']
idx = simbolos_anomalos[0]
r = ann.sample[idx]
beat = sig_f[r-90 : r+110]
beat = (beat - beat.mean()) / beat.std()
print(json.dumps({'beat': beat.tolist(), 'preprocessed': True}))
" > /tmp/beat_anomalo_real.json

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/beat_anomalo_real.json
```

**Respuesta:**

```json
{
  "prediction": 1,
  "label": "anomalia",
  "reconstruction_error": 0.061423,
  "threshold": 0.021830,
  "model_name": "autoencoder",
  "normalization_applied": false
}
```

---

## Ejemplo 9 — Error 422: beat con longitud incorrecta

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"beat": [0.1, 0.2, 0.3]}'
```

**Respuesta:**

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

## Ejemplo 10 — Error 503: servidor sin modelo cargado

```bash
curl -s http://localhost:8000/health
```

**Respuesta (si `ecg-run` aun no se ejecuto):**

```json
{
  "detail": "Modelo no inicializado"
}
```

Solucion:

```bash
poetry run ecg-run --config config/default.yaml
```

---

**Anterior:** [11 - API de Prediccion](11_api_prediccion.md) | **Indice:** [Documentacion](README.md)
