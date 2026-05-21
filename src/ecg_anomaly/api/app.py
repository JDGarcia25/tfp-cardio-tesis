"""Aplicacion FastAPI para deteccion de anomalias en latidos ECG."""

import argparse
import base64
import io
import logging
import os
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecg_anomaly.api.predictor import ModelPredictor
from ecg_anomaly.api.schemas import BeatInput, HealthOutput, PredictionOutput
from ecg_anomaly.preprocessing.filters import butterworth_bandpass
from ecg_anomaly.preprocessing.qrs_detection import pan_tompkins
from ecg_anomaly.preprocessing.segmentation import normalize_beats, segment_beats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

_predictor: ModelPredictor | None = None
_last_plot_bytes: bytes | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el mejor modelo al iniciar el servidor."""
    global _predictor
    model_dir = os.environ.get("MODEL_DIR", "./models")
    logger.info("Cargando modelo desde '%s'...", model_dir)
    _predictor = ModelPredictor(model_dir=model_dir)
    logger.info("Modelo '%s' listo para inferencia.", _predictor.model_name)
    yield
    _predictor = None


app = FastAPI(
    title="ECG Anomaly Detection API",
    description=(
        "Detecta anomalias en latidos ECG usando el mejor modelo entrenado "
        "sobre la base de datos MIT-BIH Arrhythmia. "
        "Universidad CESMAG - 2026."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


_openapi_schema: Dict[str, Any] | None = None
_openapi_orig = app.openapi


def _custom_openapi() -> Dict[str, Any]:
    global _openapi_schema
    if _openapi_schema is not None:
        return _openapi_schema
    schema = _openapi_orig()
    models = _list_models()
    if models:
        for path_data in schema.get("paths", {}).values():
            for method_data in path_data.values():
                for param in method_data.get("parameters", []):
                    if param.get("name") == "model_name":
                        param["schema"]["enum"] = models.copy()
                        param["schema"]["description"] = (
                            f"Modelo a usar. Opciones: {', '.join(models)}. "
                            "Por defecto usa el mejor modelo."
                        )
    _openapi_schema = schema
    return schema


app.openapi = _custom_openapi


PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>ECG Anomaly Detection API</title>
  <style>
    body { font-family: Arial; max-width: 900px; margin: 40px auto; padding: 0 20px; }
    h1 { color: #2c3e50; }
    h3 { color: #34495e; margin-top: 25px; }
    .endpoint { background: #e8f4f8; padding: 10px; margin: 10px 0; border-left: 4px solid #2980b9; }
    button { background: #2980b9; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 14px; }
    button:hover { background: #3498db; }
    #result { margin-top: 20px; display: none; }
    #result img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
    .summary { padding: 12px; border-radius: 5px; margin-bottom: 15px; display: flex; gap: 20px; flex-wrap: wrap; }
    .summary-item { text-align: center; }
    .summary-item .num { font-size: 24px; font-weight: bold; }
    .summary-item .label { font-size: 12px; color: #555; }
    .num-normal { color: #27ae60; }
    .num-anomalia { color: #e74c3c; }
    .num-total { color: #2980b9; }
    #loading { display: none; margin-left: 10px; color: #888; }
  </style>
</head>
<body>
  <h1>ECG Anomaly Detection API</h1>
  <p>Universidad CESMAG - 2026</p>

  <div class="endpoint">
    <strong>GET /health</strong> — Estado del servicio
  </div>
  <div class="endpoint">
    <strong>POST /predict</strong> — Clasifica un latido (JSON, 200 muestras)
  </div>
  <div class="endpoint">
    <strong>POST /predict-csv</strong> — Sube archivo CSV con la señal ECG
  </div>

  <h3>Subir archivo CSV y visualizar latidos</h3>
  <p>El CSV debe tener una columna con la señal ECG (una muestra por fila, frecuencia 360 Hz).</p>
  <form id="csvForm" enctype="multipart/form-data">
    <input type="file" id="csvFile" accept=".csv" required>
    <button type="submit">Analizar y graficar</button>
    <span id="loading">Procesando...</span>
  </form>

  <div id="result">
    <div class="summary" id="summaryContainer"></div>
    <img id="csvPlot" alt="Grafico de latidos ECG">
  </div>

  <script>
    document.getElementById('csvForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      const file = document.getElementById('csvFile').files[0];
      if (!file) { alert('Selecciona un archivo CSV'); return; }

      document.getElementById('loading').style.display = 'inline';
      document.getElementById('result').style.display = 'none';

      const formData = new FormData();
      formData.append('file', file);

      try {
        const resp = await fetch('/predict-csv', { method: 'POST', body: formData });
        const data = await resp.json();

        document.getElementById('summaryContainer').innerHTML = `
          <div class="summary-item"><div class="num num-total">${data.total_beats}</div><div class="label">Total latidos</div></div>
          <div class="summary-item"><div class="num num-normal">${data.normales}</div><div class="label">Normales</div></div>
          <div class="summary-item"><div class="num num-anomalia">${data.anomalias}</div><div class="label">Anomalias</div></div>
          <div class="summary-item"><div class="num" style="color:#555;font-size:14px;">${data.model_used}</div><div class="label">Modelo</div></div>
        `;

        if (data.csv_plot_b64) {
          document.getElementById('csvPlot').src = 'data:image/png;base64,' + data.csv_plot_b64;
        } else if (data.csv_plot) {
          document.getElementById('csvPlot').src = data.csv_plot;
        }

        document.getElementById('result').style.display = 'block';
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        document.getElementById('loading').style.display = 'none';
      }
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return PAGE


@app.get(
    "/health",
    response_model=HealthOutput,
    summary="Estado del servicio",
    tags=["sistema"],
)
async def health() -> Dict[str, Any]:
    """Verifica que el servicio esta activo y el modelo esta cargado."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    return {
        "status": "ok",
        "model_name": _predictor.model_name,
        "model_type": _predictor.model_type,
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Clasifica un latido ECG como normal o anomalia",
    tags=["prediccion"],
)
async def predict(input_data: BeatInput) -> Dict[str, Any]:
    """Recibe un latido ECG preprocesado y retorna su clasificacion.

    **Preprocesamiento requerido antes de enviar el latido:**
    1. Filtrado pasa-banda Butterworth (0.5-40 Hz, orden 4)
    2. Segmentacion: 90 muestras antes del pico R + 110 despues (200 muestras totales)
    3. Normalizacion Z-score por latido (media=0, desviacion estandar=1)

    **Respuesta:**
    - `prediction`: `0` = normal, `1` = anomalia
    - `label`: `"normal"` o `"anomalia"`
    - `reconstruction_error`: error MSE de reconstruccion (solo autoencoder)
    - `threshold`: umbral de decision del modelo
    - `model_name`: nombre del modelo utilizado
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    try:
        result = _predictor.predict(input_data.beat, preprocessed=input_data.preprocessed)
        return result
    except Exception as exc:
        logger.exception("Error durante la prediccion")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _save_plot(segments: np.ndarray, results: List[Dict[str, Any]]) -> None:
    """Genera un grid de latidos coloreados segun su prediccion y lo guarda en memoria."""
    global _last_plot_bytes
    n = len(segments)
    cols = 5
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = axes.flatten()
    for i in range(n):
        color = "#27ae60" if results[i]["prediction"] == 0 else "#e74c3c"
        axes[i].plot(segments[i], color=color, linewidth=0.8)
        axes[i].axvline(x=90, color="gray", linestyle="--", alpha=0.3)
        axes[i].set_title(f"#{i} {results[i]['label']}", fontsize=8, color=color)
        axes[i].tick_params(labelsize=6)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(
        f"Latidos detectados: {n}  |  "
        f"Normales: {sum(1 for r in results if r['prediction']==0)}  |  "
        f"Anomalias: {sum(1 for r in results if r['prediction']==1)}",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    _last_plot_bytes = buf.getvalue()


def _list_models() -> List[str]:
    """Escanea ./models en busca de subdirectorios con modelos."""
    models_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    if not models_dir.exists():
        return []
    return sorted(
        d.name for d in models_dir.iterdir()
        if d.is_dir() and (d / "scaler.joblib").exists()
    )


@app.post(
    "/predict-csv",
    summary="Sube archivo CSV con senal ECG y obtiene prediccion por latido",
    tags=["prediccion"],
)
async def predict_csv(
    file: UploadFile = File(...),
    model_name: str = Query(
        None,
        description="Modelo a usar. Si no se especifica, usa el mejor modelo.",
    ),
) -> Dict[str, Any]:
    """Sube un archivo CSV con la senal ECG (una muestra por fila, 360 Hz).

    La API aplica: filtrado -> deteccion de picos R (Pan-Tompkins) ->
    segmentacion (200 muestras por latido) -> normalizacion Z-score ->
    prediccion por latido.

    La respuesta incluye:
    - ``csv_plot``: URL del grafico PNG (``/predict-csv/plot``) para ver los latidos en el navegador
    - ``csv_plot_b64``: imagen en base64 para visualizar directamente desde Swagger UI

    Parametros:
        - **file**: Archivo CSV con la senal ECG (una muestra por fila).
        - **model_name**: (opcional) Nombre del modelo a usar.
          Si no se especifica, se usa el mejor modelo (best_model.json).
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")

    try:
        if model_name:
            _predictor.use_model(model_name)

        content = await file.read()
        signal = np.loadtxt(content.decode().splitlines(), delimiter=",")
        if signal.ndim > 1:
            signal = signal[:, 0]

        fs = 360
        filtered = butterworth_bandpass(signal, lowcut=0.5, highcut=40.0, fs=fs, order=4)
        r_peaks = pan_tompkins(filtered, fs=fs)
        segments, _ = segment_beats(filtered, r_peaks, before=90, after=110)
        segments = normalize_beats(segments)

        results: List[Dict[str, Any]] = []
        anomalias = 0
        for i in range(len(segments)):
            pred = _predictor.predict(segments[i].tolist(), preprocessed=True)
            if pred["prediction"] == 1:
                anomalias += 1
            results.append({
                "beat_index": i,
                "prediction": pred["prediction"],
                "label": pred["label"],
            })

        _save_plot(segments, results)
        plot_b64 = base64.b64encode(_last_plot_bytes).decode("utf-8") if _last_plot_bytes else None

        return {
            "model_used": _predictor.model_name,
            "model_type": _predictor.model_type,
            "total_beats": len(results),
            "anomalias": anomalias,
            "normales": len(results) - anomalias,
            "resultados": results,
            "csv_plot": "/predict-csv/plot",
            "csv_plot_b64": plot_b64,
        }
    except Exception as exc:
        logger.exception("Error procesando archivo CSV")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/predict-csv/plot",
    summary="Grafico del ultimo CSV procesado (imagen PNG)",
    tags=["prediccion"],
    include_in_schema=True,
)
async def get_csv_plot():
    """Retorna la imagen PNG del grid de latidos generado por el ultimo POST /predict-csv.

    Los latidos normales aparecen en **verde** y las anomalias en **rojo**.
    """
    if _last_plot_bytes is None:
        raise HTTPException(status_code=404, detail="No hay grafico disponible. Procesa un CSV primero.")
    return Response(content=_last_plot_bytes, media_type="image/png")


def serve() -> None:
    """Entry point CLI: ecg-serve [--model-dir DIR] [--host HOST] [--port PORT]."""
    import uvicorn  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="ECG Anomaly Detection API Server - Universidad CESMAG 2026"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Directorio con el mejor modelo guardado (default: ./models)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host del servidor (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Puerto del servidor (default: 8000)",
    )
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = args.model_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port)
