"""Aplicacion FastAPI para deteccion de anomalias en latidos ECG."""

import argparse
import base64
import io
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecg_anomaly.api.predictor import ModelPredictor
from ecg_anomaly.api.schemas import BeatInput, HealthOutput, ModelInfoOutput, PredictionOutput
from ecg_anomaly.preprocessing.filters import butterworth_bandpass
from ecg_anomaly.preprocessing.qrs_detection import pan_tompkins
from ecg_anomaly.preprocessing.segmentation import normalize_beats, segment_beats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

_predictor: Optional[ModelPredictor] = None
_last_plot_bytes: Optional[bytes] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    version="1.0.0",
    contact={
        "name": "Garcia Alvarez Elian & Garcia Zambrano Juan David",
        "email": "cesmag@edu.co",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Rate limiting simple (in-memory, por IP) ─────────────────────────────────
_rate_store: Dict[str, List[float]] = {}
_RATE_LIMIT = int(os.environ.get("RATE_LIMIT_RPM", "60"))  # requests per minute


async def _check_rate_limit(request: Request) -> None:
    import time

    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0

    if ip not in _rate_store:
        _rate_store[ip] = []

    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < window]

    if len(_rate_store[ip]) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Limite de peticiones alcanzado: {_RATE_LIMIT} req/min por IP.",
        )
    _rate_store[ip].append(now)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_models() -> List[str]:
    models_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    if not models_dir.exists():
        return []
    return sorted(
        d.name for d in models_dir.iterdir()
        if d.is_dir() and (d / "scaler.joblib").exists()
    )


def _save_plot(segments: np.ndarray, results: List[Dict[str, Any]]) -> None:
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
        f"Latidos: {n}  |  Normales: {sum(1 for r in results if r['prediction']==0)}"
        f"  |  Anomalias: {sum(1 for r in results if r['prediction']==1)}",
        fontsize=10,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    _last_plot_bytes = buf.getvalue()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return HTMLResponse(content=_LANDING_PAGE, status_code=200)


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


@app.get(
    "/model-info",
    response_model=ModelInfoOutput,
    summary="Informacion completa del modelo activo",
    tags=["sistema"],
)
async def model_info() -> Dict[str, Any]:
    """Retorna metadatos detallados del modelo cargado en memoria."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    return {
        "model_name": _predictor.model_name,
        "model_type": _predictor.model_type,
        "representation": _predictor.representation,
        "threshold": _predictor.threshold,
        "available_models": _predictor.available_models or _list_models(),
        "has_pca": _predictor.has_pca,
        "beat_length": 200,
        "sampling_rate": 360,
        "dataset": "MIT-BIH Arrhythmia Database",
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Clasifica un latido ECG como normal o anomalia",
    tags=["prediccion"],
)
async def predict(
    input_data: BeatInput,
    request: Request,
    model_name: Optional[Literal["autoencoder", "kmeans", "dbscan", "hdbscan"]] = Query(
        None,
        description="Modelo a usar. Si no se indica, usa el mejor modelo (best_model.json).",
    ),
) -> Dict[str, Any]:
    """Recibe un latido ECG preprocesado y retorna su clasificacion.

    **Preprocesamiento requerido antes de enviar el latido:**
    1. Filtrado pasa-banda Butterworth (0.5–40 Hz, orden 4)
    2. Segmentacion: 90 muestras antes del pico R + 110 despues (200 muestras totales)
    3. Normalizacion Z-score por latido (media=0, desv. est.=1) — o usar `preprocessed=false`

    **Campos de respuesta:**
    - `prediction`: `0` = normal, `1` = anomalia
    - `reconstruction_error`: error MSE (solo autoencoder; `null` para clustering)
    - `threshold`: umbral de decision (`null` para clustering)
    - `normalization_applied`: `true` si el API aplico Z-score
    """
    await _check_rate_limit(request)
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    try:
        if model_name:
            _predictor.use_model(model_name)
        result = _predictor.predict(input_data.beat, preprocessed=input_data.preprocessed)
        return result
    except Exception as exc:
        logger.exception("Error durante la prediccion")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/predict-csv",
    summary="Sube archivo CSV con senal ECG y obtiene prediccion por latido",
    tags=["prediccion"],
)
async def predict_csv(
    request: Request,
    file: UploadFile = File(...),
    model_name: Optional[Literal["autoencoder", "kmeans", "dbscan", "hdbscan"]] = Query(
        None,
        description="Modelo a usar. Si no se indica, usa el mejor modelo.",
    ),
    sensitivity: float = Query(
        1.0,
        ge=0.1,
        le=5.0,
        description=(
            "Multiplicador inverso del umbral fijo de entrenamiento (solo autoencoder). "
            "sensitivity=2.0 → umbral a la mitad. Solo actua cuando adaptive_threshold=false."
        ),
    ),
    adaptive_threshold: bool = Query(
        True,
        description=(
            "Si True (recomendado), calcula el umbral como el percentil 85 de los errores "
            "MSE del propio registro subido. Detecta anomalias relativas al registro, "
            "independientemente del umbral de entrenamiento."
        ),
    ),
    adaptive_percentile: int = Query(
        85,
        ge=50,
        le=99,
        description=(
            "Percentil para umbral adaptativo (default 85 = top 15%% anomalo). "
            "Solo aplica cuando adaptive_threshold=true."
        ),
    ),
) -> Dict[str, Any]:
    """Sube un archivo CSV con la senal ECG (una muestra por fila, 360 Hz).

    La API aplica: filtrado → deteccion de picos R (Pan-Tompkins) →
    segmentacion (200 muestras) → Z-score → prediccion por latido.

    **Umbral adaptativo (recomendado):** calcula el umbral desde el propio registro,
    marcando como anomalia el top N% de latidos con mayor error de reconstruccion.
    Esto es mas robusto que el umbral fijo de entrenamiento para registros nuevos.

    **Umbral fijo + sensitivity:** usa el umbral de entrenamiento dividido por
    el factor de sensibilidad. Util para comparar con el modelo original.
    """
    await _check_rate_limit(request)
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")

    try:
        if model_name:
            _predictor.use_model(model_name)

        # Umbral fijo de entrenamiento (solo autoencoder)
        original_threshold = _predictor.threshold
        fixed_threshold = None
        if original_threshold is not None and not adaptive_threshold:
            fixed_threshold = original_threshold / sensitivity
            _predictor._threshold = fixed_threshold
        elif original_threshold is not None:
            fixed_threshold = original_threshold

        # Cargar y parsear CSV (soporta header y multiples columnas)
        content = await file.read()
        lines = [l for l in content.decode(errors="replace").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        try:
            signal = np.loadtxt(lines, delimiter=",")
        except ValueError:
            signal = np.loadtxt(lines[1:], delimiter=",")
        if signal.ndim > 1:
            signal = signal[:, 0]

        fs = 360
        logger.info("CSV cargado: %d muestras (~%.1f s a %d Hz)", len(signal), len(signal)/fs, fs)

        filtered = butterworth_bandpass(signal, lowcut=0.5, highcut=40.0, fs=fs, order=4)
        r_peaks = pan_tompkins(filtered, fs=fs)
        logger.info("Picos R detectados: %d", len(r_peaks))

        if len(r_peaks) == 0:
            raise HTTPException(
                status_code=422,
                detail="No se detectaron picos R en la senal. "
                       "Verifica que el CSV tenga la senal ECG en formato correcto "
                       "(una muestra por fila, 360 Hz, sin cabecera)."
            )

        segments, _ = segment_beats(filtered, r_peaks, before=90, after=110)
        segments = normalize_beats(segments)
        logger.info("Latidos segmentados: %d", len(segments))

        results: List[Dict[str, Any]] = []
        mse_values: List[float] = []

        for i in range(len(segments)):
            pred = _predictor.predict(segments[i].tolist(), preprocessed=True)
            mse = pred.get("reconstruction_error")
            if mse is not None:
                mse_values.append(mse)
            results.append({
                "beat_index": i,
                "prediction": pred["prediction"],
                "label": pred["label"],
                "reconstruction_error": mse,
            })

        # Restaurar umbral original si fue modificado
        if original_threshold is not None and not adaptive_threshold:
            _predictor._threshold = original_threshold

        # ── Umbral adaptativo por registro ──────────────────────────────────
        threshold_method = "fijo_entrenamiento"
        effective_threshold = fixed_threshold
        anomalias = sum(1 for r in results if r["prediction"] == 1)

        if adaptive_threshold and mse_values:
            arr = np.array(mse_values)
            adaptive_thresh = float(np.percentile(arr, adaptive_percentile))
            anomalias = 0
            for r in results:
                if r["reconstruction_error"] is not None:
                    is_anom = r["reconstruction_error"] > adaptive_thresh
                    r["prediction"] = 1 if is_anom else 0
                    r["label"] = "anomalia" if is_anom else "normal"
                    if is_anom:
                        anomalias += 1
            effective_threshold = adaptive_thresh
            threshold_method = f"adaptativo_p{adaptive_percentile}"
            logger.info(
                "Umbral adaptativo p%d=%.5f → %d/%d anomalias",
                adaptive_percentile, adaptive_thresh, anomalias, len(results),
            )

        # Estadisticas MSE
        mse_stats = None
        if mse_values:
            arr = np.array(mse_values)
            mse_stats = {
                "min":    round(float(arr.min()),  6),
                "max":    round(float(arr.max()),  6),
                "mean":   round(float(arr.mean()), 6),
                "std":    round(float(arr.std()),  6),
                "p75":    round(float(np.percentile(arr, 75)), 6),
                "p90":    round(float(np.percentile(arr, 90)), 6),
                "p95":    round(float(np.percentile(arr, 95)), 6),
            }
            logger.info(
                "MSE stats — min:%.5f max:%.5f mean:%.5f threshold:%.5f anomalias:%d/%d",
                mse_stats["min"], mse_stats["max"], mse_stats["mean"],
                effective_threshold or 0, anomalias, len(results),
            )

        _save_plot(segments, results)
        plot_b64 = base64.b64encode(_last_plot_bytes).decode("utf-8") if _last_plot_bytes else None

        return {
            "model_used":         _predictor.model_name,
            "model_type":         _predictor.model_type,
            "total_beats":        len(results),
            "anomalias":          anomalias,
            "normales":           len(results) - anomalias,
            "anomaly_rate":       round(anomalias / len(results), 4) if results else 0.0,
            "r_peaks_detected":   len(r_peaks),
            "threshold_used":     round(effective_threshold, 6) if effective_threshold else None,
            "threshold_method":   threshold_method,
            "threshold_training": round(original_threshold, 6) if original_threshold else None,
            "sensitivity":        sensitivity,
            "adaptive_threshold": adaptive_threshold,
            "mse_stats":          mse_stats,
            "resultados":         results,
            "csv_plot":           "/predict-csv/plot",
            "csv_plot_b64":       plot_b64,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error procesando archivo CSV")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/predict-csv/plot",
    summary="Imagen PNG del ultimo CSV procesado",
    tags=["prediccion"],
)
async def get_csv_plot():
    """Retorna la imagen PNG del grid de latidos generado por el ultimo POST /predict-csv."""
    if _last_plot_bytes is None:
        raise HTTPException(status_code=404, detail="No hay grafico disponible.")
    return Response(content=_last_plot_bytes, media_type="image/png")


# ── Landing page minima ───────────────────────────────────────────────────────

_LANDING_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ECG Anomaly Detection API</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f0f4f8; color: #2d3748; min-height: 100vh;
           display: flex; align-items: center; justify-content: center; }
    .card { background: white; border-radius: 16px; padding: 48px;
            max-width: 600px; width: 90%; box-shadow: 0 4px 24px rgba(0,0,0,.1); text-align: center; }
    .badge { display: inline-block; background: #ebf8ff; color: #2b6cb0;
             border-radius: 99px; padding: 4px 14px; font-size: 13px; font-weight: 600;
             margin-bottom: 20px; }
    h1 { font-size: 1.8rem; color: #1a365d; margin-bottom: 8px; }
    p { color: #718096; margin-bottom: 28px; line-height: 1.6; }
    .links { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
    a.btn { display: inline-flex; align-items: center; gap: 8px; padding: 10px 22px;
            border-radius: 8px; font-weight: 600; text-decoration: none;
            font-size: 14px; transition: all .2s; }
    a.btn-primary { background: #2b6cb0; color: white; }
    a.btn-primary:hover { background: #2c5282; }
    a.btn-secondary { background: #e2e8f0; color: #2d3748; }
    a.btn-secondary:hover { background: #cbd5e0; }
    .footer { margin-top: 32px; font-size: 12px; color: #a0aec0; }
  </style>
</head>
<body>
  <div class="card">
    <span class="badge">v1.0.0 · Universidad CESMAG 2026</span>
    <h1>ECG Anomaly Detection API</h1>
    <p>Detecta anomalias en latidos ECG usando Deep Learning.<br>
       Modelo activo: <strong>Autoencoder</strong> entrenado sobre MIT-BIH Arrhythmia Database.</p>
    <div class="links">
      <a href="/docs" class="btn btn-primary">Swagger UI</a>
      <a href="/redoc" class="btn btn-secondary">ReDoc</a>
      <a href="/health" class="btn btn-secondary">Health Check</a>
      <a href="/model-info" class="btn btn-secondary">Model Info</a>
    </div>
    <div class="footer">POST /predict · POST /predict-csv · GET /health · GET /model-info</div>
  </div>
</body>
</html>"""


# ── CLI entry point ───────────────────────────────────────────────────────────

def serve() -> None:
    """Entry point CLI: ecg-serve [--model-dir DIR] [--host HOST] [--port PORT]."""
    import uvicorn

    parser = argparse.ArgumentParser(
        description="ECG Anomaly Detection API Server - Universidad CESMAG 2026"
    )
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Directorio con el mejor modelo guardado (default: ./models)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host del servidor (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Puerto del servidor (default: 8000)")
    parser.add_argument("--cors-origins", type=str, default="*",
                        help="Origenes CORS permitidos separados por coma (default: *)")
    args = parser.parse_args()

    os.environ["MODEL_DIR"] = args.model_dir
    os.environ["CORS_ORIGINS"] = args.cors_origins

    uvicorn.run(app, host=args.host, port=args.port)
