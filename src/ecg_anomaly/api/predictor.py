"""Cargador y ejecutor del mejor modelo entrenado para inferencia."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "./models"

# Tolerancias para la heuristica de deteccion Z-score
_ZSCORE_MEAN_TOL = 1.0   # |media| debe ser menor que esto
_ZSCORE_STD_MIN = 0.2    # std debe estar dentro de este rango
_ZSCORE_STD_MAX = 2.0


class ModelPredictor:
    """Carga el mejor modelo guardado y ejecuta inferencia sobre latidos individuales.

    El modelo debe haber sido guardado previamente ejecutando el pipeline de
    entrenamiento completo con ``ecg-run``.

    Deteccion de preprocesamiento
    ------------------------------
    El parametro ``preprocessed`` en ``predict()`` controla si se aplica
    normalizacion Z-score al beat entrante:

    - ``None`` (default): se llama a ``is_zscore_normalized()`` para decidir.
    - ``True``: se asume que el beat ya esta normalizado; no se modifica.
    - ``False``: se aplica Z-score sin importar el contenido del beat.

    Flujo de inferencia (tras la normalizacion opcional):
        1. StandardScaler.transform() — escala segun la distribucion del dataset.
        2. PCA.transform()            — solo para modelos de clustering.
        3. Prediccion del modelo      — MSE vs umbral (autoencoder) o cluster (clustering).
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._keras_model = None
        self._detector = None
        self._scaler = None
        self._pca: Optional[object] = None
        self._threshold: Optional[float] = None
        self._model_name: str = "unknown"
        self._model_type: str = "unknown"
        self._representation: str = "signal_pca"
        self._load()

    # ------------------------------------------------------------------
    # Carga desde disco
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Carga el mejor modelo y preprocesadores desde disco."""
        meta_path = self.model_dir / "best_model.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No se encontro el archivo de metadatos en '{meta_path}'. "
                "Ejecute 'ecg-run' primero para entrenar y guardar el modelo."
            )

        with open(meta_path) as f:
            meta = json.load(f)

        self._model_name = meta["model_name"]
        self._model_type = meta["model_type"]
        self._representation = meta.get("representation", "signal_pca")

        import joblib  # noqa: PLC0415

        model_path = self.model_dir / self._model_name

        self._scaler = joblib.load(model_path / "scaler.joblib")

        pca_path = model_path / "pca.joblib"
        if pca_path.exists():
            self._pca = joblib.load(pca_path)

        if self._model_type == "autoencoder":
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf  # noqa: PLC0415

            self._keras_model = tf.keras.models.load_model(str(model_path / "model.h5"), compile=False)
            with open(model_path / "config.json") as f:
                cfg = json.load(f)
            self._threshold = cfg["threshold"]
        else:
            self._detector = joblib.load(model_path / "detector.joblib")
            with open(model_path / "config.json") as f:
                json.load(f)  # reservado para parametros futuros

        logger.info(
            "Modelo '%s' (tipo: %s) cargado desde '%s'",
            self._model_name,
            self._model_type,
            self.model_dir,
        )

    # ------------------------------------------------------------------
    # Seleccion dinamica de modelo
    # ------------------------------------------------------------------

    def use_model(self, name: str) -> None:
        """Cambia el modelo activo cargando ./models/<name>/.

        Args:
            name: Nombre del subdirectorio del modelo (ej: 'dbscan', 'kmeans').
        """
        model_path = self.model_dir / name
        if not model_path.exists():
            logger.warning("Modelo '%s' no encontrado en %s", name, model_path)
            return
        self._load_specific(name, model_path)
        logger.info(
            "Modelo cambiado a '%s' (tipo: %s) desde '%s'",
            self._model_name, self._model_type, model_path,
        )

    def _load_specific(self, name: str, model_path: Path) -> None:
        """Carga un modelo especifico desde model_path, sin usar best_model.json."""
        import joblib  # noqa: PLC0415

        self._model_name = name

        scaler_path = model_path / "scaler.joblib"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)

        pca_path = model_path / "pca.joblib"
        self._pca = joblib.load(pca_path) if pca_path.exists() else None

        detector_path = model_path / "detector.joblib"
        keras_path = model_path / "model.h5"
        config_path = model_path / "config.json"

        if keras_path.exists():
            self._model_type = "autoencoder"
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf  # noqa: PLC0415
            self._keras_model = tf.keras.models.load_model(str(keras_path), compile=False)
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                self._threshold = cfg.get("threshold")
            self._detector = None
        elif detector_path.exists():
            self._model_type = name
            self._keras_model = None
            self._detector = joblib.load(detector_path)
            if config_path.exists():
                with open(config_path) as f:
                    json.load(f)

    # ------------------------------------------------------------------
    # Propiedades publicas
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Nombre del modelo cargado."""
        return self._model_name

    @property
    def model_type(self) -> str:
        """Tipo del modelo: 'autoencoder' o 'clustering'."""
        return self._model_type

    # ------------------------------------------------------------------
    # Deteccion y normalizacion del beat
    # ------------------------------------------------------------------

    @staticmethod
    def is_zscore_normalized(beat: np.ndarray) -> bool:
        """Heuristica estadistica: determina si un latido ya tiene Z-score aplicado.

        Un latido normalizado por Z-score tiene:
        - Media aproximadamente 0  (|media| < 1.0)
        - Desviacion estandar entre 0.2 y 2.0

        Args:
            beat: Array 1-D con las 200 muestras del latido.

        Returns:
            True si el beat parece estar normalizado; False en caso contrario.
        """
        mean = float(np.mean(beat))
        std = float(np.std(beat))
        return abs(mean) < _ZSCORE_MEAN_TOL and _ZSCORE_STD_MIN < std < _ZSCORE_STD_MAX

    @staticmethod
    def apply_zscore(beat: np.ndarray) -> np.ndarray:
        """Aplica normalizacion Z-score a un latido individual.

        Resta la media y divide por la desviacion estandar. Si la desviacion
        es casi cero (latido plano), solo resta la media para evitar division
        por cero.

        Args:
            beat: Array 1-D de N muestras.

        Returns:
            Array normalizado de la misma forma.
        """
        mean = float(np.mean(beat))
        std = float(np.std(beat))
        if std < 1e-8:
            return beat - mean
        return (beat - mean) / std

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def predict(self, beat: List[float], preprocessed: Optional[bool] = None) -> Dict:
        """Predice si un latido ECG es anomalo.

        Args:
            beat: Lista de 200 muestras ECG.
            preprocessed:
                - ``None`` (default): detecta automaticamente si el beat
                  ya tiene Z-score por heuristica estadistica.
                - ``True``: asume que el beat ya esta normalizado.
                  No se calcula media ni desviacion estandar.
                - ``False``: aplica Z-score antes de inferencia.

        Returns:
            Dict con claves:
                prediction, label, reconstruction_error, threshold,
                model_name, normalization_applied.
        """
        X = np.array(beat, dtype=np.float32).reshape(1, -1)
        raw_mean = float(np.mean(X[0]))
        raw_std = float(np.std(X[0]))
        logger.info(
            "Beat recibido: mean=%.4f, std=%.4f, preprocessed=%s",
            raw_mean, raw_std, preprocessed,
        )

        if preprocessed is True:
            X_scaled = self._scaler.transform(X)
            return self._run_model(X_scaled, normalization_applied=False)

        if preprocessed is None:
            already_normalized = self.is_zscore_normalized(X[0])
            source = "auto-detectado"
        else:
            already_normalized = False
            source = "explicito"

        if already_normalized:
            X_scaled = self._scaler.transform(X)
            return self._run_model(X_scaled, normalization_applied=False)

        X[0] = self.apply_zscore(X[0])
        logger.info(
            "Z-score aplicado (%s): mean_original=%.4f, std_original=%.4f, "
            "mean_final=%.4f, std_final=%.4f",
            source, raw_mean, raw_std,
            float(np.mean(X[0])), float(np.std(X[0])),
        )
        X_scaled = self._scaler.transform(X)
        return self._run_model(X_scaled, normalization_applied=True)

    def _run_model(self, X_scaled: np.ndarray, normalization_applied: bool) -> Dict:
        """Ejecuta la prediccion sobre datos ya escalados.

        Flujo clustering: Scaler -> PCA (opcional) -> detector.predict_anomalies
        Flujo autoencoder: Scaler -> modelo keras -> MSE vs umbral
        """
        if self._model_type == "autoencoder":
            reconstructed = self._keras_model.predict(X_scaled, verbose=0)
            error = float(np.mean((X_scaled - reconstructed) ** 2))
            pred = int(error > self._threshold)
            return {
                "prediction": pred,
                "label": "anomalia" if pred == 1 else "normal",
                "reconstruction_error": error,
                "threshold": self._threshold,
                "model_name": self._model_name,
                "normalization_applied": normalization_applied,
            }
        X_features = self._pca.transform(X_scaled) if self._pca is not None else X_scaled
        pred_arr = self._detector.predict_anomalies(X_features)
        pred = int(pred_arr[0])
        return {
            "prediction": pred,
            "label": "anomalia" if pred == 1 else "normal",
            "reconstruction_error": None,
            "threshold": None,
            "model_name": self._model_name,
            "normalization_applied": normalization_applied,
        }
