"""Detector de anomalias basado en Autoencoder (Deep Learning no supervisado).

El autoencoder aprende a reconstruir latidos normales. Los latidos con
alto error de reconstruccion se clasifican como anomalias.

Regla de anomalia: error de reconstruccion > percentil dinamico calculado
a partir de anomaly_rate (default: 0.105 = 10.5% mas anomalo).
"""

import logging
import os
from typing import Dict, List

import numpy as np

from ecg_anomaly.models.base import BaseAnomalyDetector

logger = logging.getLogger(__name__)

# Suprimir logs verbosos de TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class AutoencoderDetector(BaseAnomalyDetector):
    """Detector Autoencoder (Nivel 4 - Deep Learning).

    Arquitectura simetrica encoder-decoder con BatchNormalization y Dropout.
    Se entrena sin etiquetas (reconstruccion). Las anomalias se detectan
    por error de reconstruccion elevado.
    """

    def __init__(self, name: str, params: Dict):
        super().__init__(name, params)
        self.threshold_: float | None = None
        self.reconstruction_errors_: np.ndarray | None = None
        self.history_ = None

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        import tensorflow as tf
        from tensorflow import keras

        # Reproducibilidad
        tf.random.set_seed(42)

        # Construir modelo
        model = self._build_model(X.shape[1], keras)
        self.model = model

        # Entrenar (autoencoder: input == target)
        self.history_ = model.fit(
            X,
            X,
            epochs=self.params.get("epochs", 50),
            batch_size=self.params.get("batch_size", 256),
            validation_split=0.1,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                ),
            ],
        )

        # Calcular errores de reconstruccion
        reconstructed = model.predict(X, verbose=0)
        self.reconstruction_errors_ = np.mean((X - reconstructed) ** 2, axis=1)

        # Umbral dinamico basado en tasa de anomalia real
        anomaly_rate = self.params.get("anomaly_rate", 0.105)
        threshold_percentile = (1.0 - anomaly_rate) * 100
        self.threshold_ = float(np.percentile(self.reconstruction_errors_, threshold_percentile))

        # Asignar etiquetas
        self.anomaly_labels_ = np.where(
            self.reconstruction_errors_ > self.threshold_, 1, 0
        )
        self.labels_ = self.anomaly_labels_

        logger.info(
            "Autoencoder: umbral=%.6f (p%.1f, anomaly_rate=%.3f), %d anomalias (%.1f%%)",
            self.threshold_,
            threshold_percentile,
            anomaly_rate,
            int(np.sum(self.anomaly_labels_ == 1)),
            np.mean(self.anomaly_labels_) * 100,
        )

        return self

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.threshold_ is None:
            raise RuntimeError("Debe llamar fit() antes de predict_anomalies()")
        reconstructed = self.model.predict(X, verbose=0)
        errors = np.mean((X - reconstructed) ** 2, axis=1)
        return np.where(errors > self.threshold_, 1, 0)

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Error de reconstruccion MSE (mayor = mas anomalo)."""
        if self.model is None:
            raise RuntimeError("Debe llamar fit() antes de score_anomalies()")
        reconstructed = self.model.predict(X, verbose=0)
        return np.mean((X - reconstructed) ** 2, axis=1)

    def get_params(self) -> Dict:
        return {
            **self.params,
            "threshold": self.threshold_,
            "n_anomalies": int(np.sum(self.anomaly_labels_)) if self.anomaly_labels_ is not None else 0,
        }

    def _build_model(self, input_dim: int, keras) -> "keras.Model":
        """Construye la arquitectura del autoencoder."""
        hidden_layers: List[int] = self.params.get("hidden_layers", [128, 64])
        encoding_dim: int = self.params.get("encoding_dim", 32)
        learning_rate: float = self.params.get("learning_rate", 0.001)

        # Encoder
        encoder_input = keras.Input(shape=(input_dim,))
        x = encoder_input
        for units in hidden_layers:
            x = keras.layers.Dense(units, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.2)(x)
        encoded = keras.layers.Dense(encoding_dim, activation="relu", name="encoding")(x)

        # Decoder (espejo del encoder)
        x = encoded
        for units in reversed(hidden_layers):
            x = keras.layers.Dense(units, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
        decoded = keras.layers.Dense(input_dim, activation="linear")(x)

        model = keras.Model(encoder_input, decoded, name="autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )

        logger.info(
            "Autoencoder: %d -> %s -> %d -> %s -> %d",
            input_dim,
            hidden_layers,
            encoding_dim,
            list(reversed(hidden_layers)),
            input_dim,
        )

        return model
