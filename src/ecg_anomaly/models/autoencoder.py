"""Detector de anomalias basado en Autoencoder (Deep Learning no supervisado).

El autoencoder se entrena SOLO con latidos normales (fit_idx). Aprende a
reconstruir la morfologia normal; un latido anomalo, al no parecerse a
nada de lo visto en entrenamiento, produce un error de reconstruccion alto.

Regla de anomalia: error de reconstruccion > umbral, donde el umbral es el
percentil (1 - normal_fpr) de los errores DEL CONJUNTO DE ENTRENAMIENTO
(100% normal por construccion).

IMPORTANTE - sobre normal_fpr:
    normal_fpr NO es la prevalencia de anomalias del dataset. Es el
    PRESUPUESTO DE FALSAS ALARMAS: la fraccion de latidos normales que
    aceptamos marcar por error. normal_fpr=0.105 significa "tolero que
    ~10.5% de los latidos normales generen una alarma falsa".

    Este valor se fija SIN mirar las etiquetas AAMI: se deriva del
    presupuesto de revision manual de un servicio de cardiologia. Por
    eso la tasa de marcado resultante sobre el dataset completo puede no
    coincidir con la prevalencia real: el modelo decide cuantas
    anomalias hay, no se le impone.
"""

import logging
import os
from typing import Dict, List

import numpy as np
from sklearn.decomposition import PCA

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
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ModuleNotFoundError:
            logger.warning("TensorFlow no está instalado; usando fallback PCA-based para continuar.")
            return self._fit_fallback(X)

        random_state = self.params.get("random_state", 42)

        # Reproducibilidad
        tf.random.set_seed(random_state)

        # Construir modelo
        model = self._build_model(X.shape[1], keras)
        self.model = model

        # Keras toma el ULTIMO validation_split*N del array SIN barajar. Como
        # los latidos vienen ordenados por registro, ese 10% final seria un
        # unico paciente y el EarlyStopping mediria la capacidad de reconstruir
        # a ESE paciente, no la morfologia normal general. Barajamos antes.
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X))
        X_shuffled = X[perm]

        # Entrenar (autoencoder: input == target)
        self.history_ = model.fit(
            X_shuffled,
            X_shuffled,
            epochs=self.params.get("epochs", 50),
            batch_size=self.params.get("batch_size", 256),
            validation_split=0.1,
            shuffle=True,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                ),
            ],
        )

        # Los errores se calculan sobre X en su orden ORIGINAL, no sobre
        # X_shuffled: reconstruction_errors_ debe alinearse indice a indice
        # con las etiquetas que recibe el comparador.
        reconstructed = model.predict(X, verbose=0)
        self.reconstruction_errors_ = np.mean((X - reconstructed) ** 2, axis=1)

        # Umbral: percentil de los errores del conjunto de entrenamiento,
        # que es 100% normal. Define el presupuesto de falsas alarmas sobre
        # la clase normal, NO la prevalencia de anomalias del dataset.
        normal_fpr = self.params.get("normal_fpr",
                                     self.params.get("anomaly_rate", 0.105))
        threshold_percentile = (1.0 - normal_fpr) * 100
        self.threshold_ = float(
            np.percentile(self.reconstruction_errors_, threshold_percentile)
        )

        # Asignar etiquetas. Unificado con K-Means (>=): con datos continuos
        # la diferencia entre > y >= es despreciable, pero mantiene la
        # regla de anomalia consistente entre detectores.
        self.anomaly_labels_ = np.where(
            self.reconstruction_errors_ >= self.threshold_, 1, 0
        )
        self.labels_ = self.anomaly_labels_

        logger.info(
            "Autoencoder: umbral=%.6f (p%.1f sobre el fit set solo-normal, "
            "normal_fpr=%.3f). Marcadas %d/%d del FIT SET (%.1f%%). "
            "La tasa sobre el dataset completo se decide en predict_anomalies.",
            self.threshold_, threshold_percentile, normal_fpr,
            int(np.sum(self.anomaly_labels_ == 1)), len(self.anomaly_labels_),
            np.mean(self.anomaly_labels_) * 100,
        )

        return self

    def _fit_fallback(self, X: np.ndarray) -> "AutoencoderDetector":
        """Fallback simple basado en PCA cuando TensorFlow no está disponible."""
        n_components = min(4, X.shape[1])
        pca = PCA(n_components=n_components, random_state=self.params.get("random_state", 42))
        projected = pca.fit_transform(X)

        # Reconstruct with PCA and use reconstruction error as anomaly score
        reconstructed = pca.inverse_transform(projected)
        self.reconstruction_errors_ = np.mean((X - reconstructed) ** 2, axis=1)

        normal_fpr = self.params.get("normal_fpr",
                                     self.params.get("anomaly_rate", 0.105))
        threshold_percentile = (1.0 - normal_fpr) * 100
        self.threshold_ = float(np.percentile(self.reconstruction_errors_, threshold_percentile))
        self.anomaly_labels_ = np.where(self.reconstruction_errors_ >= self.threshold_, 1, 0)
        self.labels_ = self.anomaly_labels_
        self.model = pca

        logger.info(
            "Autoencoder fallback (PCA): umbral=%.6f (p%.1f, normal_fpr=%.3f), %d anomalias (%.1f%%)",
            self.threshold_,
            threshold_percentile,
            normal_fpr,
            int(np.sum(self.anomaly_labels_ == 1)),
            np.mean(self.anomaly_labels_) * 100,
        )
        return self

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.threshold_ is None:
            raise RuntimeError("Debe llamar fit() antes de predict_anomalies()")
        reconstructed = self.model.predict(X, verbose=0)
        errors = np.mean((X - reconstructed) ** 2, axis=1)
        return np.where(errors >= self.threshold_, 1, 0)

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
