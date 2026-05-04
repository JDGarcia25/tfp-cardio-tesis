"""Detector de anomalias basado en HDBSCAN.

Reemplaza OPTICS. HDBSCAN es su evolucion natural: selecciona parametros
automaticamente, es mas robusto con datos de alta dimensionalidad,
y tiene implementacion madura en Python.

Regla de anomalia: puntos etiquetados como ruido (label=-1)
se consideran anomalias.
"""

import logging
from typing import Dict

import numpy as np

from ecg_anomaly.models.base import BaseAnomalyDetector

logger = logging.getLogger(__name__)


class HDBSCANDetector(BaseAnomalyDetector):
    """Detector HDBSCAN (Nivel 3 - Densidad jerarquica).

    Seleccion automatica de densidad optima. No requiere epsilon.
    Mas robusto y facil de configurar que DBSCAN.
    """

    def fit(self, X: np.ndarray) -> "HDBSCANDetector":
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            from hdbscan import HDBSCAN

        self._train_data = X
        self.model = HDBSCAN(**self.params, copy=True)
        self.labels_ = self.model.fit_predict(X)
        self.anomaly_labels_ = np.where(self.labels_ == -1, 1, 0)

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        logger.info(
            "HDBSCAN: %d clusters, %d ruido (%.1f%%)",
            n_clusters,
            int(np.sum(self.labels_ == -1)),
            np.mean(self.labels_ == -1) * 100,
        )
        return self

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        try:
            from hdbscan import approximate_predict
            labels, _ = approximate_predict(self.model, X)
            return np.where(labels == -1, 1, 0)
        except (ImportError, AttributeError):
            from sklearn.neighbors import NearestNeighbors

            core_mask = self.labels_ >= 0
            core_data = self._train_data[core_mask]
            if core_data is None or len(core_data) == 0:
                logger.warning("HDBSCAN predict_anomalies: sin datos de entrenamiento")
                return np.zeros(len(X), dtype=int)
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(core_data)
            distances, _ = neigh.kneighbors(X)
            threshold = np.percentile(distances, 95)
            return np.where(distances.ravel() > threshold, 1, 0)

    def get_params(self) -> Dict:
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = int(np.sum(self.labels_ == -1))
        return {
            **self.params,
            "n_clusters_found": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(self.labels_) if len(self.labels_) > 0 else 0,
        }