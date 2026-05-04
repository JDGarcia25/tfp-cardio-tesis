"""Detector de anomalias basado en K-Means.

Regla de anomalia: el cluster mayoritario se considera normal,
todos los clusters minoritarios se consideran anomalos.
"""

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from ecg_anomaly.models.base import BaseAnomalyDetector


class KMeansDetector(BaseAnomalyDetector):
    """Detector K-Means (Nivel 1 - Baseline).

    Clustering particional que asume clusters esfericos.
    Establece el piso de rendimiento para la comparacion.
    """

    def fit(self, X: np.ndarray) -> "KMeansDetector":
        self._train_data = X
        self.model = KMeans(**self.params)
        self.labels_ = self.model.fit_predict(X)
        self._assign_anomalies()
        return self

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        labels = self.model.predict(X)
        return np.where(labels == self._majority_cluster, 0, 1)

    def get_params(self) -> Dict:
        return {**self.params, "n_clusters_found": len(np.unique(self.labels_))}

    def _assign_anomalies(self) -> None:
        """Cluster con mas latidos = normal. Resto = anomalo."""
        unique, counts = np.unique(self.labels_, return_counts=True)
        self._majority_cluster = unique[np.argmax(counts)]
        self.anomaly_labels_ = np.where(self.labels_ == self._majority_cluster, 0, 1)
