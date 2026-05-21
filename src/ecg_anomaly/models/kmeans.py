"""Detector de anomalias basado en K-Means con distance-scoring.

Regla de anomalia: los latidos mas lejanos a su centroide
se consideran anomalos (distance-scoring por percentil).
"""

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from ecg_anomaly.models.base import BaseAnomalyDetector


class KMeansDetector(BaseAnomalyDetector):
    """Detector K-Means (Nivel 1 - Baseline mejorado).

    Usa k=10 clusters y distance-scoring: los latidos en el percentil
    superior de distancia a su centroide se marcan como anomalos.
    """

    def fit(self, X: np.ndarray) -> "KMeansDetector":
        self._train_data = X
        params = {k: v for k, v in self.params.items() if k != "distance_percentile"}
        self.model = KMeans(**params)
        self.labels_ = self.model.fit_predict(X)
        self._scores = self.score_anomalies(X)
        dist_pct = self.params.get("distance_percentile", 89.5)
        self._threshold = float(np.percentile(self._scores, dist_pct))
        self.anomaly_labels_ = np.where(self._scores >= self._threshold, 1, 0)
        return self

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_anomalies(X)
        return np.where(scores >= self._threshold, 1, 0)

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Distancia euclidiana al centroide mas cercano (mayor = mas anomalo)."""
        distances = self.model.transform(X)
        return np.min(distances, axis=1)

    def get_params(self) -> Dict:
        return {
            **self.params,
            "n_clusters_found": len(np.unique(self.labels_)),
            "threshold_distance": float(self._threshold) if hasattr(self, "_threshold") else None,
        }
