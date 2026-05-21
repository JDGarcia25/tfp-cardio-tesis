"""Detector de anomalias basado en DBSCAN.

Regla de anomalia: puntos etiquetados como ruido (label=-1)
se consideran anomalias.
"""

import logging
from typing import Dict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from ecg_anomaly.models.base import BaseAnomalyDetector

logger = logging.getLogger(__name__)


class DBSCANDetector(BaseAnomalyDetector):
    """Detector DBSCAN (Nivel 2 - Densidad).

    No asume forma de clusters, identifica ruido como anomalia.
    Soporta auto-optimizacion de epsilon via grafico k-distancias.
    """

    def fit(self, X: np.ndarray) -> "DBSCANDetector":
        params = {**self.params}

        eps_percentile = params.pop("eps_percentile", 75)
        if params.get("eps") == "auto":
            params["eps"] = self._optimize_eps(X, params.get("min_samples", 5), eps_percentile)
            logger.info("DBSCAN eps auto-optimizado (p%d): %.4f", eps_percentile, params["eps"])

        self.model = DBSCAN(**params)
        self.labels_ = self.model.fit_predict(X)
        self.anomaly_labels_ = np.where(self.labels_ == -1, 1, 0)
        return self

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        if len(self.model.components_) == 0:
            return np.ones(len(X), dtype=int)
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(self.model.components_)
        distances, _ = neigh.kneighbors(X)
        eps = self.model.eps
        return np.where(distances.ravel() > eps, 1, 0)

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Distancia al core sample mas cercano (mayor = mas anomalo)."""
        if len(self.model.components_) == 0:
            return np.full(len(X), np.inf)
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(self.model.components_)
        distances, _ = neigh.kneighbors(X)
        return distances.ravel()

    def get_params(self) -> Dict:
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = int(np.sum(self.labels_ == -1))
        return {
            **self.params,
            "eps_used": self.model.eps if self.model else None,
            "n_clusters_found": n_clusters,
            "n_noise": n_noise,
        }

    @staticmethod
    def _optimize_eps(X: np.ndarray, min_samples: int, percentile: int = 75) -> float:
        """Auto-optimiza epsilon usando el percentil configurable de k-distancias."""
        neigh = NearestNeighbors(n_neighbors=min_samples)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
        return float(np.percentile(k_distances, percentile))
