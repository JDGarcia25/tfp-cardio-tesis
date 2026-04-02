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

        if params.get("eps") == "auto":
            params["eps"] = self._optimize_eps(X, params.get("min_samples", 5))
            logger.info("DBSCAN eps auto-optimizado: %.4f", params["eps"])

        self.model = DBSCAN(**params)
        self.labels_ = self.model.fit_predict(X)
        self.anomaly_labels_ = np.where(self.labels_ == -1, 1, 0)
        return self

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        # DBSCAN no tiene predict nativo; usar distancia a core samples
        core_mask = np.zeros(len(self.model.components_), dtype=bool)
        core_mask[:] = True
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(self.model.components_)
        distances, _ = neigh.kneighbors(X)
        eps = self.model.eps
        return np.where(distances.ravel() > eps, 1, 0)

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
    def _optimize_eps(X: np.ndarray, min_samples: int) -> float:
        """Auto-optimiza epsilon usando el percentil 90 de k-distancias."""
        neigh = NearestNeighbors(n_neighbors=min_samples)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
        return float(np.percentile(k_distances, 90))
