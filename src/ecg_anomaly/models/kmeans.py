"""Detector de anomalias basado en K-Means con distance-scoring.

Regla de anomalia: los latidos mas lejanos a su centroide (normalizado
por la dispersion del cluster) se consideran anomalos. El umbral se
deriva por defecto de la valla de Tukey (Q3 + 1.5*IQR) sobre la
distribucion de scores, sin consultar el ground truth.

Nota sobre el metodo del codo (Satopaa et al., 2011): se evaluo como
alternativa, pero la distancia a la cuerda min-max colapsa cuando la
distribucion de scores tiene una cola extrema (exactamente lo que
produce la correccion de la "paradoja del singleton": unos pocos
latidos con score muy alto). Con esa cola, el punto de maxima distancia
a la cuerda degenera al primer indice, y el umbral resultante marca
practicamente el 100% de los latidos como anomalos. La valla de Tukey,
basada en cuartiles, es insensible a esa cola por construccion.
"""

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from ecg_anomaly.models.base import BaseAnomalyDetector


class KMeansDetector(BaseAnomalyDetector):
    """Detector K-Means (Nivel 1 - Baseline mejorado).

    Usa k clusters y distance-scoring normalizado: los latidos con mayor
    distancia relativa a su centroide se marcan como anomalos. El umbral
    se deriva de la geometria de los scores (valla de Tukey) salvo que
    se solicite explicitamente el modo por percentil.
    """

    def _iqr_threshold(self, scores: np.ndarray) -> float:
        """Umbral por la valla de Tukey: Q3 + 1.5 * IQR.

        Criterio estandar de deteccion de outliers univariados (Tukey,
        1977), robusto a colas extremas porque se basa en cuartiles, no
        en el rango [min, max]. NO usa etiquetas: el umbral emerge de la
        dispersion de los scores.
        """
        q1, q3 = np.percentile(scores, [25.0, 75.0])
        iqr = q3 - q1
        return float(q3 + 1.5 * iqr)

    def fit(self, X: np.ndarray) -> "KMeansDetector":
        params = {k: v for k, v in self.params.items()
                  if k not in ("distance_percentile", "threshold_method",
                               "min_cluster_size")}
        self.model = KMeans(**params)
        self.labels_ = self.model.fit_predict(X)

        # --- Estadisticas por cluster, calculadas en fit ---
        # Se guardan para que predict_anomalies() sobre datos nuevos use
        # exactamente los mismos valores de referencia que el entrenamiento.
        k = self.model.n_clusters
        d_own = np.linalg.norm(X - self.model.cluster_centers_[self.labels_], axis=1)
        self._cluster_sizes = np.bincount(self.labels_, minlength=k)
        self._cluster_scale = np.ones(k, dtype=float)
        for c in range(k):
            m = self.labels_ == c
            if m.sum() >= 2:
                # Dispersion tipica del cluster (mediana = robusta a outliers)
                self._cluster_scale[c] = max(float(np.median(d_own[m])), 1e-6)

        # Clusters degenerados: no son patrones, son outliers con centroide
        # propio. Se marcan para que sus miembros NO hereden score bajo.
        self._min_size = int(self.params.get("min_cluster_size", 10))
        self._degenerate = self._cluster_sizes < self._min_size

        self._scores = self.score_anomalies(X)

        # El umbral se deriva de la GEOMETRIA, no de la prevalencia.
        method = self.params.get("threshold_method", "iqr")
        if method == "iqr":
            self._threshold = self._iqr_threshold(self._scores)
        else:
            # Modo percentil: solo para el analisis de sensibilidad.
            # Declarar el percentil como SUPUESTO, no como dato.
            dist_pct = self.params.get("distance_percentile", 89.5)
            self._threshold = float(np.percentile(self._scores, dist_pct))

        self.anomaly_labels_ = np.where(self._scores >= self._threshold, 1, 0)
        return self

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_anomalies(X)
        return np.where(scores >= self._threshold, 1, 0)

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Distancia al centroide NORMALIZADA por la dispersion del cluster.

        Corrige la paradoja del singleton: con la regla np.min(distances),
        un latido que forma su propio cluster obtiene distancia 0 y se
        clasifica como normal, justo al reves de lo que debe pasar.

        Dos correcciones:
          1. La distancia se divide por la dispersion tipica del cluster
             asignado -> un latido lejano DENTRO de su cluster puntua alto
             aunque el cluster sea compacto.
          2. Los miembros de clusters degenerados (< min_cluster_size)
             reciben la distancia al centroide VALIDO mas cercano, no al
             suyo propio. Un outlier con centroide propio deja de
             beneficiarse de su aislamiento.
        """
        distances = self.model.transform(X)          # [N, k]
        assign = np.argmin(distances, axis=1)
        d_own = distances[np.arange(len(X)), assign]

        # (1) Normalizar por la dispersion del cluster asignado
        scores = d_own / self._cluster_scale[assign]

        # (2) Reasignar los miembros de clusters degenerados
        if self._degenerate.any() and (~self._degenerate).any():
            valid = np.where(~self._degenerate)[0]
            d_valid = distances[:, valid]
            nearest_valid = valid[np.argmin(d_valid, axis=1)]
            d_to_valid = np.min(d_valid, axis=1)
            is_degen = self._degenerate[assign]
            scores[is_degen] = (d_to_valid[is_degen]
                                / self._cluster_scale[nearest_valid[is_degen]])

        return scores

    def get_params(self) -> Dict:
        return {
            **self.params,
            "n_clusters_found": len(np.unique(self.labels_)),
            "n_degenerate_clusters": int(self._degenerate.sum())
                                     if hasattr(self, "_degenerate") else None,
            "threshold_distance": float(self._threshold)
                                  if hasattr(self, "_threshold") else None,
        }
