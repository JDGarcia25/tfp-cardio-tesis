"""Metricas intrinsecas de clustering (sin etiquetas reales).

Responden a: ¿Los clusters son coherentes internamente?
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_intrinsic(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calcula metricas intrinsecas de calidad de clustering.

    Args:
        X: Matriz de features [N, D].
        labels: Etiquetas de clustering asignadas.

    Returns:
        Dict con silhouette, davies_bouldin, calinski_harabasz.
        Retorna valores centinela si hay menos de 2 clusters validos.
    """
    # Filtrar puntos de ruido (label == -1) para las metricas
    mask = labels >= 0
    valid_labels = labels[mask]
    valid_X = X[mask]

    n_clusters = len(set(valid_labels))

    if n_clusters < 2 or len(valid_X) < n_clusters + 1:
        return {
            "silhouette": -1.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
            "n_clusters": n_clusters,
        }

    return {
        "silhouette": float(silhouette_score(valid_X, valid_labels)),
        "davies_bouldin": float(davies_bouldin_score(valid_X, valid_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(valid_X, valid_labels)),
        "n_clusters": n_clusters,
    }
