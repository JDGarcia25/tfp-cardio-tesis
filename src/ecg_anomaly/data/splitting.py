"""Splits reproducibles para evitar fuga de datos (data leakage).

En deteccion de anomalias no supervisada, el estandar es ajustar el
preprocesamiento (scaler, PCA, autoencoder) solo con latidos normales y
evaluar sobre el resto (normales restantes + todas las anomalias). Ajustar
sobre todo el dataset contamina la estimacion de rendimiento.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from ecg_anomaly.preprocessing.pipeline import PreprocessedData


def make_normal_fit_split(
    preprocessed: PreprocessedData, seed: int = 42, val_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Separa indices para ajustar transformaciones solo con normales.

    Args:
        preprocessed: Datos preprocesados con `labels` (0=normal, 1=anomalo).
        seed: Semilla para el split reproducible.
        val_fraction: Fraccion de latidos normales reservada para evaluacion.

    Returns:
        fit_idx: subconjunto de latidos NORMALES para fit del scaler/PCA/AE.
        eval_idx: todo lo demas (normales restantes + todas las anomalias),
            que es donde se mide el rendimiento.
    """
    labels = preprocessed.labels
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    fit_idx, held_normal_idx = train_test_split(
        normal_idx, test_size=val_fraction, random_state=seed
    )
    eval_idx = np.concatenate([held_normal_idx, anomaly_idx])
    eval_idx.sort()
    return fit_idx, eval_idx
