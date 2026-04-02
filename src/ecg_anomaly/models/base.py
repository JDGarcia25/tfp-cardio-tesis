"""Clase base abstracta para detectores de anomalias en ECG."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseAnomalyDetector(ABC):
    """Interfaz base para todos los modelos de deteccion de anomalias.

    Patron Strategy: cada algoritmo implementa fit() y predict_anomalies()
    con su propia logica de asignacion de anomalias.

    Atributos tras fit():
        labels_: Etiquetas de clustering (o reconstruccion para autoencoder).
        anomaly_labels_: Etiquetas binarias (0=normal, 1=anomalia).
        fit_time_seconds: Tiempo de entrenamiento.
        peak_memory_mb: Memoria pico durante entrenamiento.
    """

    def __init__(self, name: str, params: Dict):
        self.name = name
        self.params = params
        self.model: Any = None
        self.labels_: np.ndarray | None = None
        self.anomaly_labels_: np.ndarray | None = None
        self.fit_time_seconds: float = 0.0
        self.peak_memory_mb: float = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseAnomalyDetector":
        """Entrena el modelo sobre los datos.

        Args:
            X: Matriz de features [N, D].

        Returns:
            self (para encadenamiento).
        """

    @abstractmethod
    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Predice anomalias en datos nuevos.

        Args:
            X: Matriz de features [N, D].

        Returns:
            Array binario [N] (0=normal, 1=anomalia).
        """

    @abstractmethod
    def get_params(self) -> Dict:
        """Retorna los parametros del modelo."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
