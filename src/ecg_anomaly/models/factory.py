"""Fabrica de detectores de anomalias (Patron Factory)."""

from typing import Dict, List, Type

from ecg_anomaly.models.autoencoder import AutoencoderDetector
from ecg_anomaly.models.base import BaseAnomalyDetector
from ecg_anomaly.models.dbscan import DBSCANDetector
from ecg_anomaly.models.hdbscan_model import HDBSCANDetector
from ecg_anomaly.models.kmeans import KMeansDetector


class DetectorFactory:
    """Fabrica extensible de detectores de anomalias.

    Registra los 4 algoritmos del proyecto:
    - kmeans: K-Means (Nivel 1 - Baseline)
    - dbscan: DBSCAN (Nivel 2 - Densidad)
    - hdbscan: HDBSCAN (Nivel 3 - Densidad jerarquica)
    - autoencoder: Autoencoder (Nivel 4 - Deep Learning)
    """

    _detectors: Dict[str, Type[BaseAnomalyDetector]] = {
        "kmeans": KMeansDetector,
        "dbscan": DBSCANDetector,
        "hdbscan": HDBSCANDetector,
        "autoencoder": AutoencoderDetector,
    }

    @classmethod
    def create(cls, name: str, params: Dict) -> BaseAnomalyDetector:
        """Crea una instancia del detector especificado.

        Args:
            name: Nombre del algoritmo (kmeans, dbscan, hdbscan, autoencoder).
            params: Hiperparametros del modelo.

        Raises:
            ValueError: Si el nombre no esta registrado.
        """
        if name not in cls._detectors:
            raise ValueError(
                f"Detector '{name}' no disponible. "
                f"Registrados: {cls.list_detectors()}"
            )
        return cls._detectors[name](name, params)

    @classmethod
    def register(cls, name: str, detector_class: Type[BaseAnomalyDetector]) -> None:
        """Registra un nuevo tipo de detector en tiempo de ejecucion."""
        cls._detectors[name] = detector_class

    @classmethod
    def list_detectors(cls) -> List[str]:
        """Lista los nombres de detectores disponibles."""
        return list(cls._detectors.keys())
