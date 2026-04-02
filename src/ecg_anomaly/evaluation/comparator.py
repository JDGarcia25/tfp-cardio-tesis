"""Comparador multi-modelo: ejecuta y evalua todos los detectores."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ecg_anomaly.config import SystemConfig
from ecg_anomaly.evaluation.efficiency import EfficiencyTracker
from ecg_anomaly.evaluation.extrinsic import evaluate_extrinsic
from ecg_anomaly.evaluation.intrinsic import evaluate_intrinsic
from ecg_anomaly.models.base import BaseAnomalyDetector
from ecg_anomaly.models.factory import DetectorFactory

logger = logging.getLogger(__name__)


class ModelComparator:
    """Evaluacion comparativa de multiples detectores de anomalias.

    Ejecuta cada modelo y recolecta metricas intrinsecas, extrinsecas
    y de eficiencia en un DataFrame unificado.

    Entregable final: Tabla comparativa con 4 metodos x metricas.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.results: List[Dict] = []
        self.detectors: List[BaseAnomalyDetector] = []

    def evaluate_model(
        self,
        detector: BaseAnomalyDetector,
        X: np.ndarray,
        true_labels: np.ndarray,
    ) -> Dict:
        """Entrena y evalua un detector individual.

        Args:
            detector: Instancia del detector a evaluar.
            X: Datos de entrada (PCA para clustering, raw para autoencoder).
            true_labels: Etiquetas binarias AAMI ground truth.

        Returns:
            Dict con todas las metricas del modelo.
        """
        logger.info("Evaluando %s...", detector.name)

        # Medir eficiencia
        with EfficiencyTracker() as tracker:
            detector.fit(X)

        detector.fit_time_seconds = tracker.elapsed_seconds
        detector.peak_memory_mb = tracker.peak_memory_mb

        result: Dict = {
            "model": detector.name,
        }

        # Metricas intrinsecas (solo para modelos de clustering)
        if detector.name != "autoencoder" and detector.labels_ is not None:
            intrinsic = evaluate_intrinsic(X, detector.labels_)
            result.update({f"intrinsic_{k}": v for k, v in intrinsic.items()})

        # Metricas extrinsecas (todos los modelos)
        if detector.anomaly_labels_ is not None:
            extrinsic = evaluate_extrinsic(true_labels, detector.anomaly_labels_)
            result.update({f"extrinsic_{k}": v for k, v in extrinsic.items()})

        # Metricas de eficiencia
        result.update({f"efficiency_{k}": v for k, v in tracker.to_dict().items()})

        # Parametros del modelo
        result["params"] = detector.get_params()

        # Estadisticas de anomalias
        if detector.anomaly_labels_ is not None:
            n_anomalies = int(np.sum(detector.anomaly_labels_ == 1))
            result["n_anomalies"] = n_anomalies
            result["anomaly_ratio"] = n_anomalies / len(detector.anomaly_labels_)

        self.results.append(result)
        self.detectors.append(detector)

        logger.info(
            "%s completado: F1=%.3f, Silhouette=%.3f, Tiempo=%.2fs",
            detector.name,
            result.get("extrinsic_f1", 0),
            result.get("intrinsic_silhouette", -1),
            tracker.elapsed_seconds,
        )

        return result

    def run_all(
        self,
        X_clustering: np.ndarray,
        X_autoencoder: np.ndarray,
        true_labels: np.ndarray,
    ) -> pd.DataFrame:
        """Ejecuta todos los modelos configurados.

        Args:
            X_clustering: Features para clustering (PCA o manual).
            X_autoencoder: Features para autoencoder (raw escalado).
            true_labels: Etiquetas binarias AAMI.

        Returns:
            DataFrame con resultados comparativos.
        """
        for model_name in self.config.models:
            params = getattr(self.config, f"{model_name}_params", {})
            detector = DetectorFactory.create(model_name, params)

            # Autoencoder usa datos raw, clustering usa PCA/features
            X = X_autoencoder if model_name == "autoencoder" else X_clustering
            self.evaluate_model(detector, X, true_labels)

        return self.get_comparison_table()

    def get_comparison_table(self) -> pd.DataFrame:
        """Genera tabla comparativa de todos los modelos evaluados."""
        if not self.results:
            return pd.DataFrame()

        # Seleccionar columnas relevantes para la tabla
        rows = []
        for r in self.results:
            row = {
                "Modelo": r["model"],
                "Silhouette": r.get("intrinsic_silhouette", None),
                "Davies-Bouldin": r.get("intrinsic_davies_bouldin", None),
                "Calinski-Harabasz": r.get("intrinsic_calinski_harabasz", None),
                "Accuracy": r.get("extrinsic_accuracy", None),
                "Sensitivity": r.get("extrinsic_sensitivity", None),
                "Specificity": r.get("extrinsic_specificity", None),
                "F1": r.get("extrinsic_f1", None),
                "AUC-ROC": r.get("extrinsic_auc_roc", None),
                "Tiempo (s)": r.get("efficiency_time_seconds", None),
                "Memoria (MB)": r.get("efficiency_peak_memory_mb", None),
                "Anomalias": r.get("n_anomalies", None),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_best_model(self, metric: str = "extrinsic_f1") -> Optional[str]:
        """Retorna el nombre del mejor modelo segun una metrica."""
        if not self.results:
            return None

        # Metricas donde menor es mejor
        lower_is_better = {"intrinsic_davies_bouldin", "efficiency_time_seconds"}

        best_idx = 0
        best_val = self.results[0].get(metric, float("-inf"))

        for i, r in enumerate(self.results[1:], 1):
            val = r.get(metric, float("-inf"))
            if val is None:
                continue
            if metric in lower_is_better:
                if val < best_val:
                    best_val = val
                    best_idx = i
            else:
                if val > best_val:
                    best_val = val
                    best_idx = i

        return self.results[best_idx]["model"]
