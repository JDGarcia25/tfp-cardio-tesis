"""Comparador multi-modelo: ejecuta y evalua todos los detectores."""

import logging
from typing import Dict, List, Optional, Tuple

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
        fit_idx: Optional[np.ndarray] = None,
    ) -> Dict:
        """Entrena y evalua un detector individual.

        Args:
            detector: Instancia del detector a evaluar.
            X: Datos de entrada (PCA para clustering, raw para autoencoder).
            true_labels: Etiquetas binarias AAMI ground truth.
            fit_idx: Si se especifica, el detector se entrena solo con
                X[fit_idx] (p. ej. latidos normales, para evitar fuga de
                datos) y luego se predice sobre todo X para evaluar contra
                true_labels. Si es None, se entrena con todo X (comportamiento
                por defecto, adecuado para modelos de clustering puro).

        Returns:
            Dict con todas las metricas del modelo.
        """
        logger.info("Evaluando %s...", detector.name)

        X_fit = X[fit_idx] if fit_idx is not None else X

        # Medir eficiencia
        with EfficiencyTracker() as tracker:
            detector.fit(X_fit)

        # Si se entreno solo con un subconjunto, predecir sobre todo X
        # para que la evaluacion sea comparable con el resto de modelos.
        if fit_idx is not None:
            detector.anomaly_labels_ = detector.predict_anomalies(X)

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
        autoencoder_fit_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Ejecuta todos los modelos configurados.

        Args:
            X_clustering: Features para clustering (PCA o manual).
            X_autoencoder: Features para autoencoder (raw escalado).
            true_labels: Etiquetas binarias AAMI.
            autoencoder_fit_idx: Indices (tipicamente solo-normales) usados
                para entrenar el autoencoder, evitando fuga de datos. Los
                modelos de clustering puro siguen entrenando con todo
                X_clustering (ver guia de mejoras #1, "Nota honesta").

        Returns:
            DataFrame con resultados comparativos.
        """
        for model_name in self.config.models:
            params = getattr(self.config, f"{model_name}_params", {})
            detector = DetectorFactory.create(model_name, params, seed=self.config.random_seed)

            # Autoencoder usa datos raw, clustering usa PCA/features
            X = X_autoencoder if model_name == "autoencoder" else X_clustering
            fit_idx = autoencoder_fit_idx if model_name == "autoencoder" else None
            self.evaluate_model(detector, X, true_labels, fit_idx=fit_idx)

        return self.get_comparison_table()

    def run_all_per_record(
        self,
        X_clustering: np.ndarray,
        X_autoencoder: np.ndarray,
        true_labels: np.ndarray,
        record_indices: np.ndarray,
    ) -> pd.DataFrame:
        """Ejecuta modelos con evaluacion por registro (fit global, predict por registro).

        Entrena cada modelo UNA vez con todos los datos, luego evalua
        por separado en cada registro y promedia las metricas.

        Args:
            X_clustering: Features para clustering.
            X_autoencoder: Features para autoencoder.
            true_labels: Etiquetas binarias AAMI.
            record_indices: Array [N] con indice del registro de cada latido.

        Returns:
            DataFrame con metricas promedio por modelo (macro y micro).
        """
        per_record_rows = []
        n_records = int(np.max(record_indices)) + 1

        for model_name in self.config.models:
            if model_name == "autoencoder":
                logger.info("Saltando autoencoder en evaluacion por registro (muy lento)")
                continue

            params = getattr(self.config, f"{model_name}_params", {})
            detector = DetectorFactory.create(model_name, params, seed=self.config.random_seed)
            X = X_clustering

            # Fit global (una vez)
            logger.info("Entrenando %s (global) para evaluacion por registro...", model_name)
            with EfficiencyTracker() as tracker:
                detector.fit(X)

            # Evaluar por registro
            f1_list, sens_list, spec_list, prec_list = [], [], [], []
            for rec_id in range(n_records):
                mask = record_indices == rec_id
                n_beats = int(np.sum(mask))
                if n_beats < 30:
                    continue

                X_rec = X[mask]
                y_rec = true_labels[mask]

                pred = detector.predict_anomalies(X_rec)
                extrinsic = evaluate_extrinsic(y_rec, pred)

                f1_list.append(extrinsic.get("f1", 0))
                sens_list.append(extrinsic.get("sensitivity", 0))
                spec_list.append(extrinsic.get("specificity", 0))
                prec_list.append(extrinsic.get("precision", 0))

                per_record_rows.append({
                    "model": model_name,
                    "record": int(rec_id),
                    "n_beats": n_beats,
                    **extrinsic,
                })

            # Agregar fila de promedio
            if f1_list:
                per_record_rows.append({
                    "model": f"{model_name}_macro_avg",
                    "record": -1,
                    "n_beats": len(f1_list),
                    "accuracy": np.mean(f1_list),
                    "f1": np.mean(f1_list),
                    "sensitivity": np.mean(sens_list),
                    "specificity": np.mean(spec_list),
                    "precision": np.mean(prec_list),
                    "f1_std": np.std(f1_list),
                    "sens_std": np.std(sens_list),
                    "spec_std": np.std(spec_list),
                    "prec_std": np.std(prec_list),
                    "f1_above_05": np.mean(np.array(f1_list) > 0.5),
                })

        df_per_record = pd.DataFrame(per_record_rows)
        logger.info(
            "Evaluacion por registro completada: %d filas",
            len(df_per_record),
        )
        return df_per_record

    def get_comparison_table(self) -> pd.DataFrame:
        """Genera tabla comparativa de todos los modelos evaluados."""
        if not self.results:
            return pd.DataFrame()

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

    def get_multi_criteria_ranking(
        self, weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Ranking multi-criterio combinando todas las metricas.

        Normaliza cada metrica a [0, 1] (1=mejor) y calcula suma ponderada.
        Las metricas 'lower-is-better' se invierten antes de normalizar.

        Args:
            weights: Dict con pesos personalizados. Si es None, usa valores por defecto.

        Returns:
            DataFrame con puntajes normalizados, composite y ranking.
        """
        default_weights = {
            "extrinsic_f1": 0.25,
            "extrinsic_sensitivity": 0.20,
            "extrinsic_specificity": 0.10,
            "extrinsic_accuracy": 0.10,
            "extrinsic_auc_roc": 0.10,
            "intrinsic_silhouette": 0.10,
            "intrinsic_davies_bouldin": 0.05,
            "efficiency_time_seconds": 0.05,
            "efficiency_peak_memory_mb": 0.05,
        }
        if weights is not None:
            default_weights.update(weights)

        lower_is_better = {
            "intrinsic_davies_bouldin",
            "efficiency_time_seconds",
            "efficiency_peak_memory_mb",
        }

        rows = []
        for r in self.results:
            scores = {}
            weighted_sum = 0.0
            total_weight = 0.0

            for metric, weight in default_weights.items():
                val = r.get(metric)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue

                all_vals = [
                    res.get(metric)
                    for res in self.results
                ]
                all_vals = [
                    v
                    for v in all_vals
                    if v is not None
                    and not (isinstance(v, float) and np.isnan(v))
                ]

                if not all_vals:
                    continue

                min_val = min(all_vals)
                max_val = max(all_vals)

                if max_val == min_val:
                    normalized = 1.0
                elif metric in lower_is_better:
                    normalized = 1.0 - (val - min_val) / (max_val - min_val)
                else:
                    normalized = (val - min_val) / (max_val - min_val)

                scores[metric.replace("extrinsic_", "").replace("intrinsic_", "").replace("efficiency_", "")] = round(normalized, 4)
                weighted_sum += normalized * weight
                total_weight += weight

            composite = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0
            rows.append({**scores, "Modelo": r["model"], "Composite": composite})

        df = pd.DataFrame(rows)
        df = df.sort_values("Composite", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Rank"
        return df
