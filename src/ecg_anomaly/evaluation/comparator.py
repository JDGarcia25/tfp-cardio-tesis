"""Comparador multi-modelo: ejecuta y evalua todos los detectores."""

import logging
import warnings
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
        eval_idx: Optional[np.ndarray] = None,
    ) -> Dict:
        """Entrena y evalua un detector individual.

        Args:
            detector: Instancia del detector a evaluar.
            X: Datos de entrada (PCA para clustering, raw para autoencoder).
            true_labels: Etiquetas binarias AAMI ground truth.
            fit_idx: Indices para el ajuste. Si es None, se ajusta con todo X.
            eval_idx: Indices sobre los que se MIDEN las metricas extrinsecas.
                Debe ser disjunto de fit_idx. Si es None, se mide sobre todo X
                (valido solo para modelos que se ajustan con todo el dataset,
                como el clustering puro).

        Returns:
            Dict con todas las metricas del modelo.
        """
        logger.info("Evaluando %s...", detector.name)

        # Contrato explicito: fit y eval no pueden solaparse. Si se solapan,
        # las metricas se miden sobre datos de entrenamiento y quedan infladas.
        if fit_idx is not None and eval_idx is not None:
            solapamiento = np.intersect1d(fit_idx, eval_idx)
            if len(solapamiento) > 0:
                raise ValueError(
                    f"fit_idx y eval_idx se solapan en {len(solapamiento):,} "
                    f"indices. Las metricas se mediran sobre datos de "
                    f"entrenamiento y estaran infladas."
                )

        X_fit = X[fit_idx] if fit_idx is not None else X

        # Medir eficiencia
        with EfficiencyTracker() as tracker:
            detector.fit(X_fit)

        # Predecir sobre TODO X: las visualizaciones de los notebooks
        # necesitan la prediccion completa, alineada indice a indice.
        if fit_idx is not None:
            detector.anomaly_labels_ = detector.predict_anomalies(X)

        detector.fit_time_seconds = tracker.elapsed_seconds
        detector.peak_memory_mb = tracker.peak_memory_mb

        result: Dict = {"model": detector.name}

        # --- CLAVE: las metricas se MIDEN solo sobre eval_idx ---
        # Aunque la prediccion cubre todo X, evaluar sobre los latidos que
        # el modelo uso para ajustarse mide memoria, no generalizacion.
        idx_metricas = eval_idx if eval_idx is not None else np.arange(len(X))
        result["n_eval_beats"] = int(len(idx_metricas))
        result["eval_scope"] = "held-out" if eval_idx is not None else "full"

        # Metricas intrinsecas (solo para modelos de clustering)
        if detector.name != "autoencoder" and detector.labels_ is not None:
            intrinsic = evaluate_intrinsic(X, detector.labels_)
            result.update({f"intrinsic_{k}": v for k, v in intrinsic.items()})

        # Puntuaciones continuas para un AUC-ROC real. DBSCAN no define un
        # score de anomalia con sentido de ordenamiento (su salida es una
        # particion con ruido, no un ranking), asi que su AUC quedara en NaN.
        # Reportar NaN es correcto: es mas honesto que fabricar un numero.
        scores = None
        try:
            scores = detector.score_anomalies(X)[idx_metricas]
        except (NotImplementedError, AttributeError, RuntimeError):
            logger.info(
                "%s no expone score_anomalies(); AUC-ROC quedara como NaN",
                detector.name,
            )

        # Metricas extrinsecas, medidas SOLO sobre held-out
        if detector.anomaly_labels_ is not None:
            extrinsic = evaluate_extrinsic(
                true_labels[idx_metricas],
                detector.anomaly_labels_[idx_metricas],
                scores=scores,
            )
            result.update({f"extrinsic_{k}": v for k, v in extrinsic.items()})

        # Metricas de eficiencia
        result.update({f"efficiency_{k}": v for k, v in tracker.to_dict().items()})

        # Parametros del modelo
        result["params"] = detector.get_params()

        # Estadisticas de anomalias
        if detector.anomaly_labels_ is not None:
            n_anomalies = int(np.sum(detector.anomaly_labels_[idx_metricas] == 1))
            result["n_anomalies"] = n_anomalies
            result["anomaly_ratio"] = n_anomalies / len(idx_metricas)

        self.results.append(result)
        self.detectors.append(detector)

        sil = result.get("intrinsic_silhouette")
        sil_txt = f"{sil:.3f}" if sil is not None else "n/a (sin clusters)"

        logger.info(
            "%s completado: F1=%.3f, Silhouette=%s, Tiempo=%.2fs",
            detector.name,
            result.get("extrinsic_f1", 0.0),
            sil_txt,
            tracker.elapsed_seconds,
        )

        return result

    def run_all(
        self,
        X_clustering: np.ndarray,
        X_autoencoder: np.ndarray,
        true_labels: np.ndarray,
        autoencoder_fit_idx: Optional[np.ndarray] = None,
        eval_idx: Optional[np.ndarray] = None,
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
            eval_idx: Indices held-out sobre los que se miden las metricas.
                Se aplica a TODOS los modelos, no solo al autoencoder: solo
                asi la tabla comparativa es honesta, porque todos los modelos
                se miden sobre exactamente el mismo conjunto de latidos.

        Returns:
            DataFrame con resultados comparativos.
        """
        for model_name in self.config.models:
            params = getattr(self.config, f"{model_name}_params", {})
            detector = DetectorFactory.create(model_name, params, seed=self.config.random_seed)

            # Autoencoder usa datos raw, clustering usa PCA/features
            X = X_autoencoder if model_name == "autoencoder" else X_clustering
            fit_idx = autoencoder_fit_idx if model_name == "autoencoder" else None
            self.evaluate_model(detector, X, true_labels, fit_idx=fit_idx, eval_idx=eval_idx)

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
                "Balanced Accuracy": r.get("extrinsic_balanced_accuracy", None),
                "Sensitivity": r.get("extrinsic_sensitivity", None),
                "Specificity": r.get("extrinsic_specificity", None),
                "F1": r.get("extrinsic_f1", None),
                "AUC-ROC": r.get("extrinsic_auc_roc", None),
                "Tiempo (s)": r.get("efficiency_time_seconds", None),
                "Memoria (MB)": r.get("efficiency_peak_memory_mb", None),
                "Anomalias": r.get("n_anomalies", None),
                "Latidos evaluados": r.get("n_eval_beats", None),
                "Alcance eval": r.get("eval_scope", None),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_best_model(self, metric: str = "extrinsic_f1") -> Optional[str]:
        """Retorna el nombre del mejor modelo segun una metrica.

        Los modelos que no reportan la metrica (p. ej. el autoencoder no
        produce clusters -> sin Silhouette) se ignoran en lugar de romper
        la comparacion.
        """
        if not self.results:
            return None

        lower_is_better = {"intrinsic_davies_bouldin", "efficiency_time_seconds"}

        # Filtrar los modelos que si reportan la metrica y son comparables
        candidatos = [
            r for r in self.results
            if r.get(metric) is not None
            and isinstance(r.get(metric), (int, float))
            and np.isfinite(r.get(metric))
        ]
        if not candidatos:
            logger.warning("Ningun modelo reporta la metrica %s", metric)
            return None

        fn = min if metric in lower_is_better else max
        return fn(candidatos, key=lambda r: r[metric])["model"]

    def get_multi_criteria_ranking(
        self, weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """[OBSOLETO - NO USAR EN LA EVALUACION DE LA TESIS]

        Ranking multi-criterio (Composite). Retirado de la metodologia
        tras la evaluacion del jurado por dos defectos de fondo:

        1. Mezcla en un unico escalar magnitudes que responden preguntas
           distintas e inconmensurables: capacidad de deteccion (F1),
           geometria del clustering (Silhouette) y coste computacional
           (tiempo, memoria). Un modelo puede "ganar" por ser rapido.

        2. La normalizacion min-max es RELATIVA al conjunto de modelos
           evaluados: agregar o quitar un modelo cambia el puntaje de
           todos los demas. El ranking no es estable.

        Sustituido por: seleccion con metricas extrinsecas (notebook 05,
        seccion 9) + diagnostico intrinseco por separado, siguiendo la
        recomendacion de Rendon et al. (2011) de no mezclar validacion
        interna y externa en un solo indice.

        Se conserva unicamente para reproducir los resultados del informe
        anterior. NO debe usarse para seleccionar el modelo final.
        """
        warnings.warn(
            "get_multi_criteria_ranking() esta obsoleto y no debe usarse "
            "para la seleccion del modelo. Use metricas extrinsecas "
            "(notebook 05, seccion 9).",
            DeprecationWarning,
            stacklevel=2,
        )

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
                    and np.isfinite(v)
                ]

                if not all_vals:
                    continue

                min_val = min(all_vals)
                max_val = max(all_vals)

                if not np.isfinite(val):
                    # Valor centinela (p.ej. davies_bouldin=inf con <2 clusters
                    # validos): tratar como peor caso posible, no propagar nan.
                    normalized = 0.0
                elif max_val == min_val:
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
