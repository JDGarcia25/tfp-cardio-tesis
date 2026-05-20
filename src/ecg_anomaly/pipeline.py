"""Pipeline principal de deteccion de anomalias ECG (Patron Facade).

Orquesta todo el flujo: carga -> preprocesamiento -> features -> modelos -> evaluacion -> reporte.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ecg_anomaly.config import SystemConfig
from ecg_anomaly.data.loader import MITBIHLoader
from ecg_anomaly.evaluation.comparator import ModelComparator
from ecg_anomaly.features.manual import ManualFeatureExtractor
from ecg_anomaly.features.signal_pca import SignalPCAExtractor
from ecg_anomaly.preprocessing.pipeline import PreprocessingPipeline
from ecg_anomaly.visualization.reports import save_full_report

logger = logging.getLogger(__name__)


class ECGAnomalyPipeline:
    """Sistema principal de deteccion de anomalias ECG.

    Patron Facade: proporciona una interfaz unificada para todo el pipeline.

    Flujo DSR (Fase 3 - Diseno y Desarrollo):
        1. Adquisicion de datos (MIT-BIH, 44 registros)
        2. Preprocesamiento (filtrado, segmentacion, normalizacion)
        3. Extraccion de features (Path A: signal+PCA o Path B: manual)
        4. Implementacion de algoritmos (KMeans, DBSCAN, HDBSCAN, Autoencoder)
        5. Asignacion de anomalias y evaluacion comparativa
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        config.setup_logging()
        self.comparator = ModelComparator(config)
        self._scaler: StandardScaler | None = None
        self._pca = None

    def run(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo.

        Returns:
            DataFrame con tabla comparativa de los 4 modelos.
        """
        logger.info("=" * 60)
        logger.info("SISTEMA DE DETECCION DE ANOMALIAS ECG")
        logger.info("Universidad CESMAG - 2026")
        logger.info("=" * 60)

        # 1. Carga de datos
        logger.info("[1/5] Cargando datos MIT-BIH...")
        loader = MITBIHLoader(self.config)
        dataset = loader.load(self.config.dataset_path)

        # 2. Preprocesamiento
        logger.info("[2/5] Preprocesando senales...")
        preprocessor = PreprocessingPipeline(self.config)
        preprocessed = preprocessor.run(dataset)

        # 3. Extraccion de features
        logger.info("[3/5] Extrayendo features (representacion: %s)...", self.config.representation)
        X_clustering, X_autoencoder = self._extract_features(preprocessed)

        # 4 & 5. Modelos + Evaluacion
        logger.info("[4/5] Ejecutando y evaluando modelos...")
        results_df = self.comparator.run_all(X_clustering, X_autoencoder, preprocessed.labels)

        # 6. Guardar mejor modelo
        logger.info("[5/5] Guardando mejor modelo y generando reporte...")
        self._save_best_model()

        # 7. Reporte
        self._generate_report(results_df)

        # Resumen
        best = self.comparator.get_best_model("extrinsic_f1")
        logger.info("=" * 60)
        logger.info("MEJOR MODELO (F1): %s", best)
        logger.info("=" * 60)
        print("\n" + results_df.to_string(index=False))

        return results_df

    def _extract_features(self, preprocessed):
        """Extrae features segun la representacion configurada."""
        if self.config.representation == "signal_pca":
            extractor = SignalPCAExtractor(self.config.pca_variance_threshold)
            X_clustering = extractor.fit_transform(preprocessed.segments)
            X_autoencoder = extractor.get_raw_for_autoencoder(preprocessed.segments)
            self._scaler = extractor.scaler
            self._pca = extractor.pca
        elif self.config.representation == "manual_features":
            extractor = ManualFeatureExtractor()
            X_clustering = extractor.extract(
                preprocessed.segments,
                preprocessed.r_peaks_flat,
                self.config.sampling_rate,
                preprocessed.record_indices,
            )
            scaler = StandardScaler()
            X_autoencoder = scaler.fit_transform(preprocessed.segments)
            self._scaler = extractor.scaler
            self._pca = None
        else:
            raise ValueError(
                f"Representacion '{self.config.representation}' no soportada. "
                "Use 'signal_pca' o 'manual_features'."
            )

        logger.info(
            "Features: clustering=%s, autoencoder=%s",
            X_clustering.shape,
            X_autoencoder.shape,
        )
        return X_clustering, X_autoencoder

    def _save_best_model(self) -> None:
        """Guarda el mejor modelo (por F1) en ./models/."""
        best_name = self.comparator.get_best_model("extrinsic_f1")
        if not best_name:
            logger.warning("No se pudo determinar el mejor modelo, se omite guardado.")
            return

        detector = None
        for d in self.comparator.detectors:
            if d.name == best_name:
                detector = d
                break

        if detector is None:
            logger.warning("Detector '%s' no encontrado, se omite guardado.", best_name)
            return

        best_result = None
        for r in self.comparator.results:
            if r["model"] == best_name:
                best_result = r
                break

        models_dir = Path("./models")
        model_dir = models_dir / best_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # best_model.json
        model_type = best_name
        if best_result:
            params = best_result.get("params", {})
            threshold = params.get("threshold", None)
        else:
            threshold = None

        best_meta = {
            "model_name": best_name,
            "model_type": model_type,
            "representation": self.config.representation,
            "metric": "extrinsic_f1",
        }
        with open(models_dir / "best_model.json", "w") as f:
            json.dump(best_meta, f, indent=2)

        # scaler.joblib
        if self._scaler is not None:
            joblib.dump(self._scaler, model_dir / "scaler.joblib")

        # pca.joblib (solo signal_pca)
        if self._pca is not None:
            joblib.dump(self._pca, model_dir / "pca.joblib")

        # config.json
        config_data = {"representation": self.config.representation}
        if threshold is not None:
            config_data["threshold"] = threshold
        if best_name in ("kmeans", "dbscan", "hdbscan"):
            if hasattr(detector, "_majority_cluster"):
                config_data["majority_cluster"] = int(detector._majority_cluster)
            elif hasattr(detector, "labels_"):
                unique, counts = np.unique(detector.labels_, return_counts=True)
                majority = int(unique[np.argmax(counts)])
                config_data["majority_cluster"] = majority
        with open(model_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        # Artefactos especificos del modelo
        if best_name == "autoencoder":
            if hasattr(detector, "model") and detector.model is not None:
                detector.model.save(str(model_dir / "model.h5"))
        else:
            joblib.dump(detector, model_dir / "detector.joblib")

        logger.info("Mejor modelo '%s' (F1) guardado en: %s", best_name, model_dir)

    def _generate_report(self, results_df: pd.DataFrame) -> None:
        """Genera y guarda el reporte de resultados."""
        if self.config.generate_plots:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            save_full_report(
                results_df,
                self.comparator.results,
                self.config.output_dir,
            )


def main():
    """Entry point CLI: ecg-run --config config/default.yaml."""
    parser = argparse.ArgumentParser(
        description="ECG Anomaly Detection Pipeline - Universidad CESMAG 2026"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Ruta al archivo de configuracion YAML",
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["signal_pca", "manual_features"],
        default=None,
        help="Override de la representacion de datos",
    )
    args = parser.parse_args()

    config = SystemConfig.from_yaml(args.config)
    if args.representation:
        config.representation = args.representation

    pipeline = ECGAnomalyPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
