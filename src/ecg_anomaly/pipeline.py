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

        # 6. Guardar todos los modelos entrenados (ANTES de per-record, que es muy lento)
        logger.info("[6/5] Guardando todos los modelos...")
        self._save_all_models()

        # 5b. Evaluacion por registro (opcional, labels por registro)
        record_indices = getattr(preprocessed, "record_indices", None)
        if record_indices is not None:
            logger.info("[5b/5] Evaluacion por registro...")
            per_record_df = self.comparator.run_all_per_record(
                X_clustering, X_autoencoder, preprocessed.labels, record_indices
            )
            self._per_record_df = per_record_df
        else:
            self._per_record_df = None

        # Reporte
        self._generate_report(results_df)

        # Resumen
        best = self.comparator.get_best_model("extrinsic_f1")
        logger.info("=" * 60)
        logger.info("MEJOR MODELO (F1): %s", best)
        logger.info("=" * 60)
        print("\n" + results_df.to_string(index=False))

        # Mostrar resumen per-record si existe
        if self._per_record_df is not None:
            avg_rows = self._per_record_df[
                self._per_record_df["model"].str.contains("_macro_avg", na=False)
            ]
            if len(avg_rows) > 0:
                print("\n--- Evaluacion por registro (promedio macro) ---")
                print(avg_rows.to_string(index=False, columns=[
                    "model", "f1", "sensitivity", "specificity", "precision",
                    "f1_std", "f1_above_05"
                ]))

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

    def _save_all_models(self) -> None:
        """Guarda todos los modelos entrenados en ./models/."""
        models_dir = Path("./models")
        best_name = self.comparator.get_best_model("extrinsic_f1")

        for detector in self.comparator.detectors:
            name = detector.name
            result = next((r for r in self.comparator.results if r["model"] == name), None)

            model_dir = models_dir / name
            model_dir.mkdir(parents=True, exist_ok=True)

            # scaler.joblib (compartido)
            if self._scaler is not None:
                joblib.dump(self._scaler, model_dir / "scaler.joblib")

            # pca.joblib (compartido)
            if self._pca is not None:
                joblib.dump(self._pca, model_dir / "pca.joblib")

            # config.json
            config_data = {"representation": self.config.representation}
            if result:
                params = result.get("params", {})
                threshold = params.get("threshold", None)
                if threshold is not None:
                    config_data["threshold"] = threshold
            if name == "kmeans" and hasattr(detector, "_threshold"):
                config_data["distance_threshold"] = float(detector._threshold)
            if name in ("dbscan", "hdbscan") and hasattr(detector, "labels_"):
                unique, counts = np.unique(detector.labels_, return_counts=True)
                non_noise = unique[unique >= 0]
                if len(non_noise) > 0:
                    majority = int(non_noise[np.argmax(counts[unique >= 0])])
                    config_data["majority_cluster"] = majority
            with open(model_dir / "config.json", "w") as f:
                json.dump(config_data, f, indent=2)

            # Artefactos especificos del modelo
            if name == "autoencoder":
                if hasattr(detector, "model") and detector.model is not None:
                    detector.model.save(str(model_dir / "model.h5"))
            else:
                joblib.dump(detector, model_dir / "detector.joblib")

            logger.info("Modelo '%s' guardado en: %s", name, model_dir)

        # best_model.json (solo referencia cual es el mejor)
        if best_name:
            best_meta = {
                "model_name": best_name,
                "model_type": best_name,
                "representation": self.config.representation,
                "metric": "extrinsic_f1",
                "available_models": [d.name for d in self.comparator.detectors],
            }
            with open(models_dir / "best_model.json", "w") as f:
                json.dump(best_meta, f, indent=2)

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
