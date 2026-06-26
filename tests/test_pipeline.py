"""Tests para el pipeline principal (ECGAnomalyPipeline)."""

import numpy as np
import pytest

from ecg_anomaly.config import SystemConfig
from ecg_anomaly.pipeline import ECGAnomalyPipeline


@pytest.fixture
def config():
    return SystemConfig.from_yaml("config/default.yaml")


class TestECGAnomalyPipeline:
    def test_initialization(self, config):
        pipeline = ECGAnomalyPipeline(config)
        assert pipeline.config is config
        assert pipeline.comparator is not None

    def test_extract_features_signal_pca(self, config):
        class MockPreprocessed:
            segments = np.random.RandomState(42).randn(100, 200)
            labels = np.array([1] * 10 + [0] * 90)
            r_peaks_flat = np.zeros(100)
            record_indices = np.zeros(100, dtype=int)

        config.representation = "signal_pca"
        pipeline = ECGAnomalyPipeline(config)
        X_clust, X_ae = pipeline._extract_features(MockPreprocessed())
        assert X_clust.shape[1] <= 200
        assert X_ae.shape == (100, 200)

    def test_representation_raises_on_invalid(self, config):
        pipeline = ECGAnomalyPipeline(config)
        config.representation = "invalid_rep"
        with pytest.raises(ValueError, match="no soportada"):
            pipeline._extract_features(None)

    def test_run_returns_dataframe(self, config):
        config.models = ["kmeans"]
        config.generate_plots = False
        kmeans_params = getattr(config, "kmeans_params", {})
        kmeans_params["n_clusters"] = 2
        kmeans_params["random_state"] = 42
        pipeline = ECGAnomalyPipeline(config)
        df = pipeline.run()
        assert df is not None
        assert "Modelo" in df.columns
        assert "F1" in df.columns

    def test_best_model_logged(self, config):
        config.models = ["kmeans", "dbscan"]
        config.generate_plots = False
        pipeline = ECGAnomalyPipeline(config)
        df = pipeline.run()
        best = pipeline.comparator.get_best_model("extrinsic_f1")
        assert best in ("kmeans", "dbscan")
