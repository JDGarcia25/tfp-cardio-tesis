"""Tests para el comparador de modelos (ModelComparator)."""

import numpy as np
import pandas as pd
import pytest

from ecg_anomaly.config import SystemConfig
from ecg_anomaly.evaluation.comparator import ModelComparator


@pytest.fixture
def config():
    return SystemConfig.from_yaml("config/default.yaml")


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    n = 500
    X = rng.randn(n, 10)
    X[:50] += 5
    labels = np.array([1] * 50 + [0] * 450)
    return X, labels


class TestModelComparator:
    def test_initialization(self, config):
        comparator = ModelComparator(config)
        assert comparator.results == []
        assert comparator.detectors == []

    def test_evaluate_model_returns_dict(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        detector = DetectorFactory.create("kmeans", {"n_clusters": 2, "random_state": 42})
        comparator = ModelComparator(config)
        result = comparator.evaluate_model(detector, X, labels)
        assert isinstance(result, dict)
        assert "model" in result
        assert result["model"] == "kmeans"
        assert "extrinsic_f1" in result
        assert "efficiency_time_seconds" in result

    def test_evaluate_model_appends_result(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        detector = DetectorFactory.create("kmeans", {"n_clusters": 2, "random_state": 42})
        comparator = ModelComparator(config)
        comparator.evaluate_model(detector, X, labels)
        assert len(comparator.results) == 1
        assert len(comparator.detectors) == 1

    def test_get_comparison_table_columns(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        comparator = ModelComparator(config)
        for name in ["kmeans", "dbscan"]:
            detector = DetectorFactory.create(name, {})
            comparator.evaluate_model(detector, X, labels)
        df = comparator.get_comparison_table()
        expected = {"Modelo", "Silhouette", "F1", "Accuracy", "Tiempo (s)", "Memoria (MB)"}
        assert expected.issubset(df.columns)
        assert len(df) == 2

    def test_get_best_model_by_f1(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        comparator = ModelComparator(config)
        for name in ["kmeans", "dbscan"]:
            detector = DetectorFactory.create(name, {})
            comparator.evaluate_model(detector, X, labels)
        best = comparator.get_best_model("extrinsic_f1")
        assert best in ("kmeans", "dbscan")

    def test_get_best_model_empty(self, config):
        comparator = ModelComparator(config)
        assert comparator.get_best_model() is None

    def test_get_multi_criteria_ranking(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        comparator = ModelComparator(config)
        for name in ["kmeans", "dbscan"]:
            detector = DetectorFactory.create(name, {})
            comparator.evaluate_model(detector, X, labels)
        df = comparator.get_multi_criteria_ranking()
        assert "Composite" in df.columns
        assert "Modelo" in df.columns
        assert len(df) == 2
        assert df.index.name == "Rank"
        assert df.iloc[0]["Composite"] >= df.iloc[1]["Composite"]

    def test_get_multi_criteria_ranking_custom_weights(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        comparator = ModelComparator(config)
        for name in ["kmeans", "dbscan"]:
            detector = DetectorFactory.create(name, {})
            comparator.evaluate_model(detector, X, labels)
        weights = {"extrinsic_f1": 1.0, "efficiency_time_seconds": 0.0}
        df = comparator.get_multi_criteria_ranking(weights=weights)
        assert "Composite" in df.columns

    def test_get_comparison_table_empty(self, config):
        comparator = ModelComparator(config)
        df = comparator.get_comparison_table()
        assert df.empty

    def test_run_all(self, config):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 12)
        X_ae = rng.randn(300, 200)
        labels = np.array([1] * 30 + [0] * 270)
        comparator = ModelComparator(config)
        df = comparator.run_all(X, X_ae, labels)
        assert len(df) == 4
        assert "Modelo" in df.columns
        assert "F1" in df.columns

    def test_run_all_per_record(self, config):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 12)
        X_ae = rng.randn(200, 200)
        labels = np.array([1] * 20 + [0] * 180)
        record_indices = np.array([0] * 100 + [1] * 100)
        comparator = ModelComparator(config)
        df = comparator.run_all_per_record(X, X_ae, labels, record_indices)
        assert len(df) > 0
        assert "model" in df.columns
        assert "f1" in df.columns

    def test_results_contain_intrinsic_metrics(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        detector = DetectorFactory.create("kmeans", {"n_clusters": 2, "random_state": 42})
        comparator = ModelComparator(config)
        result = comparator.evaluate_model(detector, X, labels)
        assert "intrinsic_silhouette" in result
        assert "intrinsic_davies_bouldin" in result

    def test_results_contain_extrinsic_metrics(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        detector = DetectorFactory.create("kmeans", {"n_clusters": 2, "random_state": 42})
        comparator = ModelComparator(config)
        result = comparator.evaluate_model(detector, X, labels)
        assert "extrinsic_f1" in result
        assert "extrinsic_sensitivity" in result
        assert "extrinsic_specificity" in result

    def test_results_contain_efficiency_metrics(self, config, synthetic_data):
        X, labels = synthetic_data
        from ecg_anomaly.models.factory import DetectorFactory
        detector = DetectorFactory.create("kmeans", {"n_clusters": 2, "random_state": 42})
        comparator = ModelComparator(config)
        result = comparator.evaluate_model(detector, X, labels)
        assert "efficiency_time_seconds" in result
        assert "efficiency_peak_memory_mb" in result

    def test_evaluate_autoencoder(self, config):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 200)
        labels = np.array([1] * 20 + [0] * 180)
        from ecg_anomaly.models.factory import DetectorFactory
        detector = DetectorFactory.create("autoencoder", {"epochs": 1, "batch_size": 64})
        comparator = ModelComparator(config)
        result = comparator.evaluate_model(detector, X, labels)
        assert result["model"] == "autoencoder"
        assert "extrinsic_f1" in result
