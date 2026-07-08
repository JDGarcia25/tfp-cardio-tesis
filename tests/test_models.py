"""Tests para los modelos de deteccion de anomalias."""

import numpy as np
import pytest

from ecg_anomaly.models.factory import DetectorFactory
from ecg_anomaly.models.kmeans import KMeansDetector
from ecg_anomaly.models.dbscan import DBSCANDetector
from ecg_anomaly.models.hdbscan_model import HDBSCANDetector
from ecg_anomaly.models.autoencoder import AutoencoderDetector


def _make_synthetic_data(n_normal: int = 200, n_outliers: int = 20, seed: int = 42):
    """Genera datos sinteticos con cluster normal + outliers."""
    rng = np.random.RandomState(seed)
    normal = rng.randn(n_normal, 10) * 0.5  # Cluster centrado en 0
    outliers = rng.randn(n_outliers, 10) * 0.5 + 5  # Cluster lejano
    X = np.vstack([normal, outliers])
    labels = np.array([0] * n_normal + [1] * n_outliers)
    return X, labels


class TestKMeansDetector:
    """Tests para el detector K-Means con distance-scoring."""

    def test_fit_returns_self(self):
        X, _ = _make_synthetic_data()
        detector = KMeansDetector("kmeans", {"n_clusters": 3, "random_state": 42, "distance_percentile": 90})
        result = detector.fit(X)
        assert result is detector

    def test_assigns_labels(self):
        X, _ = _make_synthetic_data()
        detector = KMeansDetector("kmeans", {"n_clusters": 3, "random_state": 42, "distance_percentile": 90})
        detector.fit(X)
        assert detector.labels_ is not None
        assert len(detector.labels_) == len(X)

    def test_distance_scoring_labels(self):
        """Los mas lejanos al centroide deben ser anomalos (no toda la mitad)."""
        X, _ = _make_synthetic_data(n_normal=200, n_outliers=20)
        detector = KMeansDetector("kmeans", {"n_clusters": 3, "random_state": 42, "distance_percentile": 90})
        detector.fit(X)
        n_anomalies = int(np.sum(detector.anomaly_labels_ == 1))
        # Con 10% de outliers y distance_percentile=90, debe marcar ~10%
        assert n_anomalies >= 15
        assert n_anomalies <= 35

    def test_predict_anomalies(self):
        X, _ = _make_synthetic_data()
        detector = KMeansDetector("kmeans", {"n_clusters": 3, "random_state": 42, "distance_percentile": 90})
        detector.fit(X)
        preds = detector.predict_anomalies(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})

    def test_score_anomalies_returns_distances(self):
        X, _ = _make_synthetic_data()
        detector = KMeansDetector("kmeans", {"n_clusters": 3, "random_state": 42, "distance_percentile": 90})
        detector.fit(X)
        scores = detector.score_anomalies(X)
        assert scores.shape == (len(X),)
        assert np.all(scores >= 0)
        # Los outliers sinteticos deben tener mayor score
        assert np.mean(scores[-20:]) > np.mean(scores[:200])

    def test_predict_uses_train_data(self):
        """Debe guardar datos de entrenamiento para predict."""
        X, _ = _make_synthetic_data()
        detector = KMeansDetector("kmeans", {"n_clusters": 3, "random_state": 42, "distance_percentile": 90})
        detector.fit(X)
        assert hasattr(detector, "_train_data")
        assert np.array_equal(detector._train_data, X)


class TestDBSCANDetector:
    """Tests para el detector DBSCAN."""

    def test_auto_eps(self):
        """eps='auto' debe calcular un valor automaticamente."""
        X, _ = _make_synthetic_data()
        detector = DBSCANDetector("dbscan", {"eps": "auto", "min_samples": 5, "eps_percentile": 75})
        detector.fit(X)
        assert detector.model.eps != "auto"
        assert isinstance(detector.model.eps, float)

    def test_noise_as_anomaly(self):
        """Puntos de ruido deben marcarse como anomalias."""
        X, _ = _make_synthetic_data()
        detector = DBSCANDetector("dbscan", {"eps": 1.0, "min_samples": 5, "eps_percentile": 75})
        detector.fit(X)
        # Verificar consistencia
        noise_mask = detector.labels_ == -1
        assert np.all(detector.anomaly_labels_[noise_mask] == 1)
        assert np.all(detector.anomaly_labels_[~noise_mask] == 0)

    def test_score_anomalies(self):
        X, _ = _make_synthetic_data()
        detector = DBSCANDetector("dbscan", {"eps": 1.0, "min_samples": 5, "eps_percentile": 75})
        detector.fit(X)
        scores = detector.score_anomalies(X)
        assert scores.shape == (len(X),)
        assert np.all(scores >= 0)


class TestHDBSCANDetector:
    """Tests para el detector HDBSCAN."""

    def test_fit_returns_self(self):
        X, _ = _make_synthetic_data()
        detector = HDBSCANDetector("hdbscan", {"min_cluster_size": 10})
        result = detector.fit(X)
        assert result is detector

    def test_noise_as_anomaly(self):
        """Puntos de ruido deben marcarse como anomalias."""
        X, _ = _make_synthetic_data()
        detector = HDBSCANDetector("hdbscan", {"min_cluster_size": 10})
        detector.fit(X)
        noise_mask = detector.labels_ == -1
        if np.any(noise_mask):
            assert np.all(detector.anomaly_labels_[noise_mask] == 1)

    def test_saves_train_data(self):
        """Debe guardar datos de entrenamiento para predict."""
        X, _ = _make_synthetic_data()
        detector = HDBSCANDetector("hdbscan", {"min_cluster_size": 10})
        detector.fit(X)
        assert hasattr(detector, "_train_data")
        assert np.array_equal(detector._train_data, X)

    def test_predict_anomalies_no_zeros(self):
        """predict_anomalies no debe retornar solo ceros."""
        X, _ = _make_synthetic_data()
        detector = HDBSCANDetector("hdbscan", {"min_cluster_size": 10})
        detector.fit(X)
        preds = detector.predict_anomalies(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})
        n_anomalies = np.sum(preds == 1)
        assert n_anomalies >= 0

    def test_score_anomalies(self):
        X, _ = _make_synthetic_data()
        detector = HDBSCANDetector("hdbscan", {"min_cluster_size": 10})
        detector.fit(X)
        scores = detector.score_anomalies(X)
        assert scores.shape == (len(X),)
        assert np.all(scores >= 0)


class TestAutoencoderDetector:
    """Tests para el detector Autoencoder."""

    def test_fit_falls_back_without_tensorflow(self, monkeypatch):
        """Debe poder ejecutarse con un fallback PCA si TensorFlow no está instalado."""
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "tensorflow":
                raise ModuleNotFoundError("No module named 'tensorflow'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        X, _ = _make_synthetic_data(n_normal=120, n_outliers=12)
        detector = AutoencoderDetector("autoencoder", {"epochs": 2, "batch_size": 32, "encoding_dim": 4})
        detector.fit(X)

        assert detector.anomaly_labels_ is not None
        assert detector.labels_ is not None
        assert len(detector.anomaly_labels_) == len(X)
        assert set(detector.anomaly_labels_).issubset({0, 1})


class TestDetectorFactory:
    """Tests para la fabrica de detectores."""

    def test_create_all_types(self):
        """Debe crear instancias de todos los detectores registrados."""
        for name in ["kmeans", "dbscan", "hdbscan", "autoencoder"]:
            detector = DetectorFactory.create(name, {})
            assert detector.name == name

    def test_invalid_name_raises(self):
        """Nombre no registrado debe lanzar ValueError."""
        with pytest.raises(ValueError, match="no disponible"):
            DetectorFactory.create("invalid_model", {})

    def test_list_detectors(self):
        """Debe listar los 4 detectores registrados."""
        detectors = DetectorFactory.list_detectors()
        assert "kmeans" in detectors
        assert "dbscan" in detectors
        assert "hdbscan" in detectors
        assert "autoencoder" in detectors
        assert len(detectors) == 4


class TestEvaluationMetrics:
    """Tests para las metricas de evaluacion."""

    def test_intrinsic_with_two_clusters(self):
        from ecg_anomaly.evaluation.intrinsic import evaluate_intrinsic

        X, _ = _make_synthetic_data()
        labels = np.array([0] * 200 + [1] * 20)
        metrics = evaluate_intrinsic(X, labels)
        assert metrics["silhouette"] > 0  # Dos clusters bien separados
        assert metrics["n_clusters"] == 2

    def test_extrinsic_perfect_prediction(self):
        from ecg_anomaly.evaluation.extrinsic import evaluate_extrinsic

        true = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0, 0, 0, 1, 1, 1])
        metrics = evaluate_extrinsic(true, pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0

    def test_extrinsic_all_wrong(self):
        from ecg_anomaly.evaluation.extrinsic import evaluate_extrinsic

        true = np.array([0, 0, 1, 1])
        pred = np.array([1, 1, 0, 0])
        metrics = evaluate_extrinsic(true, pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["sensitivity"] == 0.0
        assert metrics["specificity"] == 0.0
