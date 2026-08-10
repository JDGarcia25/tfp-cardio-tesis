"""Tests para el modulo de extraccion de features."""

import numpy as np
import pytest

from ecg_anomaly.features.manual import N_MANUAL_FEATURES_TOTAL, ManualFeatureExtractor
from ecg_anomaly.features.signal_pca import SignalPCAExtractor


class TestSignalPCAExtractor:
    """Tests para la extraccion de features Path A (signal + PCA)."""

    def _make_segments(self, n: int = 100, dim: int = 200) -> np.ndarray:
        """Genera segmentos sinteticos."""
        rng = np.random.RandomState(42)
        return rng.randn(n, dim)

    def test_reduces_dimensions(self):
        """PCA debe reducir las dimensiones."""
        segments = self._make_segments(200, 200)
        extractor = SignalPCAExtractor(variance_threshold=0.95)
        reduced = extractor.fit_transform(segments)
        assert reduced.shape[0] == 200
        assert reduced.shape[1] < 200

    def test_preserves_variance(self):
        """Debe retener al menos el % de varianza configurado."""
        segments = self._make_segments(200, 200)
        extractor = SignalPCAExtractor(variance_threshold=0.95)
        extractor.fit_transform(segments)
        assert extractor.explained_variance_ratio >= 0.95

    def test_transform_after_fit(self):
        """transform() debe funcionar tras fit_transform()."""
        segments = self._make_segments(200, 200)
        extractor = SignalPCAExtractor(variance_threshold=0.95)
        reduced = extractor.fit_transform(segments)

        new_data = self._make_segments(50, 200)
        transformed = extractor.transform(new_data)
        assert transformed.shape == (50, reduced.shape[1])

    def test_transform_before_fit_raises(self):
        """transform() sin fit debe lanzar error."""
        extractor = SignalPCAExtractor()
        with pytest.raises(RuntimeError):
            extractor.transform(self._make_segments(10, 200))

    def test_autoencoder_data_full_dim(self):
        """get_raw_for_autoencoder() debe retornar datos de dimension completa."""
        segments = self._make_segments(100, 200)
        extractor = SignalPCAExtractor(variance_threshold=0.95)
        extractor.fit_transform(segments)

        ae_data = extractor.get_raw_for_autoencoder(segments)
        assert ae_data.shape == (100, 200)

    def test_n_components_property(self):
        """n_components debe reflejar las componentes seleccionadas."""
        segments = self._make_segments(200, 200)
        extractor = SignalPCAExtractor(variance_threshold=0.95)

        assert extractor.n_components == 0
        extractor.fit_transform(segments)
        assert extractor.n_components > 0


class TestManualFeatureExtractor:
    """Tests para la extraccion de features Path B (manual)."""

    def _make_data(self, n: int = 100) -> tuple:
        """Genera segmentos sinteticos y posiciones R."""
        rng = np.random.RandomState(42)
        segments = rng.randn(n, 200)
        r_positions = np.arange(n) * 360
        record_idx = np.zeros(n, dtype=int)
        return segments, r_positions, record_idx

    def test_extract_returns_correct_shape(self):
        """Debe retornar array [N, N_MANUAL_FEATURES_TOTAL] (base + RR + ventana)."""
        segments, r_positions, record_idx = self._make_data(100)
        extractor = ManualFeatureExtractor()
        features = extractor.extract(segments, r_positions, 360, record_idx, before_r=90)
        assert features.shape == (100, N_MANUAL_FEATURES_TOTAL)

    def test_extract_without_before_r_raises(self):
        """before_r es obligatorio: no debe asumirse simetria del segmento."""
        segments, r_positions, record_idx = self._make_data(100)
        extractor = ManualFeatureExtractor()
        with pytest.raises(ValueError):
            extractor.extract(segments, r_positions, 360, record_idx)

    def test_fit_transform_matches_extract(self):
        """fit()+transform() sobre el mismo conjunto debe igualar a extract()."""
        segments, r_positions, record_idx = self._make_data(100)

        extractor_a = ManualFeatureExtractor()
        combined = extractor_a.extract(segments, r_positions, 360, record_idx, before_r=90)

        extractor_b = ManualFeatureExtractor()
        extractor_b.fit(segments, r_positions, 360, record_idx, before_r=90)
        split = extractor_b.transform(segments, r_positions, 360, record_idx, before_r=90)

        np.testing.assert_allclose(combined, split)

    def test_transform_before_fit_raises(self):
        """transform() sin fit() previo debe lanzar error."""
        segments, r_positions, record_idx = self._make_data(10)
        extractor = ManualFeatureExtractor()
        with pytest.raises(RuntimeError):
            extractor.transform(segments, r_positions, 360, record_idx, before_r=90)

    def test_fit_uses_only_given_rows(self):
        """El scaler debe ajustarse solo con las filas pasadas a fit(), no con todas."""
        segments, r_positions, record_idx = self._make_data(100)

        extractor = ManualFeatureExtractor()
        extractor.fit(segments[:50], r_positions[:50], 360, record_idx[:50], before_r=90)
        mean_fit_only = extractor.scaler.mean_.copy()

        extractor_full = ManualFeatureExtractor()
        extractor_full.fit(segments, r_positions, 360, record_idx, before_r=90)
        mean_full = extractor_full.scaler.mean_

        assert not np.allclose(mean_fit_only, mean_full)

    def test_rr_first_beat_uses_mean(self):
        """El primer latido usa mean_rr (no tiene anterior)."""
        n = 5
        segments = np.random.randn(n, 200)
        r_positions = np.arange(n) * 360
        record_idx = np.zeros(n, dtype=int)

        extractor = ManualFeatureExtractor()
        raw_features = extractor._extract_raw(segments, r_positions, 360, record_idx, before_r=90)

        assert raw_features[0, 4] == 1000.0

    def test_rr_subsequent_beats(self):
        """Latidos posteriores tienen RR hacia atras."""
        n = 5
        segments = np.random.randn(n, 200)
        r_positions = np.arange(n) * 360
        record_idx = np.zeros(n, dtype=int)

        extractor = ManualFeatureExtractor()
        raw_features = extractor._extract_raw(segments, r_positions, 360, record_idx, before_r=90)

        assert raw_features[1, 4] == 1000.0

    def test_rr_different_record_uses_mean(self):
        """Cuando cambia de registro, usa mean_rr."""
        segments = np.random.randn(10, 200)
        r_positions = np.arange(10) * 360
        record_idx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

        extractor = ManualFeatureExtractor()
        raw_features = extractor._extract_raw(segments, r_positions, 360, record_idx, before_r=90)

        assert raw_features[3, 4] == 1000.0

    def test_record_indices_optional(self):
        """record_indices puede ser None."""
        segments, r_positions, _ = self._make_data(100)
        extractor = ManualFeatureExtractor()
        features = extractor.extract(segments, r_positions, 360, before_r=90)
        assert features.shape == (100, N_MANUAL_FEATURES_TOTAL)