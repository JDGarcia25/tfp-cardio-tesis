"""Tests para el modulo de extraccion de features."""

import numpy as np
import pytest

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

        assert extractor.n_components == 0  # Antes de fit
        extractor.fit_transform(segments)
        assert extractor.n_components > 0
