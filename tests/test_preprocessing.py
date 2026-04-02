"""Tests para el modulo de preprocesamiento."""

import numpy as np
import pytest

from ecg_anomaly.preprocessing.filters import butterworth_bandpass, notch_filter
from ecg_anomaly.preprocessing.segmentation import normalize_beats, segment_beats


class TestButterworthBandpass:
    """Tests para el filtro pasa-banda Butterworth."""

    def test_removes_dc_offset(self):
        """El filtro debe eliminar componentes DC (< lowcut)."""
        fs = 360
        t = np.arange(0, 2, 1 / fs)
        # Senal con offset DC + componente de 10 Hz
        signal = 5.0 + np.sin(2 * np.pi * 10 * t)
        filtered = butterworth_bandpass(signal, 0.5, 40.0, fs)
        # El offset DC debe ser ~0
        assert abs(np.mean(filtered)) < 0.5

    def test_preserves_ecg_frequency(self):
        """El filtro debe preservar frecuencias dentro de la banda."""
        fs = 360
        t = np.arange(0, 2, 1 / fs)
        freq = 10.0  # Dentro de 0.5-40 Hz
        signal = np.sin(2 * np.pi * freq * t)
        filtered = butterworth_bandpass(signal, 0.5, 40.0, fs)
        # La amplitud debe mantenerse cerca de 1
        assert np.max(np.abs(filtered)) > 0.7

    def test_attenuates_high_frequency(self):
        """El filtro debe atenuar frecuencias altas (> highcut)."""
        fs = 360
        t = np.arange(0, 5, 1 / fs)
        # Componente de 150 Hz (well above 40 Hz cutoff, near Nyquist)
        signal = np.sin(2 * np.pi * 150 * t)
        filtered = butterworth_bandpass(signal, 0.5, 40.0, fs)
        # Check that steady-state middle portion is well attenuated
        mid = filtered[len(filtered) // 4 : len(filtered) // 2]
        assert np.max(np.abs(mid)) < 0.05

    def test_2d_input(self):
        """Debe funcionar con arrays 2D (multiples senales)."""
        fs = 360
        t = np.arange(0, 1, 1 / fs)
        signals = np.array([np.sin(2 * np.pi * 10 * t) for _ in range(3)])
        filtered = butterworth_bandpass(signals, 0.5, 40.0, fs)
        assert filtered.shape == signals.shape

    def test_output_same_length(self):
        """La salida debe tener la misma longitud que la entrada."""
        signal = np.random.randn(1000)
        filtered = butterworth_bandpass(signal, 0.5, 40.0, 360)
        assert len(filtered) == len(signal)


class TestSegmentation:
    """Tests para la segmentacion de latidos."""

    def test_segment_length(self):
        """Cada segmento debe tener exactamente before+after muestras."""
        signal = np.random.randn(5000)
        peaks = np.array([500, 1000, 1500, 2000, 2500])
        segments, _ = segment_beats(signal, peaks, before=90, after=110)
        assert segments.shape[1] == 200  # 90 + 110

    def test_boundary_exclusion(self):
        """Latidos en los bordes de la senal deben excluirse."""
        signal = np.random.randn(1000)
        # Pico muy cerca del inicio
        peaks = np.array([10, 500, 995])
        segments, valid_idx = segment_beats(signal, peaks, before=90, after=110)
        # Solo el pico en 500 debe ser valido
        assert len(segments) == 1
        assert valid_idx[0] == 1

    def test_all_peaks_valid(self):
        """Todos los picos lejos de los bordes deben producir segmentos."""
        signal = np.random.randn(10000)
        peaks = np.array([200, 600, 1000, 1400, 1800])
        segments, valid_idx = segment_beats(signal, peaks, before=90, after=110)
        assert len(segments) == 5
        assert len(valid_idx) == 5

    def test_empty_peaks(self):
        """Array vacio de picos debe retornar arrays vacios."""
        signal = np.random.randn(1000)
        segments, valid_idx = segment_beats(signal, np.array([]), before=90, after=110)
        assert segments.shape == (0, 200)
        assert len(valid_idx) == 0

    def test_custom_window(self):
        """Ventanas personalizadas deben funcionar correctamente."""
        signal = np.random.randn(5000)
        peaks = np.array([500, 1000])
        segments, _ = segment_beats(signal, peaks, before=50, after=150)
        assert segments.shape[1] == 200  # 50 + 150


class TestNormalization:
    """Tests para la normalizacion Z-score."""

    def test_zero_mean(self):
        """Cada latido normalizado debe tener media ~0."""
        segments = np.random.randn(10, 200) * 5 + 3
        normalized = normalize_beats(segments)
        means = np.mean(normalized, axis=1)
        np.testing.assert_allclose(means, 0, atol=1e-10)

    def test_unit_std(self):
        """Cada latido normalizado debe tener std ~1."""
        segments = np.random.randn(10, 200) * 5 + 3
        normalized = normalize_beats(segments)
        stds = np.std(normalized, axis=1)
        np.testing.assert_allclose(stds, 1, atol=1e-10)

    def test_constant_signal_handled(self):
        """Senales constantes no deben producir NaN."""
        segments = np.ones((3, 200)) * 5
        normalized = normalize_beats(segments)
        assert not np.any(np.isnan(normalized))
