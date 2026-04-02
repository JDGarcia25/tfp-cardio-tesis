"""Path B: Extraccion manual de caracteristicas morfologicas y estadisticas."""

import logging
from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Nombres de las 12 features extraidas
FEATURE_NAMES: List[str] = [
    "r_amplitude",
    "s_amplitude",
    "qrs_duration",
    "amplitude_range",
    "rr_current",
    "rr_ratio",
    "rr_diff",
    "mean",
    "std",
    "kurtosis",
    "dominant_freq",
    "spectral_energy",
]


class ManualFeatureExtractor:
    """Extractor Path B: caracteristicas manuales (~12 features).

    Extrae features morfologicas, temporales (intervalos RR),
    estadisticas y de frecuencia de cada latido segmentado.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def extract(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int = 360,
    ) -> np.ndarray:
        """Extrae y escala features de los segmentos.

        Args:
            segments: Array [N, beat_length] con latidos normalizados.
            r_peak_positions: Posiciones absolutas de los picos R.
            fs: Frecuencia de muestreo.

        Returns:
            Array [N, 12] con features escaladas (StandardScaler).
        """
        features = self._extract_raw(segments, r_peak_positions, fs)
        scaled = self.scaler.fit_transform(features)
        self._is_fitted = True

        logger.info("Features manuales: %d latidos x %d features", *scaled.shape)
        return scaled

    def _extract_raw(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int,
    ) -> np.ndarray:
        """Extrae features sin escalar."""
        n_beats = len(segments)
        before_r = segments.shape[1] // 2  # Posicion del pico R en el segmento

        # Calcular intervalos RR globales
        rr_intervals = np.diff(r_peak_positions) / fs * 1000.0  # en ms
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 800.0

        features = np.zeros((n_beats, len(FEATURE_NAMES)))

        for i, seg in enumerate(segments):
            r_idx = before_r  # Pico R esta en la posicion 'before' del segmento

            # --- Morfologicas (4) ---
            features[i, 0] = seg[r_idx]  # r_amplitude
            # S wave: minimo en los 30 muestras despues del pico R
            s_region = seg[r_idx : min(r_idx + 30, len(seg))]
            features[i, 1] = np.min(s_region) if len(s_region) > 0 else 0.0  # s_amplitude
            features[i, 2] = self._estimate_qrs_duration(seg, r_idx, fs)  # qrs_duration (ms)
            features[i, 3] = np.max(seg) - np.min(seg)  # amplitude_range

            # --- Intervalos RR (3) ---
            if i < len(rr_intervals):
                features[i, 4] = rr_intervals[i]  # rr_current
                features[i, 5] = (
                    rr_intervals[i] / mean_rr if mean_rr > 0 else 1.0
                )  # rr_ratio
            else:
                features[i, 4] = mean_rr
                features[i, 5] = 1.0

            if i > 0 and i < len(rr_intervals):
                features[i, 6] = rr_intervals[i] - rr_intervals[i - 1]  # rr_diff
            else:
                features[i, 6] = 0.0

            # --- Estadisticas (3) ---
            features[i, 7] = np.mean(seg)  # mean
            features[i, 8] = np.std(seg)  # std
            features[i, 9] = self._kurtosis(seg)  # kurtosis

            # --- Frecuencia (2) ---
            fft_vals = np.abs(np.fft.fft(seg))
            half = len(fft_vals) // 2
            features[i, 10] = np.argmax(fft_vals[:half])  # dominant_freq
            features[i, 11] = np.sum(fft_vals[:half] ** 2)  # spectral_energy

        return features

    @staticmethod
    def _estimate_qrs_duration(seg: np.ndarray, r_idx: int, fs: int) -> float:
        """Estima la duracion del complejo QRS en milisegundos.

        Busca los puntos donde la derivada cambia de signo alrededor del pico R.
        """
        derivative = np.diff(seg)
        search_range = int(0.06 * fs)  # ~60ms hacia cada lado

        # Buscar inicio del QRS (cambio de signo antes del R)
        start = r_idx
        for j in range(r_idx - 1, max(r_idx - search_range, 0), -1):
            if j < len(derivative) and abs(derivative[j]) < 0.1 * abs(derivative[r_idx - 1]):
                start = j
                break

        # Buscar fin del QRS (cambio de signo despues del R)
        end = r_idx
        for j in range(r_idx, min(r_idx + search_range, len(derivative))):
            if abs(derivative[j]) < 0.1 * abs(derivative[r_idx]):
                end = j
                break

        return (end - start) / fs * 1000.0  # ms

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Calcula kurtosis (excess kurtosis) sin depender de pandas."""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std < 1e-10:
            return 0.0
        m4 = np.mean((x - mean) ** 4)
        return m4 / (std ** 4) - 3.0
