"""Path B: Extraccion manual de caracteristicas morfologicas y estadisticas."""

import logging
from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

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
    "rr_pre",
    "rr_post",
    "rr_ratio_pre_post",
    "rr_dev",
    # ==== Ventana temporal (Fase 2) — indices 16-21 ====
    "rr_mean_5",
    "rr_std_5",
    "rr_mean_10",
    "rr_std_10",
    "rmssd_5",
    "pnn_5",
]

# ==== Constantes de dimensiones (Fase 2) ====
N_MANUAL_FEATURES_BASE = 16
N_MANUAL_FEATURES_WINDOW = 6
N_MANUAL_FEATURES_TOTAL = N_MANUAL_FEATURES_BASE + N_MANUAL_FEATURES_WINDOW


class ManualFeatureExtractor:
    """Extractor Path B: caracteristicas manuales (~22 features).

    Extrae features morfologicas, temporales (intervalos RR),
    estadisticas, frecuencia y ventanas deslizantes de cada latido segmentado.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def extract(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int = 360,
        record_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extrae y escala features de los segmentos.

        Args:
            segments: Array [N, beat_length] con latidos normalizados.
            r_peak_positions: Posiciones absolutas de los picos R en la senal original.
            fs: Frecuencia de muestreo.
            record_indices: Array [N] indicando el registro de cada latido.

        Returns:
            Array [N, 22] con features escaladas (StandardScaler).
        """
        features = self._extract_raw(segments, r_peak_positions, fs, record_indices)
        scaled = self.scaler.fit_transform(features)
        self._is_fitted = True

        logger.info("Features manuales: %d latidos x %d features", *scaled.shape)
        return scaled

    def _extract_raw(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int,
        record_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extrae features sin escalar."""
        n_beats = len(segments)
        before_r = segments.shape[1] // 2

        rr_intervals = self._compute_rr_intervals(r_peak_positions, fs, record_indices)
        valid_rr = rr_intervals[rr_intervals > 0]
        mean_rr = np.mean(valid_rr) if len(valid_rr) > 0 else 800.0

        features = np.zeros((n_beats, N_MANUAL_FEATURES_TOTAL))

        for i, seg in enumerate(segments):
            r_idx = before_r

            features[i, 0] = seg[r_idx]
            s_region = seg[r_idx : min(r_idx + 30, len(seg))]
            features[i, 1] = np.min(s_region) if len(s_region) > 0 else 0.0
            features[i, 2] = self._estimate_qrs_duration(seg, r_idx, fs)
            features[i, 3] = np.max(seg) - np.min(seg)

            if i > 0 and rr_intervals[i] > 0:
                features[i, 4] = rr_intervals[i]
                features[i, 5] = rr_intervals[i] / mean_rr if mean_rr > 0 else 1.0
            else:
                features[i, 4] = mean_rr
                features[i, 5] = 1.0

            if i > 1 and rr_intervals[i] > 0 and rr_intervals[i - 1] > 0:
                features[i, 6] = rr_intervals[i] - rr_intervals[i - 1]
            else:
                features[i, 6] = 0.0

            features[i, 7] = np.mean(seg)
            features[i, 8] = np.std(seg)
            features[i, 9] = self._kurtosis(seg)

            fft_vals = np.abs(np.fft.fft(seg))
            half = len(fft_vals) // 2
            features[i, 10] = np.argmax(fft_vals[:half])
            features[i, 11] = np.sum(fft_vals[:half] ** 2)

            # Nuevas features RR-interval (indices 12-15)
            rr_val = rr_intervals[i] if rr_intervals[i] > 0 else mean_rr
            features[i, 12] = rr_val

            if i < n_beats - 1 and rr_intervals[i + 1] > 0:
                features[i, 13] = rr_intervals[i + 1]
            else:
                features[i, 13] = rr_val

            if features[i, 12] > 0 and features[i, 13] > 0:
                features[i, 14] = features[i, 12] / features[i, 13]
            else:
                features[i, 14] = 1.0

            features[i, 15] = abs(rr_val - mean_rr) / mean_rr if mean_rr > 0 else 0.0

        # ==== Ventana temporal (Fase 2) — indices 16-21 ====
        window_feat = self._extract_window_features(rr_intervals, n_beats, record_indices)
        features[:, 16:22] = window_feat

        return features

    @staticmethod
    def _estimate_qrs_duration(seg: np.ndarray, r_idx: int, fs: int) -> float:
        """Estima la duracion del complejo QRS en milisegundos."""
        derivative = np.diff(seg)
        search_range = int(0.06 * fs)

        start = r_idx
        for j in range(r_idx - 1, max(r_idx - search_range, 0), -1):
            if j < len(derivative) and abs(derivative[j]) < 0.1 * abs(derivative[r_idx - 1]):
                start = j
                break

        end = r_idx
        for j in range(r_idx, min(r_idx + search_range, len(derivative))):
            if abs(derivative[j]) < 0.1 * abs(derivative[r_idx]):
                end = j
                break

        return (end - start) / fs * 1000.0

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Calcula excess kurtosis sin pandas."""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std < 1e-10:
            return 0.0
        m4 = np.mean((x - mean) ** 4)
        return m4 / (std ** 4) - 3.0

    # ==== Inicio: Ventana temporal (Fase 2) ====
    @staticmethod
    def _extract_window_features(
        rr_intervals: np.ndarray,
        n_beats: int,
        record_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Features de ventana deslizante sobre intervalos RR.

        Agregado en Fase 2 del plan de mejoras.
        Ventanas de 5 y 10 latidos para detectar patrones
        de arritmias (bigeminia, taquicardia ventricular, FA).

        Cada ventana solo incluye latidos del MISMO registro
        para evitar contaminacion entre registros distintos.
        Si la ventana tiene < 2 beats, las features quedan en 0.
        """
        window_features = np.zeros((n_beats, N_MANUAL_FEATURES_WINDOW))

        for i in range(n_beats):
            # Limitar ventana al mismo registro para evitar contaminacion
            if record_indices is not None:
                rec = record_indices[i]
                idx_5 = [
                    j
                    for j in range(max(0, i - 4), i + 1)
                    if record_indices[j] == rec
                ]
                idx_10 = [
                    j
                    for j in range(max(0, i - 9), i + 1)
                    if record_indices[j] == rec
                ]
            else:
                idx_5 = list(range(max(0, i - 4), i + 1))
                idx_10 = list(range(max(0, i - 9), i + 1))

            window_5 = rr_intervals[idx_5]
            window_10 = rr_intervals[idx_10]

            if len(window_5) >= 2:
                window_features[i, 0] = np.mean(window_5)
                window_features[i, 1] = np.std(window_5)

            if len(window_10) >= 2:
                window_features[i, 2] = np.mean(window_10)
                window_features[i, 3] = np.std(window_10)

            if len(window_5) >= 2:
                diffs = np.diff(window_5)
                window_features[i, 4] = np.sqrt(np.mean(diffs**2))

            if len(window_5) >= 2:
                mean_local = np.mean(window_5[:-1])
                if mean_local > 0:
                    abnormal = np.sum(
                        np.abs(window_5[:-1] - mean_local) > 0.2 * mean_local
                    )
                    window_features[i, 5] = abnormal / len(window_5[:-1])

        return window_features
    # ==== Fin: Ventana temporal (Fase 2) ====

    @staticmethod
    def _compute_rr_intervals(
        r_peak_positions: np.ndarray, fs: int, record_indices: np.ndarray | None = None
    ) -> np.ndarray:
        """Calcula intervalos RR hacia el latido anterior del mismo registro."""
        n = len(r_peak_positions)
        rr = np.zeros(n)

        if record_indices is None:
            diffs = np.diff(r_peak_positions) / fs * 1000.0
            rr[1:] = diffs
            return rr

        record_indices = np.asarray(record_indices)
        for i in range(1, n):
            if record_indices[i] == record_indices[i - 1]:
                rr[i] = (r_peak_positions[i] - r_peak_positions[i - 1]) / fs * 1000.0

        return rr