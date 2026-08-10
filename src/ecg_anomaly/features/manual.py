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
    "rr_diff",
    "kurtosis",
    "dominant_freq_hz",
    "spectral_energy",
    "rr_post",
    "rr_ratio_pre_post",
    "rr_dev",
    # ==== Ventana temporal (Fase 2) — indices 12-17 ====
    "rr_mean_5",
    "rr_std_5",
    "rr_mean_10",
    "rr_std_10",
    "rmssd_5",
    "pnn_5",
]

# ==== Constantes de dimensiones (Fase 2) ====
# rr_ratio, mean, std y rr_pre se eliminaron de la base (eran variables
# duplicadas o constantes por construccion, ver historial de bugs corregidos
# en manual.py): 16 -> 12.
N_MANUAL_FEATURES_BASE = 12
N_MANUAL_FEATURES_WINDOW = 6
N_MANUAL_FEATURES_TOTAL = N_MANUAL_FEATURES_BASE + N_MANUAL_FEATURES_WINDOW

# Nombres de las 6 features de ventana temporal, en orden (indices 12-17).
# Fuente unica de verdad: no asumir el slice fijo features[:, 12:18] en
# notebooks/otros modulos, derivar los indices de FEATURE_NAMES.
WINDOW_FEATURE_NAMES: List[str] = FEATURE_NAMES[N_MANUAL_FEATURES_BASE:N_MANUAL_FEATURES_TOTAL]


class ManualFeatureExtractor:
    """Extractor Path B: caracteristicas manuales (18 features).

    Extrae features morfologicas, temporales (intervalos RR),
    estadisticas, de frecuencia y de ventana deslizante de cada latido
    segmentado.

    El pico R de cada segmento NO se asume en el centro geometrico: la
    ventana de segmentacion es asimetrica (`before_r_samples` antes del
    pico, `after_r_samples` despues, ver `SystemConfig`), asi que quien
    llama debe pasar explicitamente `before_r=config.before_r_samples` a
    `extract()` / `extract_raw()`. No se acepta un valor por defecto
    derivado de `segments.shape[1] // 2` porque con la configuracion real
    (90 antes + 110 despues) eso ubicaria el pico R 10 muestras despues
    del real, sobre la rama descendente del QRS.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int = 360,
        record_indices: np.ndarray | None = None,
        *,
        before_r: int | None = None,
    ) -> "ManualFeatureExtractor":
        """Calcula features crudas y ajusta el StandardScaler.

        Separado de transform (y de extract) para poder ajustar solo con
        un subconjunto de latidos (p. ej. normales de los registros DS1,
        ver `data/splitting.make_interpatient_split`) y evitar fuga de
        datos hacia la evaluacion, igual que `SignalPCAExtractor.fit()`.

        Args:
            segments: Array [N, beat_length] con latidos normalizados.
            r_peak_positions: Posiciones absolutas de los picos R en la senal original.
            fs: Frecuencia de muestreo.
            record_indices: Array [N] indicando el registro de cada latido.
            before_r: Indice del pico R dentro del segmento
                (`config.before_r_samples`). Obligatorio.

        Returns:
            self, para poder encadenar `.fit(...).transform(...)`.
        """
        features = self._extract_raw(
            segments, r_peak_positions, fs, record_indices, before_r=before_r
        )
        self.scaler.fit(features)
        self._is_fitted = True

        logger.info(
            "ManualFeatureExtractor ajustado: %d latidos x %d features", *features.shape
        )
        return self

    def transform(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int = 360,
        record_indices: np.ndarray | None = None,
        *,
        before_r: int | None = None,
    ) -> np.ndarray:
        """Aplica el StandardScaler ya ajustado a nuevos segmentos.

        Args:
            segments, r_peak_positions, fs, record_indices, before_r: igual
                que en `fit()`, calculados sobre los latidos a transformar
                (pueden ser distintos de los usados para ajustar el scaler).

        Returns:
            Array [N, 18] con features escaladas.

        Raises:
            RuntimeError: si el scaler todavia no fue ajustado.
        """
        if not self._is_fitted:
            raise RuntimeError("Debe llamar fit() (o extract()) antes de transform()")

        features = self._extract_raw(
            segments, r_peak_positions, fs, record_indices, before_r=before_r
        )
        scaled = self.scaler.transform(features)

        logger.info(
            "Features manuales (transform): %d latidos x %d features", *scaled.shape
        )
        return scaled

    def extract(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int = 360,
        record_indices: np.ndarray | None = None,
        *,
        before_r: int | None = None,
    ) -> np.ndarray:
        """Conveniencia: ajusta el scaler y transforma en un solo paso.

        Internamente llama a `fit()` y luego a `transform()` sobre el MISMO
        conjunto de entrada: por lo tanto ajusta el StandardScaler con
        TODOS los latidos que se le pasen, anomalias incluidas si las hay.
        Usar esta funcion sobre el dataset completo (o sobre cualquier
        conjunto que mezcle latidos de entrenamiento y de evaluacion)
        reproduce fuga de datos.

        Para evitarla, usar `fit()` solo con las filas de entrenamiento
        (p. ej. normales de los registros DS1, ver
        `data/splitting.make_interpatient_split`) y `transform()` sobre el
        resto — el mismo patron que `SignalPCAExtractor`.

        Args:
            segments: Array [N, beat_length] con latidos normalizados.
            r_peak_positions: Posiciones absolutas de los picos R en la senal original.
            fs: Frecuencia de muestreo.
            record_indices: Array [N] indicando el registro de cada latido.
            before_r: Indice del pico R dentro del segmento
                (`config.before_r_samples`). Obligatorio.

        Returns:
            Array [N, 18] con features escaladas (StandardScaler ajustado
            sobre este mismo `segments`).
        """
        self.fit(segments, r_peak_positions, fs, record_indices, before_r=before_r)
        return self.transform(segments, r_peak_positions, fs, record_indices, before_r=before_r)

    def extract_raw(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int = 360,
        record_indices: np.ndarray | None = None,
        *,
        before_r: int | None = None,
    ) -> np.ndarray:
        """API publica para obtener features crudas (sin escalar).

        Util para inspeccionar/visualizar features (p. ej. las de ventana
        temporal) sin pasar por el StandardScaler interno de extract().
        """
        return self._extract_raw(
            segments, r_peak_positions, fs, record_indices, before_r=before_r
        )

    def _extract_raw(
        self,
        segments: np.ndarray,
        r_peak_positions: np.ndarray,
        fs: int,
        record_indices: np.ndarray | None = None,
        *,
        before_r: int | None = None,
    ) -> np.ndarray:
        """Extrae features sin escalar."""
        if before_r is None:
            raise ValueError(
                "before_r es obligatorio (pasar config.before_r_samples). No se "
                "deriva de segments.shape[1] // 2: con before_r_samples=90 y "
                "after_r_samples=110 (ventana asimetrica de 200 muestras), "
                "asumir simetria ubicaria el pico R en el indice 100 en vez "
                "de 90, 10 muestras despues del real, sobre la rama "
                "descendente del QRS."
            )
        if not (0 <= before_r < segments.shape[1]):
            raise ValueError(
                f"before_r={before_r} fuera de rango para segmentos de "
                f"{segments.shape[1]} muestras."
            )

        n_beats = len(segments)
        r_idx = before_r

        rr_intervals = self._compute_rr_intervals(r_peak_positions, fs, record_indices)
        mean_rr_per_beat = self._compute_mean_rr_per_beat(rr_intervals, record_indices)

        features = np.zeros((n_beats, N_MANUAL_FEATURES_TOTAL))

        for i, seg in enumerate(segments):
            mean_rr = mean_rr_per_beat[i]

            features[i, 0] = seg[r_idx]
            s_region = seg[r_idx : min(r_idx + 30, len(seg))]
            features[i, 1] = np.min(s_region) if len(s_region) > 0 else 0.0
            features[i, 2] = self._estimate_qrs_duration(seg, r_idx, fs)
            features[i, 3] = np.max(seg) - np.min(seg)

            if i > 0 and rr_intervals[i] > 0:
                features[i, 4] = rr_intervals[i]
            else:
                features[i, 4] = mean_rr

            if i > 1 and rr_intervals[i] > 0 and rr_intervals[i - 1] > 0:
                features[i, 5] = rr_intervals[i] - rr_intervals[i - 1]
            else:
                features[i, 5] = 0.0

            features[i, 6] = self._kurtosis(seg)

            fft_vals = np.abs(np.fft.fft(seg))
            half = len(fft_vals) // 2
            freq_resolution = fs / len(seg)
            features[i, 7] = np.argmax(fft_vals[:half]) * freq_resolution
            features[i, 8] = np.sum(fft_vals[:half] ** 2)

            # rr_post: intervalo hacia el latido siguiente (mismo registro).
            # Fallback: features[i, 4] (rr_current, ya con su propio
            # fallback a mean_rr) en vez de una variable "rr_pre" separada
            # -- son el mismo valor, ver bug de features duplicadas.
            if i < n_beats - 1 and rr_intervals[i + 1] > 0:
                features[i, 9] = rr_intervals[i + 1]
            else:
                features[i, 9] = features[i, 4]

            if features[i, 4] > 0 and features[i, 9] > 0:
                features[i, 10] = features[i, 4] / features[i, 9]
            else:
                features[i, 10] = 1.0

            features[i, 11] = (
                abs(features[i, 4] - mean_rr) / mean_rr if mean_rr > 0 else 0.0
            )

        # ==== Ventana temporal (Fase 2) — indices 12-17 ====
        window_feat = self._extract_window_features(rr_intervals, n_beats, record_indices)
        features[:, N_MANUAL_FEATURES_BASE:N_MANUAL_FEATURES_TOTAL] = window_feat

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

    @staticmethod
    def _compute_mean_rr_per_beat(
        rr_intervals: np.ndarray, record_indices: np.ndarray | None = None
    ) -> np.ndarray:
        """Media de RR valido para cada latido, POR REGISTRO si es posible.

        Mezclar los 44 registros en una sola media global sesga rr_dev (y
        los fallbacks que dependen de ella): un paciente con bradicardia
        sostenida quedaria con todos sus latidos marcados como "desviados"
        de una media que no es la suya, y uno con taquicardia basal con
        ninguno.

        Args:
            rr_intervals: Array [N] de intervalos RR (0 = sin intervalo valido).
            record_indices: Array [N] con el registro de cada latido, o None
                para conservar el comportamiento global (una sola media
                sobre todo rr_intervals).

        Returns:
            Array [N]: la media correspondiente a cada latido (la de su
            propio registro, o la global si record_indices es None).
        """
        n = len(rr_intervals)

        if record_indices is None:
            valid_rr = rr_intervals[rr_intervals > 0]
            global_mean = np.mean(valid_rr) if len(valid_rr) > 0 else 800.0
            return np.full(n, global_mean)

        record_indices = np.asarray(record_indices)
        mean_per_beat = np.empty(n)
        for rec in np.unique(record_indices):
            mask = record_indices == rec
            valid_rr = rr_intervals[mask & (rr_intervals > 0)]
            rec_mean = np.mean(valid_rr) if len(valid_rr) > 0 else 800.0
            mean_per_beat[mask] = rec_mean

        return mean_per_beat
