"""Filtros digitales para preprocesamiento de senales ECG."""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def butterworth_bandpass(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 4,
) -> np.ndarray:
    """Filtro pasa-banda Butterworth.

    Elimina ruido de linea base (< lowcut Hz) y ruido de alta frecuencia
    (> highcut Hz) preservando las componentes relevantes del ECG.

    Args:
        signal: Senal ECG (1D o 2D donde cada fila es una senal).
        lowcut: Frecuencia de corte inferior en Hz.
        highcut: Frecuencia de corte superior en Hz.
        fs: Frecuencia de muestreo en Hz.
        order: Orden del filtro Butterworth.

    Returns:
        Senal filtrada con las mismas dimensiones.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")

    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    return np.array([filtfilt(b, a, row) for row in signal])


def notch_filter(
    signal: np.ndarray,
    freq: float = 60.0,
    fs: int = 360,
    quality: float = 30.0,
) -> np.ndarray:
    """Filtro notch para eliminar interferencia de linea electrica.

    Args:
        signal: Senal ECG 1D.
        freq: Frecuencia a eliminar (50 Hz o 60 Hz).
        fs: Frecuencia de muestreo.
        quality: Factor de calidad Q del filtro.

    Returns:
        Senal sin la componente de frecuencia especificada.
    """
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)
