"""Algoritmos de deteccion de complejos QRS en senales ECG."""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def pan_tompkins(signal: np.ndarray, fs: int = 360) -> np.ndarray:
    """Algoritmo Pan-Tompkins para deteccion de picos R.

    Implementa el pipeline clasico: filtro pasa-banda (5-15 Hz),
    derivada, elevacion al cuadrado, integracion por ventana movil,
    y deteccion de picos con refinamiento.

    Referencia: Pan & Tompkins (1985), IEEE Trans. Biomed. Eng.

    Args:
        signal: Senal ECG 1D cruda o filtrada.
        fs: Frecuencia de muestreo en Hz.

    Returns:
        Array con indices de muestras de los picos R detectados.
    """
    nyquist = 0.5 * fs

    # 1. Filtro pasa-banda 5-15 Hz
    low = 5.0 / nyquist
    high = 15.0 / nyquist
    b, a = butter(2, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)

    # 2. Derivada (diferencia finita)
    diff = np.diff(filtered)

    # 3. Elevacion al cuadrado
    squared = diff ** 2

    # 4. Integracion por ventana movil (~150ms)
    window_size = int(0.150 * fs)
    kernel = np.ones(window_size) / window_size
    integrated = np.convolve(squared, kernel, mode="same")

    # 5. Deteccion de picos con distancia minima ~200ms entre latidos
    min_distance = int(0.2 * fs)
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, _ = find_peaks(integrated, distance=min_distance, height=threshold)

    # 6. Refinamiento: buscar el maximo real de la senal original
    #    en una ventana de +/- 50ms alrededor de cada pico detectado
    refined_peaks = []
    search_window = int(0.05 * fs)
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(signal), peak + search_window)
        local_max = start + np.argmax(signal[start:end])
        refined_peaks.append(local_max)

    return np.array(refined_peaks, dtype=int)


def xqrs_detect(signal: np.ndarray, fs: int = 360) -> np.ndarray:
    """Deteccion de picos R usando XQRS de wfdb.

    Metodo mas robusto que Pan-Tompkins para senales con ruido.

    Args:
        signal: Senal ECG 1D.
        fs: Frecuencia de muestreo.

    Returns:
        Array con indices de muestras de los picos R.
    """
    import wfdb.processing

    xqrs = wfdb.processing.XQRS(sig=signal, fs=fs)
    xqrs.detect()
    return np.array(xqrs.qrs_inds, dtype=int)
