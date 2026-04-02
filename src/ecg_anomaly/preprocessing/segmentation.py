"""Segmentacion de latidos individuales a partir de senales ECG continuas."""

from typing import Tuple

import numpy as np


def segment_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    before: int = 90,
    after: int = 110,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segmenta latidos individuales alrededor de los picos R.

    Extrae ventanas de tamano fijo (before + after muestras) centradas
    en cada pico R. Solo incluye latidos donde la ventana completa
    cabe dentro de la senal.

    Args:
        signal: Senal ECG continua 1D.
        r_peaks: Posiciones de los picos R (indices de muestras).
        before: Muestras antes del pico R (default: 90 = ~250ms a 360Hz).
        after: Muestras despues del pico R (default: 110 = ~305ms a 360Hz).

    Returns:
        Tupla de:
            - segments: Array [N, before+after] con los latidos segmentados.
            - valid_indices: Indices de r_peaks que produjeron segmentos validos.
    """
    beat_length = before + after
    segments = []
    valid_indices = []

    for i, peak in enumerate(r_peaks):
        start = peak - before
        end = peak + after

        if start < 0 or end > len(signal):
            continue

        segments.append(signal[start:end])
        valid_indices.append(i)

    if not segments:
        return np.empty((0, beat_length)), np.array([], dtype=int)

    return np.array(segments), np.array(valid_indices, dtype=int)


def normalize_beats(segments: np.ndarray) -> np.ndarray:
    """Normalizacion Z-score por latido individual.

    Cada latido se normaliza independientemente para tener
    media 0 y desviacion estandar 1.

    Args:
        segments: Array [N, beat_length] con latidos segmentados.

    Returns:
        Array [N, beat_length] con latidos normalizados.
    """
    means = segments.mean(axis=1, keepdims=True)
    stds = segments.std(axis=1, keepdims=True)
    # Evitar division por cero en senales constantes
    stds = np.where(stds < 1e-10, 1.0, stds)
    return (segments - means) / stds
