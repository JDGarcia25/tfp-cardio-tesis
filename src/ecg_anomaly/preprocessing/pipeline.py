"""Pipeline completo de preprocesamiento de senales ECG."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ecg_anomaly.config import SystemConfig
from ecg_anomaly.data.loader import ECGDataset
from ecg_anomaly.preprocessing.filters import butterworth_bandpass
from ecg_anomaly.preprocessing.segmentation import normalize_beats, segment_beats

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """Resultado del preprocesamiento."""

    segments: np.ndarray  # [N, beat_length] latidos normalizados
    labels: np.ndarray  # [N] etiquetas binarias AAMI (0=normal, 1=anomalo)
    r_peaks_flat: np.ndarray  # [N] posiciones R originales (para features RR)
    record_indices: np.ndarray  # [N] indice del registro de origen
    metadata: dict = field(default_factory=dict)


class PreprocessingPipeline:
    """Pipeline de preprocesamiento: filtrado -> segmentacion -> normalizacion.

    Para registros MIT-BIH, usa las posiciones R de las anotaciones como
    ground truth (no re-detecta QRS). El filtrado se aplica a la senal
    continua antes de segmentar.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    def run(self, dataset: ECGDataset) -> PreprocessedData:
        """Ejecuta el pipeline completo sobre un dataset.

        Args:
            dataset: ECGDataset cargado con registros MIT-BIH.

        Returns:
            PreprocessedData con latidos segmentados y normalizados.
        """
        all_segments = []
        all_labels = []
        all_r_peaks = []
        all_record_indices = []

        for idx, record in enumerate(dataset.records):
            # 1. Filtrado pasa-banda de la senal continua
            filtered = butterworth_bandpass(
                record.signal,
                lowcut=self.config.filter_lowcut,
                highcut=self.config.filter_highcut,
                fs=self.config.sampling_rate,
                order=self.config.filter_order,
            )

            # 2. Segmentacion usando posiciones R de las anotaciones
            segments, valid_idx = segment_beats(
                filtered,
                record.r_peak_positions,
                before=self.config.before_r_samples,
                after=self.config.after_r_samples,
            )

            if len(segments) == 0:
                logger.warning("Registro %s: sin segmentos validos", record.record_id)
                continue

            # 3. Normalizacion Z-score por latido
            segments = normalize_beats(segments)

            # 4. Filtrar etiquetas correspondientes a segmentos validos
            labels = record.binary_labels[valid_idx]
            r_peaks = record.r_peak_positions[valid_idx]

            all_segments.append(segments)
            all_labels.append(labels)
            all_r_peaks.append(r_peaks)
            all_record_indices.append(np.full(len(segments), idx))

            logger.debug(
                "Registro %s: %d segmentos (%d normal, %d anomalo)",
                record.record_id,
                len(segments),
                int(np.sum(labels == 0)),
                int(np.sum(labels == 1)),
            )

        segments = np.vstack(all_segments)
        labels = np.concatenate(all_labels)
        r_peaks = np.concatenate(all_r_peaks)
        record_indices = np.concatenate(all_record_indices)

        logger.info(
            "Preprocesamiento completo: %d latidos [%d, %d] (normal=%d, anomalo=%d)",
            segments.shape[0],
            segments.shape[0],
            segments.shape[1],
            int(np.sum(labels == 0)),
            int(np.sum(labels == 1)),
        )

        return PreprocessedData(
            segments=segments,
            labels=labels,
            r_peaks_flat=r_peaks,
            record_indices=record_indices,
            metadata={
                "n_beats": segments.shape[0],
                "beat_length": segments.shape[1],
                "n_normal": int(np.sum(labels == 0)),
                "n_anomalous": int(np.sum(labels == 1)),
                "n_records": len(dataset.records),
            },
        )
