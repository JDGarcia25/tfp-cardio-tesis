"""Carga de datos ECG desde MIT-BIH Arrhythmia Database."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
try:
    import wfdb
except ModuleNotFoundError:
    wfdb = None
    logging.warning("wfdb module not installed. ECG loading will not be available.")

from ecg_anomaly.config import SystemConfig
from ecg_anomaly.data.registry import RecordRegistry

logger = logging.getLogger(__name__)


@dataclass
class ECGRecord:
    """Datos de un registro ECG individual."""

    record_id: str
    signal: np.ndarray  # Senal continua (canal MLII)
    r_peak_positions: np.ndarray  # Posiciones de picos R (muestras)
    symbols: np.ndarray  # Simbolos de anotacion por latido
    binary_labels: np.ndarray  # 0=normal, 1=anomalo, filtrado a latidos validos
    sampling_rate: int = 360


@dataclass
class ECGDataset:
    """Conjunto completo de datos ECG cargados."""

    records: List[ECGRecord] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def total_beats(self) -> int:
        return sum(len(r.binary_labels) for r in self.records)

    @property
    def total_normal(self) -> int:
        return sum(int(np.sum(r.binary_labels == 0)) for r in self.records)

    @property
    def total_anomalous(self) -> int:
        return sum(int(np.sum(r.binary_labels == 1)) for r in self.records)

    @property
    def record_ids(self) -> List[str]:
        return [r.record_id for r in self.records]


class MITBIHLoader:
    """Cargador de datos MIT-BIH Arrhythmia Database.

    Soporta carga desde archivos locales o descarga de PhysioNet.
    Aplica agrupacion AAMI para clasificacion binaria (normal/anomalo).
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.registry = RecordRegistry()

    def load(
        self,
        path: str,
        records: Optional[List[str]] = None,
        channel: int = 0,
    ) -> ECGDataset:
        """Carga registros MIT-BIH.

        Args:
            path: Directorio con archivos .dat/.hea/.atr o None para PhysioNet.
            records: Lista de IDs de registros. None usa todos los validos (44).
            channel: Indice del canal a usar (0=MLII por defecto).

        Returns:
            ECGDataset con todos los registros cargados.
        """
        if records is None:
            records = self.registry.get_valid_records(set(self.config.excluded_records))

        dataset = ECGDataset(
            metadata={
                "dataset": "MIT-BIH Arrhythmia Database",
                "sampling_rate": self.config.sampling_rate,
                "channel": channel,
                "aami_grouping": True,
            }
        )

        for record_id in records:
            try:
                ecg_record = self._load_record(path, record_id, channel)
                if ecg_record is not None:
                    dataset.records.append(ecg_record)
            except Exception as e:
                logger.warning("Error cargando registro %s: %s", record_id, e)
                continue

        dataset.metadata["records_loaded"] = len(dataset.records)
        dataset.metadata["total_beats"] = dataset.total_beats
        dataset.metadata["total_normal"] = dataset.total_normal
        dataset.metadata["total_anomalous"] = dataset.total_anomalous

        logger.info(
            "Cargados %d registros: %d latidos (%d normal, %d anomalo)",
            len(dataset.records),
            dataset.total_beats,
            dataset.total_normal,
            dataset.total_anomalous,
        )

        return dataset

    def _load_record(
        self, path: str, record_id: str, channel: int
    ) -> Optional[ECGRecord]:
        """Carga un registro individual."""
        import os

        record_path = os.path.join(path, record_id)

        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")

        signal = record.p_signal[:, channel]

        # Filtrar solo simbolos que representan latidos
        beat_mask = np.array(
            [RecordRegistry.is_beat_symbol(s) for s in annotation.symbol]
        )
        beat_positions = annotation.sample[beat_mask]
        beat_symbols = np.array(annotation.symbol)[beat_mask]

        # Clasificar con AAMI
        binary_labels = np.array(
            [RecordRegistry.classify_symbol(s) for s in beat_symbols]
        )

        # Filtrar simbolos no clasificables (label == -1)
        valid_mask = binary_labels >= 0
        beat_positions = beat_positions[valid_mask]
        beat_symbols = beat_symbols[valid_mask]
        binary_labels = binary_labels[valid_mask]

        if len(beat_positions) == 0:
            logger.warning("Registro %s: sin latidos validos AAMI", record_id)
            return None

        logger.debug(
            "Registro %s: %d latidos (N=%d, A=%d)",
            record_id,
            len(binary_labels),
            int(np.sum(binary_labels == 0)),
            int(np.sum(binary_labels == 1)),
        )

        return ECGRecord(
            record_id=record_id,
            signal=signal,
            r_peak_positions=beat_positions,
            symbols=beat_symbols,
            binary_labels=binary_labels,
            sampling_rate=record.fs,
        )
