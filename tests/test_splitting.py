"""Tests para los splits reproducibles (fuga de clase y de paciente)."""

import numpy as np
import pytest

from ecg_anomaly.data.splitting import (
    DS1_RECORDS,
    DS2_RECORDS,
    make_interpatient_split,
    make_normal_fit_split,
)
from ecg_anomaly.preprocessing.pipeline import PreprocessedData


def _make_preprocessed(record_ids, beats_per_record=20, anomaly_fraction=0.2, seed=0):
    rng = np.random.RandomState(seed)
    n_records = len(record_ids)
    n_beats = n_records * beats_per_record

    segments = rng.randn(n_beats, 5)
    record_indices = np.repeat(np.arange(n_records), beats_per_record)

    labels = np.zeros(n_beats, dtype=int)
    n_anom = int(n_beats * anomaly_fraction)
    anom_pos = rng.choice(n_beats, size=n_anom, replace=False)
    labels[anom_pos] = 1

    return PreprocessedData(
        segments=segments,
        labels=labels,
        r_peaks_flat=np.arange(n_beats),
        record_indices=record_indices,
    )


class TestMakeInterpatientSplit:
    def test_no_beat_overlap(self):
        record_ids = DS1_RECORDS[:5] + DS2_RECORDS[:5]
        preprocessed = _make_preprocessed(record_ids)
        fit_idx, eval_idx = make_interpatient_split(preprocessed, record_ids)
        assert len(np.intersect1d(fit_idx, eval_idx)) == 0

    def test_no_patient_overlap(self):
        record_ids = DS1_RECORDS[:5] + DS2_RECORDS[:5]
        preprocessed = _make_preprocessed(record_ids)
        fit_idx, eval_idx = make_interpatient_split(preprocessed, record_ids)

        rec = np.asarray(record_ids)[preprocessed.record_indices]
        pac_fit = set(rec[fit_idx])
        pac_eval = set(rec[eval_idx])
        assert not (pac_fit & pac_eval)
        assert pac_fit.issubset(set(DS1_RECORDS))
        assert pac_eval.issubset(set(DS2_RECORDS))

    def test_fit_is_all_normal(self):
        record_ids = DS1_RECORDS[:5] + DS2_RECORDS[:5]
        preprocessed = _make_preprocessed(record_ids)
        fit_idx, _ = make_interpatient_split(preprocessed, record_ids)
        assert np.all(preprocessed.labels[fit_idx] == 0)

    def test_eval_has_both_classes(self):
        record_ids = DS1_RECORDS[:5] + DS2_RECORDS[:5]
        preprocessed = _make_preprocessed(record_ids)
        _, eval_idx = make_interpatient_split(preprocessed, record_ids)
        assert set(preprocessed.labels[eval_idx].tolist()) == {0, 1}

    def test_unrecognized_record_raises(self):
        record_ids = ["999"] + DS1_RECORDS[:2]
        preprocessed = _make_preprocessed(record_ids)
        with pytest.raises(ValueError, match="no pertenecen"):
            make_interpatient_split(preprocessed, record_ids)


class TestMakeNormalFitSplit:
    def test_no_overlap_and_fit_is_normal(self):
        record_ids = DS1_RECORDS[:3]
        preprocessed = _make_preprocessed(record_ids)
        fit_idx, eval_idx = make_normal_fit_split(preprocessed, seed=42)
        assert len(np.intersect1d(fit_idx, eval_idx)) == 0
        assert np.all(preprocessed.labels[fit_idx] == 0)
