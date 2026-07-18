"""Splits reproducibles para evitar fuga de datos (data leakage).

En deteccion de anomalias no supervisada sobre ECG hay DOS fugas posibles
y hay que cerrarlas las dos:

  1. Fuga de CLASE: ajustar el preprocesamiento con anomalias incluidas.
     Se cierra ajustando solo con latidos normales.

  2. Fuga de PACIENTE: que latidos del mismo registro caigan en fit y en
     eval. La morfologia del ECG es fuertemente individual, asi que un
     modelo que vio los normales del paciente X reconstruye trivialmente
     mas normales del paciente X. Se cierra particionando por REGISTRO.

La particion por registro sigue el paradigma inter-paciente de
de Chazal et al. (2004), estandar en la literatura de clasificacion de
latidos sobre MIT-BIH.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from ecg_anomaly.preprocessing.pipeline import PreprocessedData

# Particion canonica de de Chazal et al. (2004). Ambos conjuntos estan
# balanceados en tipos de arritmia y NO contienen los 4 registros con
# marcapasos (102, 104, 107, 217), que el pipeline ya excluye por
# recomendacion AAMI. 22 + 22 = 44 = los registros validos del sistema.
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]


def make_interpatient_split(
    preprocessed: PreprocessedData,
    record_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Split inter-paciente DS1/DS2 con ajuste solo-normal.

    Cierra las dos fugas a la vez:
      - fit_idx  = latidos NORMALES de los registros DS1 unicamente.
      - eval_idx = TODOS los latidos de los registros DS2 (normales y
                   anomalos), de pacientes que el modelo nunca vio.

    Args:
        preprocessed: Datos preprocesados con `labels` y `record_indices`.
        record_ids: Lista que mapea record_indices -> ID de registro, en
            el mismo orden usado por el pipeline de preprocesamiento
            (tipicamente `dataset.record_ids`).

    Returns:
        (fit_idx, eval_idx), disjuntos por construccion.
    """
    record_ids = np.asarray(record_ids)
    beat_record = record_ids[preprocessed.record_indices]

    in_ds1 = np.isin(beat_record, DS1_RECORDS)
    in_ds2 = np.isin(beat_record, DS2_RECORDS)

    # Aviso temprano si el mapeo no cuadra: es un error dificil de
    # diagnosticar despues, porque no lanza excepcion, solo da metricas raras.
    sin_asignar = int(np.sum(~(in_ds1 | in_ds2)))
    if sin_asignar > 0:
        raise ValueError(
            f"{sin_asignar:,} latidos no pertenecen ni a DS1 ni a DS2. "
            f"Registros no reconocidos: "
            f"{sorted(set(beat_record[~(in_ds1 | in_ds2)]))}"
        )

    # fit: solo normales de DS1 (cierra fuga de clase Y de paciente)
    fit_idx = np.where(in_ds1 & (preprocessed.labels == 0))[0]

    # eval: TODO DS2. Son pacientes distintos, asi que sus normales
    # tambien son informacion nueva y deben evaluarse.
    eval_idx = np.where(in_ds2)[0]

    assert len(np.intersect1d(fit_idx, eval_idx)) == 0, (
        "fit_idx y eval_idx se solapan: revisar el mapeo de registros"
    )
    return fit_idx, eval_idx


def make_normal_fit_split(
    preprocessed: PreprocessedData, seed: int = 42, val_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """[INTRA-PACIENTE - solo para comparacion con la literatura]

    Split aleatorio POR LATIDO. Cierra la fuga de clase pero NO la de
    paciente: latidos del mismo registro caen en fit y en eval, de modo
    que el modelo se evalua sobre pacientes cuya morfologia normal ya
    memorizo. Las metricas resultantes son optimistas.

    Se conserva unicamente para reportar ambos paradigmas en la tabla
    comparativa (ver notebook 05), que es lo que hace la literatura del
    area para dimensionar el sesgo. La conclusion de la tesis debe
    sustentarse en make_interpatient_split().

    Args:
        preprocessed: Datos preprocesados con `labels` (0=normal, 1=anomalo).
        seed: Semilla para el split reproducible.
        val_fraction: Fraccion de latidos normales reservada para evaluacion.

    Returns:
        fit_idx: subconjunto de latidos NORMALES para fit del scaler/PCA/AE.
        eval_idx: todo lo demas (normales restantes + todas las anomalias).
    """
    labels = preprocessed.labels
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    fit_idx, held_normal_idx = train_test_split(
        normal_idx, test_size=val_fraction, random_state=seed
    )
    eval_idx = np.concatenate([held_normal_idx, anomaly_idx])
    eval_idx.sort()
    return fit_idx, eval_idx
