"""Metricas extrinsecas de deteccion de anomalias (con ground truth AAMI).

Responden a: ¿Las anomalias detectadas coinciden con las anotaciones clinicas?

Nota: Aunque el entrenamiento se realiza sin utilizar etiquetas (no supervisado),
la evaluacion requiere un criterio de referencia. Se utilizan las anotaciones de
MIT-BIH como ground truth. Este enfoque es estandar en la literatura de deteccion
no supervisada de anomalias.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate_extrinsic(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> Dict[str, float]:
    """Calcula metricas extrinsecas de deteccion de anomalias.

    Args:
        true_labels: Etiquetas reales AAMI (0=normal, 1=anomalo).
        pred_labels: Etiquetas predichas (0=normal, 1=anomalia).

    Returns:
        Dict con accuracy, sensitivity, specificity, precision, f1, auc_roc,
        y conteos de la matriz de confusion (TP, FP, TN, FN).
    """
    # Asegurar tipos correctos
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)

    # Matriz de confusion
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Metricas derivadas
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall anomalia
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Recall normal
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    # AUC-ROC (requiere ambas clases presentes)
    try:
        auc = float(roc_auc_score(true_labels, pred_labels))
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc_roc": auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
