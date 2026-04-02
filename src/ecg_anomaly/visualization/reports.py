"""Generacion de reportes comparativos y visualizaciones finales."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_metrics_comparison(
    results_df: pd.DataFrame,
    title: str = "Comparacion de Metricas por Modelo",
    save_path: Optional[str] = None,
) -> None:
    """Grafico de barras agrupadas comparando metricas entre modelos."""
    # Seleccionar metricas numéricas relevantes
    metric_cols = [c for c in results_df.columns if c not in ["Modelo", "params"]]
    numeric_df = results_df[["Modelo"] + metric_cols].copy()

    # Excluir columnas con todos None
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # Separar en 3 paneles: intrinsecas, extrinsecas, eficiencia
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Intrinsecas
    intrinsic_cols = ["Silhouette"]
    _plot_bar_group(axes[0], numeric_df, intrinsic_cols, "Metricas Intrinsecas")

    # Panel 2: Extrinsecas
    extrinsic_cols = [c for c in ["Accuracy", "Sensitivity", "Specificity", "F1", "AUC-ROC"] if c in numeric_df.columns]
    _plot_bar_group(axes[1], numeric_df, extrinsic_cols, "Metricas Extrinsecas")

    # Panel 3: Eficiencia
    efficiency_cols = [c for c in ["Tiempo (s)", "Memoria (MB)"] if c in numeric_df.columns]
    _plot_bar_group(axes[2], numeric_df, efficiency_cols, "Eficiencia Computacional")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def _plot_bar_group(
    ax: plt.Axes,
    df: pd.DataFrame,
    columns: List[str],
    title: str,
) -> None:
    """Grafico de barras agrupadas en un eje."""
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        ax.set_title(f"{title}\n(sin datos)")
        return

    plot_data = df[["Modelo"] + valid_cols].set_index("Modelo")
    plot_data = plot_data.apply(pd.to_numeric, errors="coerce")
    plot_data.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylabel("Valor")


def plot_confusion_matrices(
    results: List[Dict],
    save_path: Optional[str] = None,
) -> None:
    """Matrices de confusion para todos los modelos."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        model_name = result["model"]
        tp = result.get("extrinsic_true_positives", 0)
        fp = result.get("extrinsic_false_positives", 0)
        tn = result.get("extrinsic_true_negatives", 0)
        fn = result.get("extrinsic_false_negatives", 0)

        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Anomalia"],
            yticklabels=["Normal", "Anomalia"],
            ax=ax,
        )
        ax.set_title(f"{model_name.upper()}")
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicho")

    plt.suptitle("Matrices de Confusion", fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_full_report(
    results_df: pd.DataFrame,
    raw_results: List[Dict],
    output_dir: str,
) -> str:
    """Guarda reporte completo: tabla CSV + graficos.

    Returns:
        Ruta del directorio con los resultados.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_dir) / f"report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Tabla CSV
    csv_path = report_dir / "comparison_table.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Tabla guardada en: %s", csv_path)

    # Graficos
    plot_metrics_comparison(
        results_df,
        save_path=str(report_dir / "metrics_comparison.png"),
    )

    if raw_results:
        plot_confusion_matrices(
            raw_results,
            save_path=str(report_dir / "confusion_matrices.png"),
        )

    logger.info("Reporte completo guardado en: %s", report_dir)
    return str(report_dir)
